"""
forex_backtester.py — Forex Backtesting Engine
================================================
Simulates forex trading with pip-based P&L, spread costs,
swap/rollover simulation, leverage tracking, and position sizing.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ForexBacktester:
    """
    Backtesting engine for forex trading strategies.

    Features:
        - Pip-based P&L calculation
        - Spread cost simulation
        - Swap/rollover cost simulation
        - Leverage and margin tracking
        - ATR-based stop/take-profit
        - Trailing stops
        - Position sizing per risk rules
        - Comprehensive trade log and metrics
    """

    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        bt_cfg = self.config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 500.0)
        self.spread_pips = bt_cfg.get("spread_pips", 1.5)
        self.slippage_pips = bt_cfg.get("slippage_pips", 0.5)
        self.swap_per_day_pips = bt_cfg.get("swap_per_day_pips", -0.5)
        self.leverage = bt_cfg.get("leverage", 10)
        self.enable_shorts = bt_cfg.get("enable_shorts", True)
        self.fixed_lots = bt_cfg.get("fixed_lots", None)

        risk_cfg = self.config.get("risk", {})
        self.sl_atr_mult = risk_cfg.get("stop_loss_atr_mult", 1.5)
        self.tp_atr_mult = risk_cfg.get("take_profit_atr_mult", 2.5)
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 0.8) / 100
        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade_pct", 1.0) / 100
        self.enable_breakeven_stop = risk_cfg.get("enable_breakeven_stop", True)
        self.breakeven_atr_mult = risk_cfg.get("breakeven_atr_mult", 1.0)

        self.report_dir = Path(self.config.get("logging", {}).get("report_dir", "reports/forex/"))
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.cooldown_bars = risk_cfg.get("cooldown_bars", 6)
        self.rsi_max_buy = risk_cfg.get("rsi_max_buy", 65.0)
        self.rsi_min_sell = risk_cfg.get("rsi_min_sell", 35.0)
        self.min_stop_loss_pips = risk_cfg.get("min_stop_loss_pips", 12.0)
        self.max_stop_loss_pips = risk_cfg.get("max_stop_loss_pips", 35.0)
        self.min_take_profit_pips = risk_cfg.get("min_take_profit_pips", 25.0)
        self.enable_compounding = risk_cfg.get("enable_compounding", True)

        regime_cfg = self.config.get("regime", {})
        self.enable_regime_filter = regime_cfg.get("enabled", True)
        self.min_adx = regime_cfg.get("trend_strength_min_adx", 22)
        self.trend_filter_ema = regime_cfg.get("trend_filter_ema", 200)

    def get_pip_size(self, pair: str) -> float:
        clean = pair.replace("=X", "").replace("/", "").upper()
        return 0.01 if clean in self.JPY_PAIRS else 0.0001

    def run(
        self, pair: str, prices: pd.DataFrame, signals: pd.DataFrame,
        features: pd.DataFrame = None,
    ) -> dict:
        """
        Run backtest simulation.

        Args:
            pair: Forex pair name
            prices: DataFrame with OHLCV data
            signals: DataFrame with signal, confidence columns
            features: Optional features DataFrame for confirmation checks

        Returns:
            Dict with metrics, trade log, and equity curve
        """
        pip_size = self.get_pip_size(pair)

        # Pip value lookup
        pip_values = self.config.get("backtest", {}).get("pip_values", {})
        clean_pair = pair.replace("=X", "").replace("/", "").upper()
        pip_value_per_lot = pip_values.get(clean_pair, 10.0)  # Per standard lot

        capital = self.initial_capital
        equity_curve = [capital]
        trade_log = []
        position = None
        max_drawdown = 0
        peak_equity = capital
        bars_in_position = 0
        last_sl_bar = -999
        last_sl_dir = 0

        for i in range(len(prices)):
            idx = prices.index[i]
            price = float(prices["Close"].iloc[i])
            high = float(prices["High"].iloc[i])
            low = float(prices["Low"].iloc[i])

            if idx not in signals.index:
                equity_curve.append(capital)
                continue

            signal = int(signals.loc[idx, "signal"])

            # Check existing position
            if position is not None:
                bars_in_position += 1
                direction = position["direction"]

                # Update high/low watermark
                if direction == 1:
                    position["high_watermark"] = max(position["high_watermark"], high)
                else:
                    position["low_watermark"] = min(position["low_watermark"], low)

                # Move stop to breakeven after advancing in favor by breakeven_atr_mult * ATR
                if self.enable_breakeven_stop and not position.get("is_breakeven", False):
                    pos_atr = position.get("atr", 0.003)
                    be_dist = pos_atr * self.breakeven_atr_mult
                    if direction == 1 and (position["high_watermark"] - position["entry_price"]) >= be_dist:
                        position["stop_loss"] = position["entry_price"] + self.spread_pips * pip_size
                        position["is_breakeven"] = True
                    elif direction == -1 and (position["entry_price"] - position["low_watermark"]) >= be_dist:
                        position["stop_loss"] = position["entry_price"] - self.spread_pips * pip_size
                        position["is_breakeven"] = True

                # Check stop loss
                hit_sl = False
                hit_tp = False
                hit_trail = False

                if direction == 1:
                    if low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        hit_sl = True
                    elif high >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        hit_tp = True
                    else:
                        # Trailing stop (only active once trail is in profit)
                        trail = position["high_watermark"] * (1 - self.trailing_stop_pct)
                        if trail > position["entry_price"] and low <= trail:
                            exit_price = trail
                            hit_trail = True
                else:  # Short
                    if high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        hit_sl = True
                    elif low <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        hit_tp = True
                    else:
                        # Trailing stop (only active once trail is in profit)
                        trail = position["low_watermark"] * (1 + self.trailing_stop_pct)
                        if trail < position["entry_price"] and high >= trail:
                            exit_price = trail
                            hit_trail = True

                if hit_sl or hit_tp or hit_trail:
                    # Calculate P&L
                    pnl_price = (exit_price - position["entry_price"]) * direction
                    pnl_pips = pnl_price / pip_size
                    pnl_usd = pnl_pips * pip_value_per_lot * (position["units"] / 100000)

                    # Subtract spread cost
                    spread_cost = self.spread_pips * pip_value_per_lot * (position["units"] / 100000)
                    pnl_usd -= spread_cost

                    # Subtract swap cost for overnight positions (rough: 1 swap per 24 bars for H1)
                    bars_per_day = 24
                    overnight_days = max(0, bars_in_position // bars_per_day)
                    swap_cost = abs(self.swap_per_day_pips) * overnight_days * pip_value_per_lot * (position["units"] / 100000)
                    pnl_usd -= swap_cost

                    capital += pnl_usd

                    if hit_sl:
                        last_sl_bar = i
                        last_sl_dir = direction

                    reason = "SL" if hit_sl else ("TP" if hit_tp else "Trail")
                    trade_log.append({
                        "pair": clean_pair,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "entry_price": position["entry_price"],
                        "exit_price": round(exit_price, 5),
                        "units": position["units"],
                        "pnl_pips": round(pnl_pips, 1),
                        "pnl_usd": round(pnl_usd, 2),
                        "spread_cost": round(spread_cost, 2),
                        "swap_cost": round(swap_cost, 2),
                        "bars_held": bars_in_position,
                        "reason": reason,
                        "entry_date": str(position["entry_date"]),
                        "exit_date": str(idx),
                    })

                    position = None
                    bars_in_position = 0

                # Signal-based exit (opposite signal)
                elif position is not None and signal != 0 and signal != direction:
                    exit_price = price
                    pnl_price = (exit_price - position["entry_price"]) * direction
                    pnl_pips = pnl_price / pip_size
                    pnl_usd = pnl_pips * pip_value_per_lot * (position["units"] / 100000)
                    spread_cost = self.spread_pips * pip_value_per_lot * (position["units"] / 100000)
                    pnl_usd -= spread_cost
                    bars_per_day = 24
                    overnight_days = max(0, bars_in_position // bars_per_day)
                    swap_cost = abs(self.swap_per_day_pips) * overnight_days * pip_value_per_lot * (position["units"] / 100000)
                    pnl_usd -= swap_cost
                    capital += pnl_usd

                    trade_log.append({
                        "pair": clean_pair,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "entry_price": position["entry_price"],
                        "exit_price": round(exit_price, 5),
                        "units": position["units"],
                        "pnl_pips": round(pnl_pips, 1),
                        "pnl_usd": round(pnl_usd, 2),
                        "spread_cost": round(spread_cost, 2),
                        "swap_cost": round(swap_cost, 2),
                        "bars_held": bars_in_position,
                        "reason": "Signal flip",
                        "entry_date": str(position["entry_date"]),
                        "exit_date": str(idx),
                    })

                    position = None
                    bars_in_position = 0

            # Open new position
            if position is None and signal != 0:
                if signal == -1 and not self.enable_shorts:
                    equity_curve.append(capital)
                    continue

                # 0. Cooldown after Stop Loss: prevent repeated entries during adverse swings
                if (i - last_sl_bar) < self.cooldown_bars and signal == last_sl_dir:
                    equity_curve.append(capital)
                    continue

                # Regime / Trend filters if features are available
                if self.enable_regime_filter and features is not None and idx in features.index:
                    # 1. ADX filter: skip entering during choppy/flat consolidation
                    if "adx" in features.columns and self.min_adx > 0:
                        cur_adx = float(features.loc[idx, "adx"])
                        if cur_adx < self.min_adx:
                            equity_curve.append(capital)
                            continue

                    # 2. Trend filter (e.g. 200 EMA): only BUY above EMA, only SELL below EMA
                    if self.trend_filter_ema:
                        ema_col = f"ema_{self.trend_filter_ema}"
                        if ema_col in features.columns:
                            ema_val = float(features.loc[idx, ema_col])
                            if signal == 1 and price < ema_val:
                                equity_curve.append(capital)
                                continue
                            elif signal == -1 and price > ema_val:
                                equity_curve.append(capital)
                                continue
                        elif "close_vs_ema_200" in features.columns:
                            close_vs_ema = float(features.loc[idx, "close_vs_ema_200"])
                            if signal == 1 and close_vs_ema < 0:
                                equity_curve.append(capital)
                                continue
                            elif signal == -1 and close_vs_ema > 0:
                                equity_curve.append(capital)
                                continue

                    # 3. RSI pullback filter: avoid buying overbought or shorting oversold
                    if "rsi_14" in features.columns:
                        cur_rsi = float(features.loc[idx, "rsi_14"])
                        if signal == 1 and cur_rsi > self.rsi_max_buy:
                            equity_curve.append(capital)
                            continue
                        elif signal == -1 and cur_rsi < self.rsi_min_sell:
                            equity_curve.append(capital)
                            continue

                direction = signal

                # ATR for stop/target
                if features is not None and "atr_14" in features.columns and idx in features.index:
                    atr = float(features.loc[idx, "atr_14"])
                else:
                    # Estimate ATR from recent bars
                    lookback = min(14, i)
                    if lookback > 0:
                        recent = prices.iloc[max(0, i-lookback):i+1]
                        tr = pd.concat([
                            recent["High"] - recent["Low"],
                            abs(recent["High"] - recent["Close"].shift(1)),
                            abs(recent["Low"] - recent["Close"].shift(1)),
                        ], axis=1).max(axis=1)
                        atr = float(tr.mean())
                    else:
                        atr = price * 0.003

                # Entry with slippage
                slippage = self.slippage_pips * pip_size
                entry_price = price + slippage * direction

                # Stop/target with minimum pip floors and maximum risk cap
                min_sl_dist = self.min_stop_loss_pips * pip_size
                min_tp_dist = self.min_take_profit_pips * pip_size
                max_sl_dist = self.max_stop_loss_pips * pip_size
                sl_dist = min(max_sl_dist, max(atr * self.sl_atr_mult, min_sl_dist))
                tp_dist = max(atr * self.tp_atr_mult, min_tp_dist)

                if direction == 1:
                    stop_loss = entry_price - sl_dist
                    take_profit = entry_price + tp_dist
                else:
                    stop_loss = entry_price + sl_dist
                    take_profit = entry_price - tp_dist

                # Position sizing (fixed lots with dynamic compounding, or risk-based)
                if self.fixed_lots is not None and self.fixed_lots > 0:
                    current_lots = self.fixed_lots
                    if self.enable_compounding and capital > 0:
                        if capital >= 110.0:
                            current_lots = max(self.fixed_lots, 0.04)
                        elif capital >= 75.0:
                            current_lots = max(self.fixed_lots, 0.03)
                    units = int(round(current_lots * 100000))
                else:
                    sl_pips = sl_dist / pip_size
                    risk_usd = capital * self.max_risk_per_trade
                    pip_value_for_sizing = pip_value_per_lot / 100000  # Per unit
                    if sl_pips > 0 and pip_value_for_sizing > 0:
                        units = int(risk_usd / (sl_pips * pip_value_for_sizing))
                    else:
                        units = 1000

                    # Leverage constraint
                    max_units = int(capital * self.leverage / entry_price)
                    units = min(units, max_units)

                    # Round to micro lots
                    units = max(1000, (units // 1000) * 1000)

                position = {
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "units": units,
                    "direction": direction,
                    "high_watermark": high,
                    "low_watermark": low,
                    "entry_date": idx,
                    "atr": atr,
                    "is_breakeven": False,
                }
                bars_in_position = 0

            # Track equity
            equity_curve.append(capital)
            peak_equity = max(peak_equity, capital)
            drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Close any remaining position at last price
        if position is not None:
            exit_price = float(prices["Close"].iloc[-1])
            direction = position["direction"]
            pnl_price = (exit_price - position["entry_price"]) * direction
            pnl_pips = pnl_price / pip_size
            pnl_usd = pnl_pips * pip_value_per_lot * (position["units"] / 100000)
            spread_cost = self.spread_pips * pip_value_per_lot * (position["units"] / 100000)
            pnl_usd -= spread_cost
            capital += pnl_usd

            trade_log.append({
                "pair": clean_pair, "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": position["entry_price"],
                "exit_price": round(exit_price, 5), "units": position["units"],
                "pnl_pips": round(pnl_pips, 1), "pnl_usd": round(pnl_usd, 2),
                "spread_cost": round(spread_cost, 2), "swap_cost": 0,
                "bars_held": bars_in_position, "reason": "End of backtest",
                "entry_date": str(position["entry_date"]),
                "exit_date": str(prices.index[-1]),
            })

        # Compute metrics
        metrics = self._compute_metrics(trade_log, equity_curve, capital)

        # Print report
        self._print_report(pair, metrics, trade_log)

        # Save report
        self._save_report(pair, metrics, trade_log, equity_curve)

        return {"metrics": metrics, "trades": trade_log, "equity_curve": equity_curve}

    def _compute_metrics(self, trade_log: list, equity_curve: list, final_capital: float) -> dict:
        """Compute comprehensive backtest metrics."""
        if not trade_log:
            return {
                "total_trades": 0,
                "long_trades": 0,
                "short_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "net_pnl_usd": 0.0,
                "net_pnl_pips": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win_usd": 0.0,
                "avg_loss_usd": 0.0,
                "avg_pnl_pips": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_bars_held": 0.0,
                "total_spread_cost": 0.0,
                "total_swap_cost": 0.0,
                "initial_capital": self.initial_capital,
                "final_capital": round(final_capital, 2),
                "leverage": self.leverage,
            }

        pnl_list = [t["pnl_usd"] for t in trade_log]
        pip_list = [t["pnl_pips"] for t in trade_log]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0

        # Max drawdown from equity curve
        peak = self.initial_capital
        max_dd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        long_trades = [t for t in trade_log if t["direction"] == "LONG"]
        short_trades = [t for t in trade_log if t["direction"] == "SHORT"]

        avg_bars = np.mean([t["bars_held"] for t in trade_log]) if trade_log else 0
        total_spread = sum(t.get("spread_cost", 0) for t in trade_log)
        total_swap = sum(t.get("swap_cost", 0) for t in trade_log)

        return {
            "total_trades": len(trade_log),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(trade_log) * 100, 1) if trade_log else 0,
            "net_pnl_usd": round(sum(pnl_list), 2),
            "net_pnl_pips": round(sum(pip_list), 1),
            "total_return_pct": round((final_capital / self.initial_capital - 1) * 100, 2),
            "profit_factor": round(total_profit / total_loss, 2) if total_loss > 0 else float("inf"),
            "avg_win_usd": round(np.mean(wins), 2) if wins else 0,
            "avg_loss_usd": round(np.mean(losses), 2) if losses else 0,
            "avg_pnl_pips": round(np.mean(pip_list), 1) if pip_list else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_bars_held": round(avg_bars, 1),
            "total_spread_cost": round(total_spread, 2),
            "total_swap_cost": round(total_swap, 2),
            "initial_capital": self.initial_capital,
            "final_capital": round(final_capital, 2),
            "leverage": self.leverage,
        }

    def _print_report(self, pair: str, metrics: dict, trade_log: list):
        """Print backtest results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FOREX BACKTEST REPORT — {pair}")
        logger.info(f"{'='*60}")
        logger.info(f"  Capital: ${metrics['initial_capital']:.2f} → ${metrics['final_capital']:.2f}")
        logger.info(f"  Return: {metrics['total_return_pct']:+.2f}%")
        logger.info(f"  Net P&L: ${metrics['net_pnl_usd']:+.2f} ({metrics['net_pnl_pips']:+.1f} pips)")
        logger.info(f"  Trades: {metrics['total_trades']} (L:{metrics['long_trades']}, S:{metrics['short_trades']})")
        logger.info(f"  Win rate: {metrics['win_rate']}%")
        logger.info(f"  Profit factor: {metrics['profit_factor']}")
        logger.info(f"  Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Avg bars held: {metrics['avg_bars_held']}")
        logger.info(f"  Spread cost: ${metrics['total_spread_cost']:.2f}")
        logger.info(f"  Swap cost: ${metrics['total_swap_cost']:.2f}")
        logger.info(f"  Leverage: {metrics['leverage']}:1")
        if self.fixed_lots:
            logger.info(f"  Lot size: {self.fixed_lots:.2f} lots (fixed)")
        logger.info(f"{'='*60}")

        if trade_log:
            logger.info(f"\n📋 Last 10 trades:")
            for t in trade_log[-10:]:
                logger.info(f"  {t['direction']:5} | {t['entry_price']:.5f} → {t['exit_price']:.5f} | "
                            f"{t['pnl_pips']:+6.1f} pips (${t['pnl_usd']:+.2f}) | "
                            f"{t['bars_held']} bars | {t['reason']}")

    def _save_report(self, pair: str, metrics: dict, trade_log: list, equity_curve: list):
        """Save backtest report to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Trade log
        if trade_log:
            df = pd.DataFrame(trade_log)
            path = self.report_dir / f"backtest_{pair}_{timestamp}.csv"
            df.to_csv(path, index=False)
            logger.info(f"📁 Report saved: {path}")

        # Equity curve
        eq_path = self.report_dir / f"equity_{pair}_{timestamp}.csv"
        pd.DataFrame({"equity": equity_curve}).to_csv(eq_path, index=False)
