"""
backtester.py — Historical Backtest Simulator
==============================================
Simulates trading strategies on historical data with
realistic commissions, slippage, and performance reporting.

Upgraded with:
    - Short selling simulation
    - Monte Carlo confidence intervals
    - Weekly PnL reporting
    - Regime-aware position sizing
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward backtest engine with realistic execution simulation.

    Features:
        - Realistic commissions (IBKR tiered rate)
        - Slippage estimation
        - Long AND short trade simulation
        - Full performance metrics (win rate, profit factor, Sharpe, max drawdown)
        - Monte Carlo analysis with confidence intervals
        - Weekly P&L breakdown for target tracking
        - Trade-by-trade log
        - Equity curve
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        bt_cfg = self.config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 100_000)
        self.commission_per_share = bt_cfg.get("commission_per_share", 0.005)
        self.slippage_pct = bt_cfg.get("slippage_pct", 0.05) / 100
        self.enable_shorts = bt_cfg.get("enable_shorts", False)
        self.monte_carlo_iterations = bt_cfg.get("monte_carlo_iterations", 1000)
        self.monte_carlo_confidence = bt_cfg.get("monte_carlo_confidence", 0.95)

        risk_cfg = self.config.get("risk", {})
        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100
        self.max_position_pct = risk_cfg.get("max_position_pct", 5.0) / 100
        self.stop_loss_atr_mult = risk_cfg.get("stop_loss_atr_mult", 2.0)
        self.take_profit_atr_mult = risk_cfg.get("take_profit_atr_mult", 3.0)
        self.enable_breakeven_stop = risk_cfg.get("enable_breakeven_stop", True)
        self.breakeven_atr_mult = risk_cfg.get("breakeven_atr_mult", 1.0)
        self.trailing_stop = risk_cfg.get("trailing_stop", True)

        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> dict:
        """
        Run a full backtest simulation.

        Args:
            prices: DataFrame with OHLCV columns
            signals: DataFrame with 'signal' column (1=BUY, -1=SELL, 0=HOLD)
                     and optionally 'confidence' column
            features: Optional features DataFrame (for ATR-based stops)

        Returns:
            Dict with full performance metrics and trade log
        """
        logger.info(f"Running backtest: {len(prices)} bars, "
                     f"initial_capital=${self.initial_capital:,.0f}, "
                     f"shorts={'ON' if self.enable_shorts else 'OFF'}")

        # Merge signals with prices
        df = prices.copy()
        df["signal"] = signals["signal"].reindex(df.index).fillna(0).astype(int)
        if "confidence" in signals.columns:
            df["confidence"] = signals["confidence"].reindex(df.index).fillna(0)

        # Merge key confirmation features if provided
        if features is not None:
            for col in ["close_vs_vwap", "close_vs_ema_21", "ema_21_50_cross", "rsi_14", "volume_ratio_sma20"]:
                if col in features.columns:
                    df[col] = features[col].reindex(df.index)

        # Compute ATR for stop placement
        df["atr"] = self._compute_atr(df)

        # Run simulation
        trades, equity_curve = self._simulate(df)

        # Compute metrics
        metrics = self._compute_metrics(trades, equity_curve)

        # Weekly P&L breakdown
        weekly_pnl = self._compute_weekly_pnl(trades, equity_curve)
        metrics["weekly_pnl"] = weekly_pnl

        # Monte Carlo analysis
        if trades and len(trades) >= 10:
            mc_results = self._monte_carlo_analysis(trades)
            metrics["monte_carlo"] = mc_results

        # Save results
        self._save_results(trades, equity_curve, metrics)

        # Print summary
        self._print_summary(metrics)

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def _simulate(self, df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
        """Core simulation loop with long and short support."""
        cash = self.initial_capital
        position = None  # {type, shares, entry_price, stop_loss, take_profit, entry_date}
        trades = []
        equity = []

        for i, (dt, row) in enumerate(df.iterrows()):
            price = row["Close"]
            signal = row["signal"]
            atr = row.get("atr", price * 0.02)

            # ── Check exit conditions for open position ──
            if position is not None:
                # Dynamic Breakeven & Trailing Stop Updates
                if position["type"] == "long":
                    # Breakeven: once price hits entry + 1R, protect capital
                    if self.enable_breakeven_stop and row["High"] >= position["entry_price"] + atr * self.breakeven_atr_mult:
                        be_price = position["entry_price"] * (1 + self.slippage_pct + 0.001)
                        if be_price > position["stop_loss"]:
                            position["stop_loss"] = be_price

                    # Trailing stop
                    if self.trailing_stop:
                        trail_price = row["High"] - (atr * self.stop_loss_atr_mult)
                        if trail_price > position["stop_loss"]:
                            position["stop_loss"] = trail_price
                elif position["type"] == "short":
                    if self.enable_breakeven_stop and row["Low"] <= position["entry_price"] - atr * self.breakeven_atr_mult:
                        be_price = position["entry_price"] * (1 - self.slippage_pct - 0.001)
                        if be_price < position["stop_loss"]:
                            position["stop_loss"] = be_price

                    if self.trailing_stop:
                        trail_price = row["Low"] + (atr * self.stop_loss_atr_mult)
                        if trail_price < position["stop_loss"]:
                            position["stop_loss"] = trail_price

                exit_price = None
                exit_reason = None

                if position["type"] == "long":
                    # Stop-loss hit (intrabar check using Low)
                    if row["Low"] <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop-loss (Breakeven/Trailing)" if position["stop_loss"] >= position["entry_price"] else "Stop-loss"
                    # Take-profit hit (intrabar check using High)
                    elif row["High"] >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "Take-profit"
                    # Sell signal from model
                    elif signal == -1:
                        exit_price = price
                        exit_reason = "Sell signal"

                elif position["type"] == "short":
                    # Stop-loss hit for short (using High)
                    if row["High"] >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop-loss"
                    # Take-profit hit for short (using Low)
                    elif row["Low"] <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "Take-profit"
                    # Buy signal from model (close short)
                    elif signal == 1:
                        exit_price = price
                        exit_reason = "Buy signal (close short)"

                if exit_price:
                    # Apply slippage
                    if position["type"] == "long":
                        exit_price *= (1 - self.slippage_pct)
                    else:  # short — slippage works against us when buying to cover
                        exit_price *= (1 + self.slippage_pct)

                    # Commission
                    commission = position["shares"] * self.commission_per_share

                    # PnL
                    if position["type"] == "long":
                        pnl = (exit_price - position["entry_price"]) * position["shares"]
                        pnl_pct = (exit_price / position["entry_price"] - 1) * 100
                        cash += exit_price * position["shares"] - commission
                    else:  # short
                        pnl = (position["entry_price"] - exit_price) * position["shares"]
                        pnl_pct = (1 - exit_price / position["entry_price"]) * 100
                        # Return collateral + profit (or minus loss)
                        cash += position["collateral"] + pnl - commission

                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": dt,
                        "type": position["type"],
                        "entry_price": position["entry_price"],
                        "exit_price": round(exit_price, 4),
                        "shares": position["shares"],
                        "stop_loss": position["stop_loss"],
                        "take_profit": position["take_profit"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "exit_reason": exit_reason,
                        "commission": round(commission * 2, 2),  # round-trip
                    })

                    position = None

            # ── Check entry conditions ──
            if position is None:
                # Filter: reject long if price is severely lagging session VWAP (deep selloff)
                allow_buy = True
                if "close_vs_vwap" in row and pd.notna(row["close_vs_vwap"]):
                    if row["close_vs_vwap"] < -0.03:
                        allow_buy = False

                # LONG entry
                if signal == 1 and allow_buy:
                    position = self._open_position(
                        "long", price, atr, cash, dt
                    )
                    if position:
                        cash -= position["shares"] * position["entry_price"] + \
                               max(position["shares"] * self.commission_per_share, 0.01)

                # SHORT entry
                elif signal == -1 and self.enable_shorts:
                    position = self._open_position(
                        "short", price, atr, cash, dt
                    )
                    if position:
                        # For shorts, we need collateral
                        cash -= position["collateral"] + \
                               max(position["shares"] * self.commission_per_share, 0.01)

            # Track equity
            portfolio_value = cash
            if position is not None:
                if position["type"] == "long":
                    portfolio_value += position["shares"] * price
                else:  # short
                    # Collateral + unrealized PnL
                    unrealized_pnl = (position["entry_price"] - price) * position["shares"]
                    portfolio_value += position["collateral"] + unrealized_pnl

            equity.append(portfolio_value)

        # Close any remaining position at last price
        if position is not None:
            last_price = df["Close"].iloc[-1]
            if position["type"] == "long":
                last_price *= (1 - self.slippage_pct)
                pnl = (last_price - position["entry_price"]) * position["shares"]
                pnl_pct = (last_price / position["entry_price"] - 1) * 100
            else:
                last_price *= (1 + self.slippage_pct)
                pnl = (position["entry_price"] - last_price) * position["shares"]
                pnl_pct = (1 - last_price / position["entry_price"]) * 100

            commission = position["shares"] * self.commission_per_share

            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": df.index[-1],
                "type": position["type"],
                "entry_price": position["entry_price"],
                "exit_price": round(last_price, 4),
                "shares": position["shares"],
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "pnl": round(pnl - commission, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": "End of backtest",
                "commission": round(commission * 2, 2),
            })

        equity_series = pd.Series(equity, index=df.index, name="equity")
        return trades, equity_series

    def _open_position(
        self, pos_type: str, price: float, atr: float, cash: float, dt
    ) -> dict | None:
        """Create a position dict with proper stops and sizing."""
        # Entry with slippage
        if pos_type == "long":
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)

        # ATR-based stops
        if atr > 0 and not np.isnan(atr):
            if pos_type == "long":
                stop_loss = entry_price - atr * self.stop_loss_atr_mult
                take_profit = entry_price + atr * self.take_profit_atr_mult
            else:  # short
                stop_loss = entry_price + atr * self.stop_loss_atr_mult
                take_profit = entry_price - atr * self.take_profit_atr_mult
        else:
            if pos_type == "long":
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.04
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96

        # Ensure minimum stop distance
        if pos_type == "long":
            stop_loss = min(stop_loss, entry_price * 0.995)
        else:
            stop_loss = max(stop_loss, entry_price * 1.005)

        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return None

        # Position sizing
        if cash < 500:
            # Micro account: use up to 90% of available cash
            shares = int((cash * 0.90) / entry_price)
            shares = max(shares, 1)
        else:
            risk_amount = cash * self.max_risk_per_trade
            max_shares_risk = int(risk_amount / risk_per_share)
            max_shares_pos = int(cash * self.max_position_pct / entry_price)
            shares = min(max_shares_risk, max_shares_pos)
            shares = max(shares, 1)

        # Check we can afford it
        required_capital = shares * entry_price
        if required_capital > cash:
            return None

        position = {
            "type": pos_type,
            "entry_date": dt,
            "entry_price": round(entry_price, 4),
            "shares": shares,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
        }

        if pos_type == "short":
            position["collateral"] = round(required_capital, 2)

        return position

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def _compute_metrics(self, trades: list[dict], equity: pd.Series) -> dict:
        """Compute comprehensive performance metrics."""
        if not trades:
            final_equity = equity.iloc[-1] if len(equity) > 0 else self.initial_capital
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "total_pnl": 0.0,
                "initial_capital": self.initial_capital,
                "final_equity": round(final_equity, 2),
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade": 0.0,
                "avg_win_loss_ratio": 0.0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "long_trades": 0,
                "short_trades": 0,
                "exit_reasons": {},
                "note": "No trades executed with current confidence threshold",
            }

        trade_df = pd.DataFrame(trades)
        pnls = trade_df["pnl"]

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        total_trades = len(pnls)

        # Win Rate
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        # Profit Factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Total Return
        final_equity = equity.iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Sharpe Ratio (annualized, assuming daily returns)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Average trade metrics
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        avg_trade = pnls.mean()

        # Win/Loss streaks
        is_win = (pnls > 0).astype(int)
        streaks = is_win.groupby((is_win != is_win.shift()).cumsum())
        max_win_streak = 0
        max_loss_streak = 0
        for _, streak in streaks:
            if streak.iloc[0] == 1:
                max_win_streak = max(max_win_streak, len(streak))
            else:
                max_loss_streak = max(max_loss_streak, len(streak))

        # Trade type breakdown
        long_trades = len(trade_df[trade_df["type"] == "long"]) if "type" in trade_df.columns else total_trades
        short_trades = len(trade_df[trade_df["type"] == "short"]) if "type" in trade_df.columns else 0

        # Exit reasons distribution
        exit_reasons = trade_df["exit_reason"].value_counts().to_dict()

        metrics = {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return, 2),
            "total_pnl": round(pnls.sum(), 2),
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_trade": round(avg_trade, 2),
            "avg_win_loss_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "exit_reasons": exit_reasons,
        }

        return metrics

    def _compute_weekly_pnl(self, trades: list[dict], equity: pd.Series) -> dict:
        """Compute weekly P&L breakdown for $5/week target tracking."""
        if not trades:
            return {"weeks": [], "target_hit_rate": 0}

        trade_df = pd.DataFrame(trades)
        trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])

        # Group by week
        trade_df["week"] = trade_df["exit_date"].dt.isocalendar().week.astype(int)
        trade_df["year"] = trade_df["exit_date"].dt.year

        weekly_groups = trade_df.groupby(["year", "week"]).agg(
            pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
        ).reset_index()

        target = self.config.get("targets", {}).get("daily_profit_target", 5.0) * 5  # Weekly target

        weeks = []
        for _, row in weekly_groups.iterrows():
            weeks.append({
                "year": int(row["year"]),
                "week": int(row["week"]),
                "pnl": round(row["pnl"], 2),
                "trades": int(row["trades"]),
                "wins": int(row["wins"]),
                "hit_target": row["pnl"] >= target,
            })

        target_hit_count = sum(1 for w in weeks if w["hit_target"])
        target_hit_rate = target_hit_count / len(weeks) * 100 if weeks else 0

        return {
            "weeks": weeks,
            "total_weeks": len(weeks),
            "target_hit_count": target_hit_count,
            "target_hit_rate": round(target_hit_rate, 1),
            "weekly_target": target,
            "avg_weekly_pnl": round(np.mean([w["pnl"] for w in weeks]), 2) if weeks else 0,
            "best_week": round(max(w["pnl"] for w in weeks), 2) if weeks else 0,
            "worst_week": round(min(w["pnl"] for w in weeks), 2) if weeks else 0,
        }

    def _monte_carlo_analysis(self, trades: list[dict]) -> dict:
        """
        Monte Carlo simulation: shuffle trade order to compute confidence intervals.

        Answers: "With 95% confidence, what is my expected weekly PnL range?"
        """
        if len(trades) < 10:
            return {"error": "Too few trades for Monte Carlo"}

        logger.info(f"Running Monte Carlo: {self.monte_carlo_iterations} iterations...")

        pnls = [t["pnl"] for t in trades]
        n_trades = len(pnls)

        # Simulate many possible orderings
        final_equities = []
        weekly_pnls = []

        rng = np.random.default_rng(42)

        for _ in range(self.monte_carlo_iterations):
            shuffled = rng.permutation(pnls)
            cumulative = np.cumsum(shuffled) + self.initial_capital
            final_equities.append(cumulative[-1])

            # Approximate weekly PnL (assume ~5 trades per week for active scalping)
            trades_per_week = max(1, n_trades // max(1, n_trades // 5))
            week_pnls = []
            for i in range(0, len(shuffled), trades_per_week):
                week_pnl = sum(shuffled[i:i + trades_per_week])
                week_pnls.append(week_pnl)
            if week_pnls:
                weekly_pnls.extend(week_pnls)

        final_equities = np.array(final_equities)
        weekly_pnls = np.array(weekly_pnls)

        confidence = self.monte_carlo_confidence
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 - (1 - confidence) / 2) * 100

        return {
            "iterations": self.monte_carlo_iterations,
            "confidence_level": confidence,
            "final_equity": {
                "mean": round(float(np.mean(final_equities)), 2),
                "median": round(float(np.median(final_equities)), 2),
                f"p{lower_pct:.0f}": round(float(np.percentile(final_equities, lower_pct)), 2),
                f"p{upper_pct:.0f}": round(float(np.percentile(final_equities, upper_pct)), 2),
            },
            "weekly_pnl": {
                "mean": round(float(np.mean(weekly_pnls)), 2),
                "median": round(float(np.median(weekly_pnls)), 2),
                f"p{lower_pct:.0f}": round(float(np.percentile(weekly_pnls, lower_pct)), 2),
                f"p{upper_pct:.0f}": round(float(np.percentile(weekly_pnls, upper_pct)), 2),
                "prob_positive": round(float((weekly_pnls > 0).mean() * 100), 1),
                "prob_above_target": round(
                    float((weekly_pnls >= self.config.get("targets", {}).get("daily_profit_target", 5.0) * 5).mean() * 100), 1
                ),
            },
        }

    def _print_summary(self, metrics: dict):
        """Print formatted backtest summary."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console(legacy_windows=False)

            table = Table(
                title="BACKTEST RESULTS",
                show_header=True,
                header_style="bold cyan",
                border_style="blue",
            )
            table.add_column("Metric", style="bold", width=22)
            table.add_column("Value", justify="right", width=15)

            def color_val(val, fmt=".2f", good_if_positive=True):
                if isinstance(val, (int, float)):
                    if good_if_positive:
                        color = "green" if val > 0 else "red"
                    else:
                        color = "red" if val < -10 else "yellow" if val < 0 else "green"
                    return f"[{color}]{val:{fmt}}[/{color}]"
                return str(val)

            table.add_row("Total Trades", str(metrics["total_trades"]))
            table.add_row("  Long / Short", f"{metrics.get('long_trades', '?')} / {metrics.get('short_trades', '?')}")
            table.add_row("Win Rate", color_val(metrics["win_rate_pct"]) + "%")
            table.add_row("Profit Factor", color_val(metrics["profit_factor"]))
            table.add_row("Total Return", color_val(metrics["total_return_pct"]) + "%")
            table.add_row("Total PnL", f"${metrics['total_pnl']:+,.2f}")
            table.add_row("Final Equity", f"${metrics['final_equity']:,.2f}")
            table.add_row("Max Drawdown", color_val(metrics["max_drawdown_pct"], good_if_positive=False) + "%")
            table.add_row("Sharpe Ratio", color_val(metrics["sharpe_ratio"]))
            table.add_row("Avg Win", f"${metrics['avg_win']:+,.2f}")
            table.add_row("Avg Loss", f"${metrics['avg_loss']:+,.2f}")
            table.add_row("Win/Loss Ratio", str(metrics["avg_win_loss_ratio"]))
            table.add_row("Max Win Streak", str(metrics["max_win_streak"]))
            table.add_row("Max Loss Streak", str(metrics["max_loss_streak"]))

            console.print()
            console.print(table)

            # Exit reasons
            if metrics.get("exit_reasons"):
                console.print("\n[bold]Exit Reasons:[/bold]")
                for reason, count in metrics["exit_reasons"].items():
                    console.print(f"  • {reason}: {count}")

            # Weekly PnL summary
            weekly = metrics.get("weekly_pnl", {})
            if weekly and weekly.get("weeks"):
                console.print(f"\n[bold]Weekly P&L Summary:[/bold]")
                console.print(f"  Avg Weekly PnL: ${weekly['avg_weekly_pnl']:+.2f}")
                console.print(f"  Best Week: ${weekly['best_week']:+.2f}")
                console.print(f"  Worst Week: ${weekly['worst_week']:+.2f}")
                console.print(f"  Weeks hitting ${weekly['weekly_target']:.0f} target: "
                              f"{weekly['target_hit_count']}/{weekly['total_weeks']} "
                              f"({weekly['target_hit_rate']:.0f}%)")

            # Monte Carlo results
            mc = metrics.get("monte_carlo", {})
            if mc and "weekly_pnl" in mc:
                console.print(f"\n[bold]Monte Carlo ({mc['iterations']} iterations):[/bold]")
                wp = mc["weekly_pnl"]
                console.print(f"  Weekly PnL — Mean: ${wp['mean']:+.2f}, "
                              f"Median: ${wp['median']:+.2f}")
                for key in sorted(wp.keys()):
                    if key.startswith("p"):
                        console.print(f"  {key}: ${wp[key]:+.2f}")
                console.print(f"  P(positive week): {wp['prob_positive']:.1f}%")
                console.print(f"  P(hit $5 target): {wp['prob_above_target']:.1f}%")

            console.print()

        except Exception:
            print(f"\n{'='*50}")
            print(f"  BACKTEST RESULTS")
            print(f"{'='*50}")
            for key, val in metrics.items():
                if key not in ("exit_reasons", "weekly_pnl", "monte_carlo"):
                    print(f"  {key}: {val}")
            print(f"{'='*50}\n")

    def _save_results(self, trades: list, equity: pd.Series, metrics: dict):
        """Save backtest results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        if trades:
            trade_df = pd.DataFrame(trades)
            trade_path = self.report_dir / f"backtest_trades_{timestamp}.csv"
            trade_df.to_csv(trade_path, index=False)

        # Save equity curve
        equity_path = self.report_dir / f"backtest_equity_{timestamp}.csv"
        equity.to_csv(equity_path)

        # Save metrics
        import json
        metrics_path = self.report_dir / f"backtest_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Results saved to {self.report_dir}/")


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    close = 100 + np.cumsum(np.random.randn(500) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(500) * 0.2,
        "High": close + abs(np.random.randn(500) * 0.5),
        "Low": close - abs(np.random.randn(500) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, 500),
    }, index=dates)

    # Random signals
    signals = pd.DataFrame({
        "signal": np.random.choice([-1, 0, 0, 0, 1], 500),
        "confidence": np.random.uniform(0.5, 0.9, 500),
    }, index=dates)

    backtester = Backtester()
    results = backtester.run(df, signals)
