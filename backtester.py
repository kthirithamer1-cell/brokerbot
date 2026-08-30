"""
backtester.py — Historical Backtest Simulator
==============================================
Simulates trading strategies on historical data with
realistic commissions, slippage, and performance reporting.
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
        - Full performance metrics (win rate, profit factor, Sharpe, max drawdown)
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

        risk_cfg = self.config.get("risk", {})
        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100
        self.max_position_pct = risk_cfg.get("max_position_pct", 5.0) / 100
        self.stop_loss_atr_mult = risk_cfg.get("stop_loss_atr_mult", 2.0)
        self.take_profit_atr_mult = risk_cfg.get("take_profit_atr_mult", 3.0)

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
                     f"initial_capital=${self.initial_capital:,.0f}")

        # Merge signals with prices
        df = prices.copy()
        df["signal"] = signals["signal"].reindex(df.index).fillna(0).astype(int)
        if "confidence" in signals.columns:
            df["confidence"] = signals["confidence"].reindex(df.index).fillna(0)

        # Compute ATR for stop placement
        df["atr"] = self._compute_atr(df)

        # Run simulation
        trades, equity_curve = self._simulate(df)

        # Compute metrics
        metrics = self._compute_metrics(trades, equity_curve)

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
        """Core simulation loop."""
        cash = self.initial_capital
        position = None  # {symbol, shares, entry_price, stop_loss, take_profit}
        trades = []
        equity = []

        for i, (dt, row) in enumerate(df.iterrows()):
            price = row["Close"]
            signal = row["signal"]
            atr = row.get("atr", price * 0.02)

            # ── Check exit conditions for open position ──
            if position is not None:
                exit_price = None
                exit_reason = None

                # Stop-loss hit (intrabar check using Low)
                if row["Low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "Stop-loss"

                # Take-profit hit (intrabar check using High)
                elif row["High"] >= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "Take-profit"

                # Sell signal from model
                elif signal == -1:
                    exit_price = price
                    exit_reason = "Sell signal"

                if exit_price:
                    # Apply slippage
                    exit_price *= (1 - self.slippage_pct)

                    # Commission
                    commission = position["shares"] * self.commission_per_share

                    # PnL
                    pnl = (exit_price - position["entry_price"]) * position["shares"]
                    pnl -= commission  # entry commission already deducted

                    cash += exit_price * position["shares"] - commission

                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": dt,
                        "entry_price": position["entry_price"],
                        "exit_price": round(exit_price, 4),
                        "shares": position["shares"],
                        "stop_loss": position["stop_loss"],
                        "take_profit": position["take_profit"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(
                            (exit_price / position["entry_price"] - 1) * 100, 2
                        ),
                        "exit_reason": exit_reason,
                        "commission": round(commission * 2, 2),  # round-trip
                    })

                    position = None

            # ── Check entry conditions ──
            if position is None and signal == 1:
                # Entry with slippage
                entry_price = price * (1 + self.slippage_pct)

                # Position sizing with Hard Loss Cap (Max -8% loss per trade)
                hard_stop = entry_price * 0.92  # Capped at -8% max loss
                atr_stop = entry_price - atr * self.stop_loss_atr_mult
                stop_loss = max(atr_stop, hard_stop)  # Whichever is tighter

                # Target at least 2x the risk (minimum +16% gain)
                risk_pct = (entry_price - stop_loss) / entry_price
                min_target = entry_price * (1 + max(risk_pct * 2.0, 0.16))
                take_profit = max(entry_price + atr * self.take_profit_atr_mult, min_target)

                risk_per_share = entry_price - stop_loss

                if risk_per_share > 0:
                    if cash < 500:
                        # Micro account ($30 - $500): use up to 90% of available cash on 1 trade
                        shares = int((cash * 0.90) / entry_price)
                        shares = max(shares, 1)
                    else:
                        risk_amount = cash * self.max_risk_per_trade
                        max_shares_risk = int(risk_amount / risk_per_share)
                        max_shares_pos = int(cash * self.max_position_pct / entry_price)
                        shares = min(max_shares_risk, max_shares_pos)
                        shares = max(shares, 1)

                    if shares * entry_price <= cash:
                        commission = max(shares * self.commission_per_share, 0.01)
                        cash -= shares * entry_price + commission

                        position = {
                            "entry_date": dt,
                            "entry_price": round(entry_price, 4),
                            "shares": shares,
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                        }

            # Track equity
            portfolio_value = cash
            if position is not None:
                portfolio_value += position["shares"] * price
            equity.append(portfolio_value)

        # Close any remaining position at last price
        if position is not None:
            last_price = df["Close"].iloc[-1] * (1 - self.slippage_pct)
            commission = position["shares"] * self.commission_per_share
            pnl = (last_price - position["entry_price"]) * position["shares"] - commission

            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": df.index[-1],
                "entry_price": position["entry_price"],
                "exit_price": round(last_price, 4),
                "shares": position["shares"],
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "pnl": round(pnl, 2),
                "pnl_pct": round((last_price / position["entry_price"] - 1) * 100, 2),
                "exit_reason": "End of backtest",
                "commission": round(commission * 2, 2),
            })

        equity_series = pd.Series(equity, index=df.index, name="equity")
        return trades, equity_series

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
            "exit_reasons": exit_reasons,
        }

        return metrics

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
            console.print()

        except Exception:
            print(f"\n{'='*50}")
            print(f"  BACKTEST RESULTS")
            print(f"{'='*50}")
            for key, val in metrics.items():
                if key != "exit_reasons":
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
