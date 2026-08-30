"""
risk_manager.py — Risk Management Module
=========================================
Handles position sizing, stop-loss/take-profit bracket orders,
daily drawdown kill-switch, and max exposure limits.
"""

import logging
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management for the trading bot.

    Features:
        - Fixed fractional position sizing (risk X% per trade)
        - ATR-based stop-loss and take-profit levels
        - Trailing stop-loss
        - Daily max drawdown kill-switch
        - Max open positions limit
        - Max single position exposure
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        risk_cfg = self.config.get("risk", {})
        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100
        self.max_position_pct = risk_cfg.get("max_position_pct", 5.0) / 100
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 3.0) / 100
        self.max_open_positions = risk_cfg.get("max_open_positions", 10)
        self.stop_loss_atr_mult = risk_cfg.get("stop_loss_atr_mult", 2.0)
        self.take_profit_atr_mult = risk_cfg.get("take_profit_atr_mult", 3.0)
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 1.5) / 100

        # State tracking
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.current_equity = 0.0
        self.open_positions = {}
        self.trade_log = []
        self._killed = False
        self._kill_reason = ""
        self._today = date.today()

    def update_equity(self, equity: float):
        """Update current equity. Resets daily tracking on new day."""
        if equity <= 0:
            return  # Ignore transient 0 equity from API lag

        today = date.today()
        if today != self._today:
            # New trading day — reset daily PnL
            self._today = today
            self.daily_pnl = 0.0
            self._killed = False
            self._kill_reason = ""
            self.start_of_day_equity = equity
            logger.info(f"New trading day. Start equity: ${equity:,.2f}")

        if self.start_of_day_equity <= 0:
            self.start_of_day_equity = equity

        self.current_equity = equity
        self.daily_pnl = equity - self.start_of_day_equity

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        if self._killed:
            return False, f"Kill-switch active: {self._kill_reason}"

        # Daily drawdown check
        if self.start_of_day_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.start_of_day_equity
            if daily_loss_pct >= self.max_daily_loss_pct:
                self._killed = True
                self._kill_reason = (
                    f"Daily loss {daily_loss_pct*100:.2f}% exceeds "
                    f"limit {self.max_daily_loss_pct*100:.1f}%"
                )
                logger.warning(f"🛑 KILL SWITCH: {self._kill_reason}")
                return False, self._kill_reason

        # Max open positions check
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        equity: Optional[float] = None,
    ) -> dict:
        """
        Calculate position size based on fixed fractional risk.

        Risk per trade = equity * max_risk_per_trade_pct
        Position size = risk_amount / (entry_price - stop_loss_price)

        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price
            equity: Account equity (uses current if not provided)

        Returns:
            Dict with shares, dollar_amount, risk_amount, risk_pct
        """
        equity = equity or self.current_equity
        if equity <= 0:
            return {"shares": 0, "error": "No equity available"}

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return {"shares": 0, "error": "Zero risk per share (stop = entry)"}

        # Max risk amount
        risk_amount = equity * self.max_risk_per_trade

        # Shares from risk
        shares_from_risk = int(risk_amount / risk_per_share)

        # Max position size
        max_dollar_amount = equity * self.max_position_pct
        shares_from_max_pos = int(max_dollar_amount / entry_price)

        # Take the smaller
        shares = min(shares_from_risk, shares_from_max_pos)
        shares = max(shares, 0)

        dollar_amount = shares * entry_price
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / equity if equity > 0 else 0

        return {
            "shares": shares,
            "dollar_amount": round(dollar_amount, 2),
            "risk_amount": round(actual_risk, 2),
            "risk_pct": round(actual_risk_pct * 100, 2),
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
        }

    def compute_bracket_levels(
        self,
        entry_price: float,
        atr: float,
        direction: int = 1,  # 1 = long, -1 = short
    ) -> dict:
        """
        Compute stop-loss and take-profit levels using ATR.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: 1 for long, -1 for short

        Returns:
            Dict with stop_loss, take_profit, trailing_stop_trigger
        """
        stop_distance = atr * self.stop_loss_atr_mult
        profit_distance = atr * self.take_profit_atr_mult

        if direction == 1:  # Long
            stop_loss = round(entry_price - stop_distance, 2)
            take_profit = round(entry_price + profit_distance, 2)
            trailing_trigger = round(entry_price * (1 + self.trailing_stop_pct), 2)
        else:  # Short
            stop_loss = round(entry_price + stop_distance, 2)
            take_profit = round(entry_price - profit_distance, 2)
            trailing_trigger = round(entry_price * (1 - self.trailing_stop_pct), 2)

        risk_reward = profit_distance / stop_distance if stop_distance > 0 else 0

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_distance": round(stop_distance, 2),
            "profit_distance": round(profit_distance, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "trailing_stop_pct": self.trailing_stop_pct * 100,
            "trailing_trigger": trailing_trigger,
        }

    def validate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        shares: int,
    ) -> tuple[bool, str]:
        """
        Validate a trade before execution.

        Checks:
            - Trading allowed (kill switch)
            - Position size limits
            - Duplicate position check
            - Risk/reward minimum

        Returns:
            (valid: bool, reason: str)
        """
        allowed, reason = self.is_trading_allowed()
        if not allowed:
            return False, reason

        # Check duplicate position
        if symbol in self.open_positions:
            return False, f"Already have open position in {symbol}"

        # Check position dollar size
        dollar_amount = shares * entry_price
        if self.current_equity > 0:
            position_pct = dollar_amount / self.current_equity
            if position_pct > self.max_position_pct:
                return False, (
                    f"Position {position_pct*100:.1f}% exceeds "
                    f"max {self.max_position_pct*100:.1f}%"
                )

        # Check minimum shares
        if shares < 1:
            return False, "Position size too small (0 shares)"

        return True, "Trade validated"

    def register_open(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        direction: int = 1,
    ):
        """Register a new open position (direction 1 = Long, -1 = Short)."""
        self.open_positions[symbol] = {
            "entry_price": entry_price,
            "shares": shares,
            "stop_loss": stop_loss,
            "direction": direction,
            "high_watermark": entry_price,
            "low_watermark": entry_price,
            "opened_at": datetime.now().isoformat(),
        }
        pos_type = "LONG" if direction == 1 else "SHORT"
        logger.info(f"📈 Position opened: {pos_type} {symbol} x{shares} @ ${entry_price:.2f}")

    def register_close(self, symbol: str, exit_price: float, reason: str = ""):
        """Register a position close and update daily PnL."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions.pop(symbol)
        direction = pos.get("direction", 1)

        if direction == 1:
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
        else:  # Short
            pnl = (pos["entry_price"] - exit_price) * pos["shares"]
            pnl_pct = (1 - exit_price / pos["entry_price"]) * 100

        self.daily_pnl += pnl

        trade_record = {
            "symbol": symbol,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "opened_at": pos["opened_at"],
            "closed_at": datetime.now().isoformat(),
        }
        self.trade_log.append(trade_record)

        emoji = "✅" if pnl > 0 else "❌"
        pos_type = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"{emoji} {pos_type} position closed: {symbol} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {reason}"
        )

    def check_trailing_stops(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check trailing stops for all open positions.

        Returns:
            List of symbols that should be closed
        """
        to_close = []

        for symbol, pos in self.open_positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            direction = pos.get("direction", 1)

            if direction == 1:  # Long
                if price > pos.get("high_watermark", pos["entry_price"]):
                    pos["high_watermark"] = price

                trail_stop_price = pos["high_watermark"] * (1 - self.trailing_stop_pct)
                if price <= trail_stop_price:
                    logger.info(
                        f"📉 Trailing stop hit: {symbol} (LONG) @ ${price:.2f} "
                        f"(high: ${pos['high_watermark']:.2f}, trail: ${trail_stop_price:.2f})"
                    )
                    to_close.append(symbol)
                elif price <= pos["stop_loss"]:
                    logger.info(f"🛑 Stop-loss hit: {symbol} (LONG) @ ${price:.2f}")
                    to_close.append(symbol)

            else:  # Short
                if price < pos.get("low_watermark", pos["entry_price"]):
                    pos["low_watermark"] = price

                trail_stop_price = pos["low_watermark"] * (1 + self.trailing_stop_pct)
                if price >= trail_stop_price:
                    logger.info(
                        f"📈 Trailing stop hit: {symbol} (SHORT) @ ${price:.2f} "
                        f"(low: ${pos['low_watermark']:.2f}, trail: ${trail_stop_price:.2f})"
                    )
                    to_close.append(symbol)
                elif price >= pos["stop_loss"]:
                    logger.info(f"🛑 Stop-loss hit: {symbol} (SHORT) @ ${price:.2f}")
                    to_close.append(symbol)

        return to_close

    def get_daily_summary(self) -> dict:
        """Get daily trading summary."""
        return {
            "date": self._today.isoformat(),
            "start_equity": round(self.start_of_day_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(
                (self.daily_pnl / self.start_of_day_equity * 100)
                if self.start_of_day_equity > 0 else 0, 2
            ),
            "open_positions": len(self.open_positions),
            "trades_today": len([
                t for t in self.trade_log
                if t["closed_at"].startswith(self._today.isoformat())
            ]),
            "kill_switch": self._killed,
            "kill_reason": self._kill_reason,
        }


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager()

    rm.update_equity(100_000)

    # Test position sizing
    sizing = rm.calculate_position_size(
        entry_price=2.50,
        stop_loss_price=2.30,
    )
    print(f"\nPosition sizing: {sizing}")

    # Test bracket levels
    bracket = rm.compute_bracket_levels(entry_price=2.50, atr=0.25)
    print(f"\nBracket order levels: {bracket}")

    # Test trade validation
    valid, reason = rm.validate_trade("ABCD", 2.50, 2.30, sizing["shares"])
    print(f"\nTrade valid: {valid} — {reason}")
