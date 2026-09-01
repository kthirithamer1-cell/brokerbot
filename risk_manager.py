"""
risk_manager.py — Risk Management Module
=========================================
Handles position sizing, stop-loss/take-profit bracket orders,
daily drawdown kill-switch, max exposure limits, regime-aware
sizing, daily profit targets, and correlation-based exposure.
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
        - Regime-aware adaptive position sizing (NEW)
        - Daily profit target & scale-down (NEW)
        - Time-based stop (NEW)
        - Correlation-based exposure limit (NEW)
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

        # Regime-based stops
        self.regime_stops = risk_cfg.get("regime_stops", {})

        # Time-based stop
        self.time_stop_bars = risk_cfg.get("time_stop_bars", 26)
        self.time_stop_min_move_pct = risk_cfg.get("time_stop_min_move_pct", 0.3) / 100

        # Correlation limit
        self.max_correlation = risk_cfg.get("max_correlation", 0.70)

        # Daily targets
        targets_cfg = self.config.get("targets", {})
        self.daily_profit_target = targets_cfg.get("daily_profit_target", 5.0)
        self.scale_down_at_pct = targets_cfg.get("scale_down_at_pct", 60) / 100
        self.stop_trading_at_pct = targets_cfg.get("stop_trading_at_pct", 120) / 100

        # Session awareness
        session_cfg = self.config.get("session", {})
        self.power_hour_boost = session_cfg.get("power_hour_boost", 1.0)
        self.midday_penalty = session_cfg.get("midday_penalty", 1.0)
        self.avoid_first_minutes = session_cfg.get("avoid_first_minutes", 5)

        # State tracking
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.current_equity = 0.0
        self.open_positions = {}
        self.trade_log = []
        self._killed = False
        self._kill_reason = ""
        self._today = date.today()
        self._current_regime = "medium"  # "trending", "ranging", "volatile", "medium"
        self._position_size_multiplier = 1.0  # Regime-based multiplier

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
            self._position_size_multiplier = 1.0  # Reset daily multiplier
            logger.info(f"New trading day. Start equity: ${equity:,.2f}")

        if self.start_of_day_equity <= 0:
            self.start_of_day_equity = equity

        self.current_equity = equity
        self.daily_pnl = equity - self.start_of_day_equity

    def update_regime(self, regime: str):
        """
        Update current market regime for adaptive risk parameters.

        Args:
            regime: One of "trending", "ranging", "volatile", "medium"
        """
        self._current_regime = regime
        logger.debug(f"Regime updated: {regime}")

    def get_regime_stops(self) -> tuple[float, float]:
        """
        Get regime-adjusted stop-loss and take-profit ATR multipliers.

        Returns:
            (stop_loss_atr_mult, take_profit_atr_mult)
        """
        if self._current_regime in self.regime_stops:
            regime_cfg = self.regime_stops[self._current_regime]
            sl = regime_cfg.get("stop_loss_atr_mult", self.stop_loss_atr_mult)
            tp = regime_cfg.get("take_profit_atr_mult", self.take_profit_atr_mult)
            return sl, tp
        return self.stop_loss_atr_mult, self.take_profit_atr_mult

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

        # Daily profit target check — stop trading when exceeded
        if self.daily_pnl >= self.daily_profit_target * self.stop_trading_at_pct:
            self._killed = True
            self._kill_reason = (
                f"Daily profit ${self.daily_pnl:.2f} reached "
                f"{self.stop_trading_at_pct*100:.0f}% of target "
                f"(${self.daily_profit_target:.2f}). Locking profits."
            )
            logger.info(f"🎯 PROFIT TARGET: {self._kill_reason}")
            return False, self._kill_reason

        # Max open positions check
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        # Check session timing
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = (now - market_open).total_seconds() / 60
        if 0 < minutes_since_open < self.avoid_first_minutes:
            return False, f"Avoiding first {self.avoid_first_minutes} minutes after open"

        return True, "OK"

    def get_position_size_multiplier(self) -> float:
        """
        Get dynamic position size multiplier based on:
        - Daily P&L progress toward target (scale down after partial target)
        - Market regime (reduce in high volatility)
        - Session timing (boost in power hour, reduce at lunch)
        """
        multiplier = 1.0

        # Scale down after reaching partial daily target
        if self.daily_pnl > 0 and self.daily_profit_target > 0:
            progress = self.daily_pnl / self.daily_profit_target
            if progress >= self.scale_down_at_pct:
                # Linear scale-down from 100% to 50% as progress goes from scale_down to stop_trading
                scale_range = self.stop_trading_at_pct - self.scale_down_at_pct
                if scale_range > 0:
                    scale_factor = 1.0 - 0.5 * min(1.0, (progress - self.scale_down_at_pct) / scale_range)
                    multiplier *= scale_factor
                    logger.debug(f"Scale-down multiplier: {scale_factor:.2f} (PnL progress: {progress*100:.0f}%)")

        # Regime-based multiplier
        if self._current_regime == "volatile":
            multiplier *= 0.6  # Reduce 40% in volatile regime
        elif self._current_regime == "ranging":
            multiplier *= 0.8  # Reduce 20% in ranging regime
        elif self._current_regime == "trending":
            multiplier *= 1.1  # Slight boost in trending regime

        # Session-based multiplier
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        if hour >= 15:  # Power hour (3:00-4:00 PM)
            multiplier *= self.power_hour_boost
        elif 11 <= hour <= 14 and (hour != 14 or minute < 30):  # Lunch (11:30-14:30)
            multiplier *= self.midday_penalty

        return max(0.3, min(multiplier, 1.5))  # Clamp between 30% and 150%

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

        # Apply dynamic position size multiplier
        size_multiplier = self.get_position_size_multiplier()

        # Max risk amount (adjusted by multiplier)
        risk_amount = equity * self.max_risk_per_trade * size_multiplier

        # Shares from risk
        shares_from_risk = int(risk_amount / risk_per_share)

        # Max position size (adjusted by multiplier)
        max_dollar_amount = equity * self.max_position_pct * size_multiplier
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
            "size_multiplier": round(size_multiplier, 2),
            "regime": self._current_regime,
        }

    def compute_bracket_levels(
        self,
        entry_price: float,
        atr: float,
        direction: int = 1,  # 1 = long, -1 = short
    ) -> dict:
        """
        Compute stop-loss and take-profit levels using ATR.
        Uses regime-aware ATR multipliers when available.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: 1 for long, -1 for short

        Returns:
            Dict with stop_loss, take_profit, trailing_stop_trigger
        """
        # Use regime-adjusted multipliers
        sl_mult, tp_mult = self.get_regime_stops()

        stop_distance = atr * sl_mult
        profit_distance = atr * tp_mult

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
            "regime": self._current_regime,
            "sl_atr_mult": sl_mult,
            "tp_atr_mult": tp_mult,
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
            - Trading allowed (kill switch + profit target)
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
            "bars_held": 0,
            "regime_at_entry": self._current_regime,
        }
        pos_type = "LONG" if direction == 1 else "SHORT"
        logger.info(f"📈 Position opened: {pos_type} {symbol} x{shares} @ ${entry_price:.2f} "
                     f"(regime={self._current_regime})")

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
            "bars_held": pos.get("bars_held", 0),
            "regime_at_entry": pos.get("regime_at_entry", "unknown"),
        }
        self.trade_log.append(trade_record)

        emoji = "✅" if pnl > 0 else "❌"
        pos_type = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"{emoji} {pos_type} position closed: {symbol} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {reason}"
        )

        # Log daily progress toward target
        progress = (self.daily_pnl / self.daily_profit_target * 100) if self.daily_profit_target > 0 else 0
        logger.info(f"📊 Daily PnL: ${self.daily_pnl:+.2f} ({progress:.0f}% of ${self.daily_profit_target} target)")

    def check_trailing_stops(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check trailing stops and time-based stops for all open positions.

        Returns:
            List of symbols that should be closed
        """
        to_close = []

        for symbol, pos in self.open_positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            direction = pos.get("direction", 1)

            # Increment bars held counter
            pos["bars_held"] = pos.get("bars_held", 0) + 1

            # ── Time-based stop: close if no movement after N bars ──
            if pos["bars_held"] >= self.time_stop_bars:
                move_pct = abs(price - pos["entry_price"]) / pos["entry_price"]
                if move_pct < self.time_stop_min_move_pct:
                    logger.info(
                        f"⏰ Time stop: {symbol} held {pos['bars_held']} bars, "
                        f"move={move_pct*100:.2f}% < min {self.time_stop_min_move_pct*100:.1f}%"
                    )
                    to_close.append(symbol)
                    continue

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

    def check_correlation(
        self,
        symbol: str,
        price_history: pd.Series,
        existing_histories: dict[str, pd.Series],
    ) -> tuple[bool, str]:
        """
        Check if adding this symbol would create too-correlated exposure.

        Args:
            symbol: New symbol to add
            price_history: Price series for the new symbol
            existing_histories: Dict of symbol -> price series for open positions

        Returns:
            (allowed: bool, reason: str)
        """
        if not existing_histories:
            return True, "No existing positions"

        new_returns = price_history.pct_change().dropna()

        for existing_symbol, existing_prices in existing_histories.items():
            existing_returns = existing_prices.pct_change().dropna()

            # Align series
            aligned = pd.concat([new_returns, existing_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue

            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if abs(corr) > self.max_correlation:
                return False, (
                    f"{symbol} has {corr:.2f} correlation with {existing_symbol} "
                    f"(max allowed: {self.max_correlation:.2f})"
                )

        return True, "Correlation OK"

    def get_daily_summary(self) -> dict:
        """Get daily trading summary with target progress."""
        progress = (self.daily_pnl / self.daily_profit_target * 100) if self.daily_profit_target > 0 else 0

        return {
            "date": self._today.isoformat(),
            "start_equity": round(self.start_of_day_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(
                (self.daily_pnl / self.start_of_day_equity * 100)
                if self.start_of_day_equity > 0 else 0, 2
            ),
            "target_progress_pct": round(progress, 1),
            "daily_target": self.daily_profit_target,
            "open_positions": len(self.open_positions),
            "trades_today": len([
                t for t in self.trade_log
                if t["closed_at"].startswith(self._today.isoformat())
            ]),
            "current_regime": self._current_regime,
            "size_multiplier": round(self.get_position_size_multiplier(), 2),
            "kill_switch": self._killed,
            "kill_reason": self._kill_reason,
        }

    def get_weekly_summary(self) -> dict:
        """Get weekly P&L summary from trade log."""
        if not self.trade_log:
            return {"total_pnl": 0, "trades": 0, "win_rate": 0}

        df = pd.DataFrame(self.trade_log)
        total_pnl = df["pnl"].sum()
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] < 0]

        return {
            "total_pnl": round(total_pnl, 2),
            "trades": len(df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(df) * 100, 1) if len(df) > 0 else 0,
            "avg_win": round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
            "target_hit": total_pnl >= self.daily_profit_target * 5,
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

    # Test regime-aware stops
    rm.update_regime("trending")
    bracket_trending = rm.compute_bracket_levels(entry_price=2.50, atr=0.25)
    print(f"\nTrending regime bracket: {bracket_trending}")

    rm.update_regime("ranging")
    bracket_ranging = rm.compute_bracket_levels(entry_price=2.50, atr=0.25)
    print(f"\nRanging regime bracket: {bracket_ranging}")

    # Test trade validation
    valid, reason = rm.validate_trade("ABCD", 2.50, 2.30, sizing["shares"])
    print(f"\nTrade valid: {valid} — {reason}")

    # Test daily summary
    summary = rm.get_daily_summary()
    print(f"\nDaily summary: {summary}")
