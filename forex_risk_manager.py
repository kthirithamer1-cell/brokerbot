"""
forex_risk_manager.py — Forex Risk Management Module
=====================================================
Handles leverage-aware position sizing, pip-based stop/take-profit,
margin tracking, cross-pair correlation limits, session-aware sizing,
spread filtering, and daily P&L targets.
"""

import logging
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ForexRiskManager:
    """
    Forex-specific risk management.

    Features:
        - Leverage-aware position sizing in lots (standard/mini/micro)
        - Pip-based stop-loss and take-profit
        - Margin tracking (used margin, free margin, margin level %)
        - Cross-pair correlation risk limiting
        - Same-currency exposure limits
        - Session-aware position sizing
        - Spread filter (skip trades with wide spreads)
        - Swap/rollover cost awareness
        - Regime-aware adaptive sizing
        - Daily P&L targets
        - Trailing stops
        - Time-based stops
    """

    # JPY pairs
    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        risk_cfg = self.config.get("risk", {})
        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade_pct", 1.0) / 100
        self.max_position_pct = risk_cfg.get("max_position_pct", 5.0) / 100
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 3.0) / 100
        self.max_open_positions = risk_cfg.get("max_open_positions", 5)
        self.stop_loss_atr_mult = risk_cfg.get("stop_loss_atr_mult", 1.5)
        self.take_profit_atr_mult = risk_cfg.get("take_profit_atr_mult", 2.5)
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 0.8) / 100
        self.max_entry_spread_pips = risk_cfg.get("max_entry_spread_pips", 3.0)
        self.max_correlation = risk_cfg.get("max_correlation", 0.75)
        self.max_same_currency_exposure = risk_cfg.get("max_same_currency_exposure", 3)
        self.allow_weekend_trading = risk_cfg.get("allow_weekend_trading", False)
        self.fixed_lots = risk_cfg.get("fixed_lots", None)
        self.enable_compounding = risk_cfg.get("enable_compounding", True)
        self.max_stop_loss_pips = risk_cfg.get("max_stop_loss_pips", 35.0)

        # Regime stops
        self.regime_stops = risk_cfg.get("regime_stops", {})

        # Time-based stop
        self.time_stop_bars = risk_cfg.get("time_stop_bars", 48)
        self.time_stop_min_move_pct = risk_cfg.get("time_stop_min_move_pct", 0.15) / 100

        # Leverage config
        lev_cfg = self.config.get("leverage", {})
        self.max_leverage = lev_cfg.get("max_leverage", 10)
        self.margin_requirement_pct = lev_cfg.get("margin_requirement_pct", 3.0) / 100
        self.margin_call_level = lev_cfg.get("margin_call_level", 100)
        self.stop_out_level = lev_cfg.get("stop_out_level", 50)

        # Lot sizes
        pip_cfg = self.config.get("strategy", {}).get("pip_value", {})
        self.standard_lot = pip_cfg.get("standard_lot", 100000)
        self.mini_lot = pip_cfg.get("mini_lot", 10000)
        self.micro_lot = pip_cfg.get("micro_lot", 1000)
        self.default_lot_type = pip_cfg.get("default_lot_type", "micro")

        # Daily targets
        targets_cfg = self.config.get("targets", {})
        self.daily_profit_target_usd = targets_cfg.get("daily_profit_target_usd", 20.0)
        self.scale_down_at_pct = targets_cfg.get("scale_down_at_pct", 60) / 100
        self.stop_trading_at_pct = targets_cfg.get("stop_trading_at_pct", 120) / 100

        # Session config
        self.sessions_cfg = self.config.get("sessions", {})

        # State tracking
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.current_equity = 0.0
        self.open_positions = {}
        self.trade_log = []
        self._killed = False
        self._kill_reason = ""
        self._current_regime = "medium"
        self._bar_counter = {}  # Track bars since entry per position

    def get_pip_size(self, pair: str) -> float:
        """Get pip size for a pair."""
        clean = pair.replace("=X", "").replace("/", "").upper()
        return 0.01 if clean in self.JPY_PAIRS else 0.0001

    def get_lot_size(self) -> int:
        """Get the lot size in units based on config."""
        if self.default_lot_type == "standard":
            return self.standard_lot
        elif self.default_lot_type == "mini":
            return self.mini_lot
        else:
            return self.micro_lot

    def get_pip_value(self, pair: str, lot_units: int = None) -> float:
        """
        Calculate pip value in USD for a given position size.
        For most pairs: pip_value = lot_size * pip_size
        For XXX/USD pairs: pip_value = lot_size * 0.0001 = $0.10 per micro lot
        For USD/XXX pairs: pip_value ≈ lot_size * 0.0001 / rate
        """
        if lot_units is None:
            lot_units = self.get_lot_size()

        pip_size = self.get_pip_size(pair)
        clean = pair.replace("=X", "").replace("/", "").upper()

        # Simplified pip value (assumes USD account)
        # For pairs ending in USD (EURUSD, GBPUSD, etc.): pip_value = lot_size * pip_size
        # For pairs starting with USD (USDJPY, etc.): pip_value ≈ lot_size * pip_size / rate
        backtest_pip_values = self.config.get("backtest", {}).get("pip_values", {})
        if clean in backtest_pip_values:
            # Scale from standard lot to actual lot size
            return backtest_pip_values[clean] * (lot_units / self.standard_lot)

        # Default: approximate
        return lot_units * pip_size

    # ──────────────────────────────────────────────
    #  EQUITY & MARGIN
    # ──────────────────────────────────────────────

    def update_equity(self, equity: float):
        """Update current account equity."""
        if self.start_of_day_equity <= 0:
            self.start_of_day_equity = equity
        self.current_equity = equity

    def get_used_margin(self) -> float:
        """Calculate total margin used by open positions."""
        total = 0.0
        for pair, pos in self.open_positions.items():
            position_value = pos["units"] * pos["entry_price"]
            total += position_value * self.margin_requirement_pct
        return total

    def get_free_margin(self) -> float:
        """Calculate available margin."""
        return max(0, self.current_equity - self.get_used_margin())

    def get_margin_level(self) -> float:
        """Calculate margin level percentage."""
        used = self.get_used_margin()
        if used <= 0:
            return float("inf")
        return (self.current_equity / used) * 100

    def update_regime(self, regime: str):
        """Update current market regime."""
        self._current_regime = regime

    # ──────────────────────────────────────────────
    #  POSITION SIZING (LEVERAGE-AWARE)
    # ──────────────────────────────────────────────

    def calculate_position_size(
        self, pair: str, entry_price: float, stop_loss_price: float,
        session_multiplier: float = 1.0,
    ) -> dict:
        """
        Calculate position size in currency units (lots) based on risk.

        Uses fixed-fractional sizing: risk X% of equity per trade.
        Size is constrained by leverage limit and available margin.

        Returns:
            Dict with units, lots, risk_amount, risk_pct, size_multiplier
        """
        equity = self.current_equity
        if equity <= 0:
            return {"units": 0, "lots": 0.0, "risk_amount": 0, "risk_pct": 0, "size_multiplier": 0}

        pip_size = self.get_pip_size(pair)
        sl_pips = abs(entry_price - stop_loss_price) / pip_size

        if sl_pips <= 0:
            return {"units": 0, "lots": 0.0, "risk_amount": 0, "risk_pct": 0, "size_multiplier": 0}

        # Fixed lot override if specified (with dynamic compounding if enabled)
        if self.fixed_lots is not None and self.fixed_lots > 0:
            current_lots = self.fixed_lots
            if self.enable_compounding and equity > 0:
                if equity >= 110.0:
                    current_lots = max(self.fixed_lots, 0.04)
                elif equity >= 75.0:
                    current_lots = max(self.fixed_lots, 0.03)
            units = int(round(current_lots * 100000))
            actual_risk = sl_pips * self.get_pip_value(pair, units)
            risk_pct = (actual_risk / equity * 100) if equity > 0 else 0
            return {
                "units": units,
                "lots": round(current_lots, 2),
                "risk_amount": round(actual_risk, 2),
                "risk_pct": round(risk_pct, 2),
                "size_multiplier": 1.0,
                "margin_needed": round(units * entry_price * self.margin_requirement_pct, 2),
            }

        # Max risk in dollars
        risk_amount = equity * self.max_risk_per_trade

        # Session & regime multiplier
        size_multiplier = session_multiplier
        if self._current_regime == "volatile":
            size_multiplier *= 0.7
        elif self._current_regime == "trending":
            size_multiplier *= 1.1
        elif self._current_regime == "ranging":
            size_multiplier *= 0.9

        # Scale down if approaching daily target
        if self.daily_profit_target_usd > 0:
            progress = self.daily_pnl / self.daily_profit_target_usd
            if progress >= self.stop_trading_at_pct:
                return {"units": 0, "lots": 0.0, "risk_amount": 0, "risk_pct": 0, "size_multiplier": 0}
            elif progress >= self.scale_down_at_pct:
                size_multiplier *= 0.5

        risk_amount *= size_multiplier

        # Calculate units from risk
        lot_unit = self.get_lot_size()
        pip_value_per_unit = self.get_pip_value(pair, lot_units=lot_unit)
        pip_value_per_single = pip_value_per_unit  # Per lot_unit

        if pip_value_per_single <= 0:
            return {"units": 0, "lots": 0.0, "risk_amount": 0, "risk_pct": 0, "size_multiplier": 0}

        # Units that risk exactly risk_amount at sl_pips distance
        max_lots = risk_amount / (sl_pips * pip_value_per_single)
        units = int(max_lots * lot_unit)

        # Constrain by leverage
        max_units_by_leverage = int(equity * self.max_leverage / entry_price)
        units = min(units, max_units_by_leverage)

        # Constrain by max position percentage
        max_units_by_pct = int(equity * self.max_position_pct / (entry_price * self.margin_requirement_pct))
        units = min(units, max_units_by_pct)

        # Constrain by free margin
        margin_needed = units * entry_price * self.margin_requirement_pct
        free_margin = self.get_free_margin()
        if margin_needed > free_margin:
            units = int(free_margin / (entry_price * self.margin_requirement_pct))

        # Round to lot size (for micro accounts $15+, allow 1 micro lot if free margin covers it)
        if units < lot_unit and equity >= 15.0:
            margin_needed_min = lot_unit * entry_price * self.margin_requirement_pct
            if margin_needed_min <= self.get_free_margin():
                units = lot_unit
            else:
                units = 0
        else:
            units = max(0, (units // lot_unit) * lot_unit)
        lots = units / lot_unit

        actual_risk = sl_pips * self.get_pip_value(pair, units)
        risk_pct = (actual_risk / equity * 100) if equity > 0 else 0

        return {
            "units": units,
            "lots": lots,
            "risk_amount": round(actual_risk, 2),
            "risk_pct": round(risk_pct, 2),
            "size_multiplier": round(size_multiplier, 2),
            "margin_needed": round(units * entry_price * self.margin_requirement_pct, 2),
        }

    # ──────────────────────────────────────────────
    #  BRACKET LEVELS (STOP/TARGET)
    # ──────────────────────────────────────────────

    def compute_bracket_levels(
        self, pair: str, entry_price: float, atr: float, direction: int = 1,
    ) -> dict:
        """
        Compute stop-loss, take-profit, and breakeven levels.

        Args:
            pair: Forex pair
            entry_price: Entry price
            atr: ATR value (in price, not pips)
            direction: 1 for long, -1 for short

        Returns:
            Dict with stop_loss, take_profit, breakeven, risk_reward_ratio
        """
        # Get regime-aware multipliers
        sl_mult = self.stop_loss_atr_mult
        tp_mult = self.take_profit_atr_mult

        regime_cfg = self.regime_stops.get(self._current_regime, {})
        if regime_cfg:
            sl_mult = regime_cfg.get("stop_loss_atr_mult", sl_mult)
            tp_mult = regime_cfg.get("take_profit_atr_mult", tp_mult)

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        # Apply minimum floors and maximum risk cap
        pip_size = self.get_pip_size(pair)
        min_sl_pips = self.config.get("risk", {}).get("min_stop_loss_pips", 0.0)
        max_sl_pips = self.max_stop_loss_pips
        min_tp_pips = self.config.get("risk", {}).get("min_take_profit_pips", 0.0)
        if min_sl_pips > 0:
            sl_distance = max(sl_distance, min_sl_pips * pip_size)
        if max_sl_pips > 0:
            sl_distance = min(sl_distance, max_sl_pips * pip_size)
        if min_tp_pips > 0:
            tp_distance = max(tp_distance, min_tp_pips * pip_size)

        if direction == 1:  # Long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Breakeven level
        be_mult = self.config.get("risk", {}).get("breakeven_atr_mult", 1.0)
        if direction == 1:
            breakeven_trigger = entry_price + atr * be_mult
        else:
            breakeven_trigger = entry_price - atr * be_mult

        risk_reward = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0

        # Convert to pips for logging
        sl_pips = round(sl_distance / pip_size, 1)
        tp_pips = round(tp_distance / pip_size, 1)

        return {
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "breakeven_trigger": round(breakeven_trigger, 5),
            "risk_reward_ratio": risk_reward,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "regime": self._current_regime,
        }

    # ──────────────────────────────────────────────
    #  TRADE VALIDATION
    # ──────────────────────────────────────────────

    def validate_trade(
        self, pair: str, price: float, stop_loss: float,
        units: int, direction: int = 1,
        current_spread_pips: float = 0.0,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> tuple[bool, str]:
        """
        Validate a proposed trade against all risk rules.

        Returns:
            (is_valid, reason_string)
        """
        # Kill switch
        if self._killed:
            return False, f"Kill switch: {self._kill_reason}"

        # Max positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max positions reached ({self.max_open_positions})"

        # Already in this pair
        if pair in self.open_positions:
            return False, f"Already have position in {pair}"

        # Daily loss limit
        if self.start_of_day_equity > 0:
            daily_loss = -self.daily_pnl
            max_loss = self.start_of_day_equity * self.max_daily_loss_pct
            if daily_loss >= max_loss:
                self._killed = True
                self._kill_reason = f"Daily loss limit hit (${daily_loss:.2f} >= ${max_loss:.2f})"
                return False, self._kill_reason

        # Spread filter
        if current_spread_pips > self.max_entry_spread_pips:
            return False, f"Spread too wide ({current_spread_pips:.1f} > {self.max_entry_spread_pips:.1f} pips)"

        # Margin check
        margin_needed = units * price * self.margin_requirement_pct
        if margin_needed > self.get_free_margin():
            return False, f"Insufficient margin (need ${margin_needed:.2f}, have ${self.get_free_margin():.2f})"

        # Margin level check
        projected_margin_level = self.get_margin_level()
        if projected_margin_level < self.margin_call_level * 1.5:
            return False, f"Margin level too low ({projected_margin_level:.0f}%)"

        # Same-currency exposure limit
        clean_pair = pair.replace("=X", "").replace("/", "").upper()
        base = clean_pair[:3]
        quote = clean_pair[3:]
        currency_count = {}
        for open_pair in self.open_positions:
            op = open_pair.replace("=X", "").replace("/", "").upper()
            for curr in [op[:3], op[3:]]:
                currency_count[curr] = currency_count.get(curr, 0) + 1

        for curr in [base, quote]:
            if currency_count.get(curr, 0) >= self.max_same_currency_exposure:
                return False, f"Max {curr} exposure reached ({self.max_same_currency_exposure} positions)"

        # Correlation check
        if correlation_matrix is not None and not correlation_matrix.empty:
            for open_pair in self.open_positions:
                op_clean = open_pair.replace("=X", "").replace("/", "").upper()
                if clean_pair in correlation_matrix.columns and op_clean in correlation_matrix.index:
                    corr = abs(correlation_matrix.loc[op_clean, clean_pair])
                    if corr > self.max_correlation:
                        return False, f"Correlation too high with {op_clean} ({corr:.2f} > {self.max_correlation})"

        return True, "OK"

    # ──────────────────────────────────────────────
    #  POSITION TRACKING
    # ──────────────────────────────────────────────

    def register_open(
        self, pair: str, entry_price: float, units: int,
        stop_loss: float, direction: int = 1,
    ):
        """Register a new open position."""
        self.open_positions[pair] = {
            "entry_price": entry_price,
            "units": units,
            "stop_loss": stop_loss,
            "direction": direction,
            "high_watermark": entry_price,
            "low_watermark": entry_price,
            "opened_at": datetime.now().isoformat(),
            "bars_held": 0,
        }
        self._bar_counter[pair] = 0
        logger.info(f"📝 Position opened: {pair} {'LONG' if direction == 1 else 'SHORT'} "
                     f"{units} units @ {entry_price:.5f}")

    def register_close(self, pair: str, exit_price: float, reason: str = ""):
        """Register a position close and update P&L."""
        if pair not in self.open_positions:
            return

        pos = self.open_positions[pair]
        direction = pos["direction"]
        pnl_per_unit = (exit_price - pos["entry_price"]) * direction
        pip_size = self.get_pip_size(pair)
        pnl_pips = pnl_per_unit / pip_size
        pnl_usd = pnl_pips * self.get_pip_value(pair, pos["units"])

        self.daily_pnl += pnl_usd

        trade_record = {
            "pair": pair,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "units": pos["units"],
            "pnl_pips": round(pnl_pips, 1),
            "pnl_usd": round(pnl_usd, 2),
            "reason": reason,
            "opened_at": pos["opened_at"],
            "closed_at": datetime.now().isoformat(),
            "bars_held": pos.get("bars_held", 0),
        }
        self.trade_log.append(trade_record)

        logger.info(f"📝 Position closed: {pair} | PnL: {pnl_pips:+.1f} pips (${pnl_usd:+.2f}) | {reason}")

        del self.open_positions[pair]
        self._bar_counter.pop(pair, None)

    def check_trailing_stops(self, current_prices: dict) -> list[str]:
        """Check trailing stops, breakeven, and time stops for all positions."""
        to_close = []

        for pair, pos in list(self.open_positions.items()):
            price = current_prices.get(pair, 0)
            if price <= 0:
                continue

            direction = pos["direction"]

            # Update high/low watermark
            if direction == 1:
                pos["high_watermark"] = max(pos["high_watermark"], price)
            else:
                pos["low_watermark"] = min(pos["low_watermark"], price)

            # Check stop loss
            if direction == 1 and price <= pos["stop_loss"]:
                to_close.append(pair)
                continue
            elif direction == -1 and price >= pos["stop_loss"]:
                to_close.append(pair)
                continue

            # Trailing stop
            if direction == 1:
                trail_price = pos["high_watermark"] * (1 - self.trailing_stop_pct)
                if price <= trail_price and price > pos["entry_price"]:
                    to_close.append(pair)
                    continue
            else:
                trail_price = pos["low_watermark"] * (1 + self.trailing_stop_pct)
                if price >= trail_price and price < pos["entry_price"]:
                    to_close.append(pair)
                    continue

            # Breakeven stop
            enable_be = self.config.get("risk", {}).get("enable_breakeven_stop", True)
            if enable_be:
                be_mult = self.config.get("risk", {}).get("breakeven_atr_mult", 1.0)
                move = abs(price - pos["entry_price"])
                entry = pos["entry_price"]
                atr_estimate = entry * 0.003  # Rough 30-pip estimate
                if move > atr_estimate * be_mult:
                    # Move stop to breakeven
                    if direction == 1:
                        pos["stop_loss"] = max(pos["stop_loss"], entry)
                    else:
                        pos["stop_loss"] = min(pos["stop_loss"], entry)

            # Time-based stop
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= self.time_stop_bars:
                move_pct = abs(price - pos["entry_price"]) / pos["entry_price"]
                if move_pct < self.time_stop_min_move_pct:
                    to_close.append(pair)

        return to_close

    # ──────────────────────────────────────────────
    #  SESSION AWARENESS
    # ──────────────────────────────────────────────

    def get_session_multiplier(self, pair: str) -> float:
        """Get position size multiplier based on current session and pair."""
        now = datetime.utcnow()
        hour = now.hour

        # Check London/NY overlap (highest liquidity)
        overlap_cfg = self.sessions_cfg.get("london_ny_overlap", {})
        if 12 <= hour < 16:
            return overlap_cfg.get("size_multiplier", 1.2)

        # Check individual sessions
        for session_name in ["london", "new_york", "tokyo"]:
            session = self.sessions_cfg.get(session_name, {})
            start_h = int(session.get("start", "00:00").split(":")[0])
            end_h = int(session.get("end", "24:00").split(":")[0])

            if start_h <= hour < end_h:
                active_pairs = session.get("active_pairs", [])
                clean_pair = pair.replace("=X", "").replace("/", "").upper()

                if clean_pair in active_pairs:
                    return session.get("size_multiplier", 1.0)
                else:
                    return session.get("size_multiplier", 1.0) * 0.7  # Less liquid for this pair

        return 0.6  # Off-hours (reduced)

    # ──────────────────────────────────────────────
    #  TRADING CONTROL
    # ──────────────────────────────────────────────

    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if trading is currently allowed."""
        if self._killed:
            return False, self._kill_reason

        # Check daily loss
        if self.start_of_day_equity > 0:
            daily_loss = -self.daily_pnl
            max_loss = self.start_of_day_equity * self.max_daily_loss_pct
            if daily_loss >= max_loss:
                self._killed = True
                self._kill_reason = f"Daily loss limit: ${daily_loss:.2f}"
                return False, self._kill_reason

        # Check daily profit target (stop trading if exceeded)
        if self.daily_profit_target_usd > 0:
            if self.daily_pnl >= self.daily_profit_target_usd * self.stop_trading_at_pct:
                return False, f"Daily target exceeded: ${self.daily_pnl:.2f}"

        # Check margin level
        margin_level = self.get_margin_level()
        if margin_level < self.stop_out_level:
            self._killed = True
            self._kill_reason = f"Margin stop-out: {margin_level:.0f}%"
            return False, self._kill_reason

        # Weekend check
        if not self.allow_weekend_trading:
            now = datetime.utcnow()
            # Saturday is closed all day
            if now.weekday() == 5:
                return False, "Weekend (Saturday) — forex market closed"

            # Sunday is closed until 21:00 UTC (Sydney/Asian session opens Sunday ~21:00 UTC)
            if now.weekday() == 6 and now.hour < 21:
                return False, "Weekend (Sunday before 21:00 UTC) — forex market closed"

            # Friday evening shutdown (after 21:00 UTC)
            if now.weekday() == 4 and now.hour >= 21:
                return False, "Friday evening — preparing for weekend"

        return True, "OK"

    def reset_daily(self):
        """Reset daily tracking at start of new trading day."""
        self.daily_pnl = 0.0
        self._killed = False
        self._kill_reason = ""
        if self.current_equity > 0:
            self.start_of_day_equity = self.current_equity

    def get_daily_summary(self) -> dict:
        """Get daily trading summary."""
        target = self.daily_profit_target_usd
        progress = (self.daily_pnl / target * 100) if target > 0 else 0

        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl / self.current_equity * 100, 2) if self.current_equity > 0 else 0,
            "target_progress_pct": round(progress, 1),
            "trades_today": len([t for t in self.trade_log
                                if t.get("closed_at", "").startswith(date.today().isoformat())]),
            "open_positions": len(self.open_positions),
            "used_margin": round(self.get_used_margin(), 2),
            "free_margin": round(self.get_free_margin(), 2),
            "margin_level": round(self.get_margin_level(), 1),
            "equity": round(self.current_equity, 2),
            "regime": self._current_regime,
        }
