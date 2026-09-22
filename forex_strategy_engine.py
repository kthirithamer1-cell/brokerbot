"""
forex_strategy_engine.py — Classical Rule-Based Forex Strategies & Hybrid Gate
==============================================================================
Implements proven institutional forex strategies:
  1. H1 Trend-Following (EMA 50/200 + ADX + Donchian Breakout/Pullback)
  2. London Open Breakout (Asian Range Breakout 07:00–10:00 UTC)
  3. Hybrid Validation Gate (combines Rule Trigger + ML Ensemble + Currency Strength)
"""

import logging
from datetime import datetime, time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class H1TrendStrategy:
    """
    H1 Trend-Following Strategy:
      - Trend Direction: EMA 50 > EMA 200 (Uptrend), EMA 50 < EMA 200 (Downtrend)
      - Trend Strength: ADX(14) >= min_adx (filters out sideways chop)
      - Trigger: 20-bar Donchian breakout OR EMA 20 pullback bounce
      - Momentum Safety: RSI(14) between 40 and 65 (avoids buying exhaustion)
    """

    def __init__(self, min_adx: float = 22.0, rsi_min: float = 40.0, rsi_max: float = 65.0):
        self.min_adx = min_adx
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max

    def generate_signal(self, df: pd.DataFrame, pair: str) -> Dict[str, Any]:
        """
        Evaluate H1 trend signal on recent candle history.
        Returns:
            dict: {
                'signal': 1 (BUY), -1 (SELL), or 0 (HOLD),
                'strategy': 'h1_trend',
                'reason': str,
                'adx': float,
                'rsi': float,
                'ema_trend': 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            }
        """
        if df.empty or len(df) < 50:
            return {"signal": 0, "strategy": "h1_trend", "reason": "Insufficient data"}

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        # 1. EMAs for Trend Direction
        ema_50 = last.get("ema_50")
        ema_200 = last.get("ema_200")
        ema_20 = last.get("ema_20")
        close = last.get("Close")

        if ema_50 is None or ema_200 is None or ema_20 is None or close is None:
            # If not pre-computed, calculate on the fly
            close_series = df["Close"]
            if ema_50 is None:
                ema_50 = close_series.ewm(span=50, adjust=False).mean().iloc[-1]
            if ema_200 is None:
                ema_200 = close_series.ewm(span=200, adjust=False).mean().iloc[-1]
            if ema_20 is None:
                ema_20 = close_series.ewm(span=20, adjust=False).mean().iloc[-1]
            if close is None:
                close = close_series.iloc[-1]

        bullish_trend = (close > ema_50) and (ema_50 > ema_200)
        bearish_trend = (close < ema_50) and (ema_50 < ema_200)

        # 2. ADX Filter
        adx = float(last.get("adx", 0.0))
        if adx < self.min_adx:
            return {
                "signal": 0,
                "strategy": "h1_trend",
                "reason": f"ADX {adx:.1f} < {self.min_adx} (ranging market)",
                "adx": adx,
                "ema_trend": "BULLISH" if bullish_trend else ("BEARISH" if bearish_trend else "NEUTRAL"),
            }

        # 3. RSI Filter
        rsi = float(last.get("rsi_14", last.get("rsi", 50.0)))
        if bullish_trend and rsi > self.rsi_max:
            return {
                "signal": 0,
                "strategy": "h1_trend",
                "reason": f"RSI {rsi:.1f} > {self.rsi_max} (overbought exhaustion)",
                "adx": adx,
                "rsi": rsi,
                "ema_trend": "BULLISH",
            }
        if bearish_trend and rsi < self.rsi_min:
            return {
                "signal": 0,
                "strategy": "h1_trend",
                "reason": f"RSI {rsi:.1f} < {self.rsi_min} (oversold exhaustion)",
                "adx": adx,
                "rsi": rsi,
                "ema_trend": "BEARISH",
            }

        # 4. Breakout / Pullback Trigger
        # Donchian 20-period high/low excluding current bar
        lookback = min(20, len(df) - 1)
        high_series = df["High"] if "High" in df.columns else df["Close"]
        low_series = df["Low"] if "Low" in df.columns else df["Close"]
        recent_high = high_series.iloc[-lookback-1:-1].max()
        recent_low = low_series.iloc[-lookback-1:-1].min()

        # Bullish: Breakout above recent 20-bar high OR EMA 20 pullback bounce
        if bullish_trend:
            is_breakout = close >= recent_high
            prev_low = prev.get("Low", prev.get("Close", close))
            is_pullback_bounce = (prev_low <= ema_20) and (close > ema_20)
            if is_breakout or is_pullback_bounce:
                trigger_type = "Donchian Breakout" if is_breakout else "EMA 20 Bounce"
                return {
                    "signal": 1,
                    "strategy": "h1_trend",
                    "reason": f"Bullish H1 Trend ({trigger_type}, ADX={adx:.1f}, RSI={rsi:.1f})",
                    "adx": adx,
                    "rsi": rsi,
                    "ema_trend": "BULLISH",
                }

        # Bearish: Breakdown below recent 20-bar low OR EMA 20 pullback rejection
        if bearish_trend:
            is_breakdown = close <= recent_low
            is_pullback_rejection = (prev.get("High", close) >= ema_20) and (close < ema_20)
            if is_breakdown or is_pullback_rejection:
                trigger_type = "Donchian Breakdown" if is_breakdown else "EMA 20 Rejection"
                return {
                    "signal": -1,
                    "strategy": "h1_trend",
                    "reason": f"Bearish H1 Trend ({trigger_type}, ADX={adx:.1f}, RSI={rsi:.1f})",
                    "adx": adx,
                    "rsi": rsi,
                    "ema_trend": "BEARISH",
                }

        return {
            "signal": 0,
            "strategy": "h1_trend",
            "reason": "Trend aligned, waiting for breakout or pullback trigger",
            "adx": adx,
            "rsi": rsi,
            "ema_trend": "BULLISH" if bullish_trend else ("BEARISH" if bearish_trend else "NEUTRAL"),
        }


class LondonBreakoutStrategy:
    """
    London Open Asian Range Breakout:
      - Asian Session: 00:00 to 06:45 UTC (marks high and low of the session)
      - Trade Window: 07:00 to 11:00 UTC (London Open volume surge)
      - Long Entry: Breakout above Asian High + buffer
      - Short Entry: Breakdown below Asian Low - buffer
      - Best pairs: EURUSD, GBPUSD, EURGBP, GBPJPY
    """

    def __init__(self, buffer_pips: float = 3.0, pip_size: float = 0.0001):
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size

    def generate_signal(self, df: pd.DataFrame, pair: str, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Evaluate London breakout signal."""
        if df.empty or len(df) < 15:
            return {"signal": 0, "strategy": "london_breakout", "reason": "Insufficient data"}

        now = current_time or datetime.utcnow()
        current_hour = now.hour

        # Check if in London Open execution window (07:00 to 11:00 UTC)
        if not (7 <= current_hour < 11):
            return {
                "signal": 0,
                "strategy": "london_breakout",
                "reason": f"Outside London open window (current hour: {current_hour}:00 UTC)",
            }

        # Identify bars belonging to Asian session today (00:00 to 06:59 UTC)
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"signal": 0, "strategy": "london_breakout", "reason": "Index is not datetime"}

        today_utc = now.date()
        asian_bars = df[(df.index.date == today_utc) & (df.index.hour < 7)]

        if len(asian_bars) < 3:
            return {"signal": 0, "strategy": "london_breakout", "reason": "Asian session bars not fully formed"}

        asian_high = asian_bars["High"].max() if "High" in asian_bars.columns else asian_bars["Close"].max()
        asian_low = asian_bars["Low"].min() if "Low" in asian_bars.columns else asian_bars["Close"].min()
        asian_range_pips = (asian_high - asian_low) / self.pip_size

        # Asian range filter: ignore if Asian range was unreasonably massive (> 60 pips on EUR/GBP)
        if asian_range_pips > 60.0 or asian_range_pips < 10.0:
            return {
                "signal": 0,
                "strategy": "london_breakout",
                "reason": f"Asian range {asian_range_pips:.1f} pips outside 10-60 pip sweet spot",
            }

        buffer = self.buffer_pips * self.pip_size
        last_close = df["Close"].iloc[-1]

        if last_close > asian_high + buffer:
            return {
                "signal": 1,
                "strategy": "london_breakout",
                "reason": f"London Bullish Breakout above Asian High {asian_high:.5f} (+{self.buffer_pips} pips)",
                "asian_high": asian_high,
                "asian_low": asian_low,
                "range_pips": asian_range_pips,
            }

        if last_close < asian_low - buffer:
            return {
                "signal": -1,
                "strategy": "london_breakout",
                "reason": f"London Bearish Breakdown below Asian Low {asian_low:.5f} (-{self.buffer_pips} pips)",
                "asian_high": asian_high,
                "asian_low": asian_low,
                "range_pips": asian_range_pips,
            }

        return {
            "signal": 0,
            "strategy": "london_breakout",
            "reason": f"Price inside Asian range ({asian_low:.5f} - {asian_high:.5f})",
        }


class ForexStrategyEngine:
    """
    Unified Hybrid Strategy Engine:
      1. Evaluates classical rule strategies (H1 Trend & London Breakout)
      2. Validates against Machine Learning Ensemble predictions
      3. Validates against Relative Currency Strength Matrix
    """

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        strat_cfg = self.config.get("strategy", {})
        hybrid_cfg = self.config.get("hybrid", {})

        # Rule strategy parameters
        min_adx = hybrid_cfg.get("min_adx", 22.0)
        self.h1_trend = H1TrendStrategy(min_adx=min_adx)
        self.london_breakout = LondonBreakoutStrategy()

        # Hybrid gate rules
        self.require_ml = hybrid_cfg.get("require_ml_confirmation", True)
        self.require_cs = hybrid_cfg.get("require_currency_strength", True)
        self.min_confidence = strat_cfg.get("confidence_threshold", 0.40)
        self.min_strength_diff = hybrid_cfg.get("min_strength_divergence", 0.002)

    def evaluate_rules(self, df: pd.DataFrame, pair: str, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run rule strategies to see if any strategy triggered an entry.
        Priority:
          1. London Breakout (time-sensitive institutional volume)
          2. H1 Trend-Following (macro momentum)
        """
        # 1. Try London Breakout
        london_result = self.london_breakout.generate_signal(df, pair, current_time=current_time)
        if london_result["signal"] != 0:
            return london_result

        # 2. Try H1 Trend Following
        trend_result = self.h1_trend.generate_signal(df, pair)
        if trend_result["signal"] != 0:
            return trend_result

        # If neither triggered, return the informative trend reason
        return trend_result

    def evaluate_hybrid_gate(
        self,
        rule_signal: int,
        rule_reason: str,
        ml_prediction: Dict[str, Any],
        currency_strength: Optional[Dict[str, float]],
        pair: str,
    ) -> Tuple[int, str]:
        """
        Hybrid Decision Gate:
        Combines rule trigger with ML conviction and macro currency strength.

        Returns:
            (final_signal, explanation_string)
        """
        if rule_signal == 0:
            return 0, f"No rule trigger: {rule_reason}"

        ml_signal = int(ml_prediction.get("signal", 0))
        confidence = float(ml_prediction.get("confidence", 0.0))
        prob_buy = float(ml_prediction.get("prob_buy", 0.0))
        prob_sell = float(ml_prediction.get("prob_sell", 0.0))

        # Check 1: ML Agreement
        if self.require_ml:
            # If ML says HOLD or opposite direction with strong lead, block trade
            if rule_signal == 1:
                # Need buy probability to exceed sell probability and meet minimum gate
                if prob_buy < prob_sell or prob_buy < self.min_confidence:
                    return 0, f"Blocked by ML: Buy prob ({prob_buy*100:.1f}%) < threshold ({self.min_confidence*100:.1f}%) or < Sell prob ({prob_sell*100:.1f}%)"
            elif rule_signal == -1:
                # Need sell probability to exceed buy probability and meet minimum gate
                if prob_sell < prob_buy or prob_sell < self.min_confidence:
                    return 0, f"Blocked by ML: Sell prob ({prob_sell*100:.1f}%) < threshold ({self.min_confidence*100:.1f}%) or < Buy prob ({prob_buy*100:.1f}%)"

        # Check 2: Currency Strength Divergence
        if self.require_cs and currency_strength:
            clean_pair = pair.replace("=X", "").replace("/", "").upper()
            if len(clean_pair) >= 6:
                base = clean_pair[:3]
                quote = clean_pair[3:6]
                base_str = currency_strength.get(base, 0.0)
                quote_str = currency_strength.get(quote, 0.0)
                strength_diff = base_str - quote_str

                if rule_signal == 1 and strength_diff < self.min_strength_diff:
                    return 0, f"Blocked by Currency Strength: {base} ({base_str:+.4f}) not stronger than {quote} ({quote_str:+.4f}) [diff {strength_diff:+.4f} < {self.min_strength_diff}]"
                elif rule_signal == -1 and strength_diff > -self.min_strength_diff:
                    return 0, f"Blocked by Currency Strength: {base} ({base_str:+.4f}) not weaker than {quote} ({quote_str:+.4f}) [diff {strength_diff:+.4f} > -{self.min_strength_diff}]"

        # All hybrid checks passed!
        direction_str = "BUY" if rule_signal == 1 else "SELL"
        ml_prob = prob_buy if rule_signal == 1 else prob_sell
        return rule_signal, f"HYBRID CONFIRMED {direction_str} | Rule: {rule_reason} | ML Conf: {ml_prob*100:.1f}%"

    def generate_hybrid_signals(
        self,
        df: pd.DataFrame,
        ml_signals: pd.DataFrame,
        currency_strength: Optional[Dict[str, float]] = None,
        pair: str = "EURUSD",
    ) -> pd.DataFrame:
        """
        Fast historical signal generation for backtesting.
        Vectorized evaluation of H1 Trend + ML confirmation.
        """
        result = ml_signals.copy()
        if df.empty or ml_signals.empty:
            return result

        close = df["Close"]
        ema_50 = df["ema_50"] if "ema_50" in df.columns else close.ewm(span=50, adjust=False).mean()
        ema_200 = df["ema_200"] if "ema_200" in df.columns else close.ewm(span=200, adjust=False).mean()
        ema_20 = df["ema_20"] if "ema_20" in df.columns else close.ewm(span=20, adjust=False).mean()
        adx = df["adx"] if "adx" in df.columns else pd.Series(25.0, index=df.index)
        rsi = df["rsi_14"] if "rsi_14" in df.columns else (df["rsi"] if "rsi" in df.columns else pd.Series(50.0, index=df.index))

        high_series = df["High"] if "High" in df.columns else df["Close"]
        low_series = df["Low"] if "Low" in df.columns else df["Close"]

        recent_high = high_series.shift(1).rolling(20, min_periods=5).max()
        recent_low = low_series.shift(1).rolling(20, min_periods=5).min()

        bullish_trend = (close > ema_50) & (ema_50 > ema_200) & (adx >= self.h1_trend.min_adx) & (rsi <= self.h1_trend.rsi_max)
        bearish_trend = (close < ema_50) & (ema_50 < ema_200) & (adx >= self.h1_trend.min_adx) & (rsi >= self.h1_trend.rsi_min)

        bullish_trigger = (close >= recent_high) | ((low_series.shift(1) <= ema_20) & (close > ema_20))
        bearish_trigger = (close <= recent_low) | ((high_series.shift(1) >= ema_20) & (close < ema_20))

        rule_signals = pd.Series(0, index=df.index)
        rule_signals[bullish_trend & bullish_trigger] = 1
        rule_signals[bearish_trend & bearish_trigger] = -1

        # Align with ml_signals
        rule_signals = rule_signals.reindex(ml_signals.index).fillna(0).astype(int)

        prob_buy = ml_signals.get("prob_buy", pd.Series(0.0, index=ml_signals.index))
        prob_sell = ml_signals.get("prob_sell", pd.Series(0.0, index=ml_signals.index))

        final_signals = pd.Series(0, index=ml_signals.index)
        if self.require_ml:
            valid_buy = (rule_signals == 1) & (prob_buy >= self.min_confidence) & (prob_buy > prob_sell)
            valid_sell = (rule_signals == -1) & (prob_sell >= self.min_confidence) & (prob_sell > prob_buy)
        else:
            valid_buy = (rule_signals == 1)
            valid_sell = (rule_signals == -1)

        final_signals[valid_buy] = 1
        final_signals[valid_sell] = -1

        # Currency strength alignment filter
        if self.require_cs and currency_strength is not None:
            clean_pair = pair.replace("=X", "").replace("/", "").upper()
            if len(clean_pair) >= 6:
                base = clean_pair[:3]
                quote = clean_pair[3:6]
                diff = currency_strength.get(base, 0.0) - currency_strength.get(quote, 0.0)
                if diff < self.min_strength_diff:
                    final_signals[final_signals == 1] = 0
                if diff > -self.min_strength_diff:
                    final_signals[final_signals == -1] = 0

        result["signal"] = final_signals
        return result

