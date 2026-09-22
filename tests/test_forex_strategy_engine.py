"""
test_forex_strategy_engine.py — Unit Tests for Hybrid Strategy Engine
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from forex_strategy_engine import H1TrendStrategy, LondonBreakoutStrategy, ForexStrategyEngine


def test_h1_trend_strategy_adx_filter():
    """Verify ADX < min_adx blocks trend trades."""
    strat = H1TrendStrategy(min_adx=25.0)

    # 60 candles of synthetic data with low ADX
    dates = pd.date_range("2026-01-01", periods=60, freq="1h")
    df = pd.DataFrame({
        "Close": np.linspace(1.10, 1.15, 60),
        "High": np.linspace(1.11, 1.16, 60),
        "Low": np.linspace(1.09, 1.14, 60),
        "ema_50": np.full(60, 1.12),
        "ema_200": np.full(60, 1.10),
        "ema_20": np.full(60, 1.13),
        "adx": np.full(60, 18.0),  # Below 25.0
        "rsi_14": np.full(60, 52.0),
    }, index=dates)

    res = strat.generate_signal(df, "EURUSD")
    assert res["signal"] == 0
    assert "ADX" in res["reason"]


def test_h1_trend_strategy_bullish_breakout():
    """Verify clean breakout with ADX >= 25 generates BUY signal."""
    strat = H1TrendStrategy(min_adx=22.0)

    dates = pd.date_range("2026-01-01", periods=60, freq="1h")
    closes = np.full(60, 1.1400)
    highs = np.full(60, 1.1420)
    lows = np.full(60, 1.1380)

    # Last bar breaks out above recent highs
    closes[-1] = 1.1450
    highs[-1] = 1.1455

    df = pd.DataFrame({
        "Close": closes,
        "High": highs,
        "Low": lows,
        "ema_50": np.full(60, 1.1350),
        "ema_200": np.full(60, 1.1200),
        "ema_20": np.full(60, 1.1380),
        "adx": np.full(60, 28.0),
        "rsi_14": np.full(60, 58.0),
    }, index=dates)

    res = strat.generate_signal(df, "EURUSD")
    assert res["signal"] == 1
    assert "Bullish" in res["reason"]


def test_london_breakout_outside_window():
    """Verify London breakout holds when outside 07:00-11:00 UTC."""
    strat = LondonBreakoutStrategy()
    dates = pd.date_range("2026-01-01 00:00", periods=20, freq="1h")
    df = pd.DataFrame({
        "Close": np.full(20, 1.14),
        "High": np.full(20, 1.145),
        "Low": np.full(20, 1.135),
    }, index=dates)

    outside_time = datetime(2026, 1, 1, 14, 30)  # 14:30 UTC
    res = strat.generate_signal(df, "EURUSD", current_time=outside_time)
    assert res["signal"] == 0
    assert "Outside London" in res["reason"]


def test_hybrid_gate_agreement_and_disagreement():
    """Verify Hybrid Gate requires both rule signal and ML confirmation."""
    engine = ForexStrategyEngine("forex_config.yaml")

    # Case 1: Rule says BUY (+1), but ML is Bearish (prob_sell > prob_buy)
    ml_bearish = {
        "signal": -1,
        "confidence": 0.45,
        "prob_buy": 0.25,
        "prob_sell": 0.45,
    }
    sig, reason = engine.evaluate_hybrid_gate(
        rule_signal=1,
        rule_reason="H1 Bullish Breakout",
        ml_prediction=ml_bearish,
        currency_strength={"EUR": 0.005, "USD": 0.001},
        pair="EURUSD",
    )
    assert sig == 0
    assert "Blocked by ML" in reason

    # Case 2: Rule says BUY (+1), ML agrees (prob_buy >= 0.40 and > prob_sell), Currency strength agrees
    ml_bullish = {
        "signal": 1,
        "confidence": 0.48,
        "prob_buy": 0.48,
        "prob_sell": 0.22,
    }
    sig2, reason2 = engine.evaluate_hybrid_gate(
        rule_signal=1,
        rule_reason="H1 Bullish Breakout",
        ml_prediction=ml_bullish,
        currency_strength={"EUR": 0.005, "USD": 0.001},
        pair="EURUSD",
    )
    assert sig2 == 1
    assert "HYBRID CONFIRMED BUY" in reason2


def test_hybrid_gate_currency_strength_contradiction():
    """Verify strong currency strength contradiction blocks trade."""
    engine = ForexStrategyEngine("forex_config.yaml")

    ml_bullish = {
        "signal": 1,
        "confidence": 0.48,
        "prob_buy": 0.48,
        "prob_sell": 0.22,
    }
    # EUR is significantly weaker than USD (-0.01 vs +0.01)
    sig, reason = engine.evaluate_hybrid_gate(
        rule_signal=1,
        rule_reason="H1 Bullish Breakout",
        ml_prediction=ml_bullish,
        currency_strength={"EUR": -0.010, "USD": 0.010},
        pair="EURUSD",
    )
    assert sig == 0
    assert "Blocked by Currency Strength" in reason
