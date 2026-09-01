import pytest
import numpy as np
import pandas as pd
from feature_engine import FeatureEngine


@pytest.fixture
def sample_ohlcv_data():
    """Generates synthetic 15-minute style OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=300, freq="15min")
    close = 10.0 + np.cumsum(np.random.randn(300) * 0.1)
    high = close + np.random.uniform(0.05, 0.2, 300)
    low = close - np.random.uniform(0.05, 0.2, 300)
    open_p = close + np.random.uniform(-0.05, 0.05, 300)
    volume = np.random.randint(50000, 500000, 300)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def test_compute_features_shape_and_columns(sample_ohlcv_data):
    engine = FeatureEngine()
    features = engine.compute_features(sample_ohlcv_data)

    assert not features.empty
    assert len(features) > 0
    # Make sure raw OHLCV are not in features (except Close or feature transformed cols)
    for raw_col in ["Open", "High", "Low", "Volume"]:
        assert raw_col not in features.columns


def test_microstructure_and_regime_features(sample_ohlcv_data):
    engine = FeatureEngine()
    features = engine.compute_features(sample_ohlcv_data)

    # Check Microstructure features
    expected_micro = [
        "spread_estimate",
        "tick_intensity",
        "price_efficiency",
        "cum_delta_proxy",
        "amihud_illiquidity",
    ]
    for col in expected_micro:
        assert col in features.columns, f"Missing microstructure feature: {col}"

    # Check Regime detection features
    expected_regime = [
        "volatility_percentile",
        "volatility_regime",
        "trend_regime",
        "mean_reversion_score",
        "hurst_estimate",
        "regime_composite",
    ]
    for col in expected_regime:
        assert col in features.columns, f"Missing regime feature: {col}"


def test_statistical_and_intraday_features(sample_ohlcv_data):
    engine = FeatureEngine()
    features = engine.compute_features(sample_ohlcv_data)

    # Statistical features
    expected_stat = [
        "return_skewness_20d",
        "return_kurtosis_20d",
        "autocorrelation_lag1",
        "zscore_20d",
        "return_entropy",
    ]
    for col in expected_stat:
        assert col in features.columns, f"Missing statistical feature: {col}"

    # Intraday pattern features
    expected_intraday = [
        "bar_range_vs_atr",
        "body_to_wick_ratio",
        "consecutive_up_bars",
        "consecutive_down_bars",
        "vwap_deviation",
        "first_hour_momentum",
    ]
    for col in expected_intraday:
        assert col in features.columns, f"Missing intraday feature: {col}"


def test_fixed_and_atr_relative_labeling(sample_ohlcv_data):
    engine = FeatureEngine()

    # Fixed labeling
    labels_fixed = engine.create_labels(
        sample_ohlcv_data, horizon=5, threshold_pct=1.0, mode="fixed"
    )
    assert isinstance(labels_fixed, pd.Series)
    assert set(labels_fixed.dropna().unique()).issubset({-1, 0, 1})

    # ATR-relative dynamic labeling
    labels_atr = engine.create_labels(
        sample_ohlcv_data, horizon=5, mode="atr_relative", atr_multiplier=1.5
    )
    assert isinstance(labels_atr, pd.Series)
    assert set(labels_atr.dropna().unique()).issubset({-1, 0, 1})


def test_higher_tf_and_sentiment_injection(sample_ohlcv_data):
    engine = FeatureEngine()
    sentiment = {"sentiment_score": 0.75, "sentiment_confidence": 0.90}
    higher_tf = {"daily_ema_trend": 1.0, "daily_rsi": 58.2}

    features = engine.compute_features(
        sample_ohlcv_data,
        sentiment_features=sentiment,
        higher_tf_data=higher_tf,
    )

    assert "sentiment_score" in features.columns
    assert "daily_ema_trend" in features.columns
    assert features["sentiment_score"].iloc[-1] == 0.75
    assert features["daily_ema_trend"].iloc[-1] == 1.0
