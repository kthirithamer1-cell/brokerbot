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


def test_liquidity_and_macro_and_fundamental_features(sample_ohlcv_data):
    engine = FeatureEngine()

    macro_data = pd.DataFrame(
        {
            "spy_return_1d": [0.005] * len(sample_ohlcv_data),
            "spy_return_5d": [0.015] * len(sample_ohlcv_data),
            "qqq_return_1d": [0.008] * len(sample_ohlcv_data),
            "qqq_return_5d": [0.020] * len(sample_ohlcv_data),
            "iwm_return_1d": [0.012] * len(sample_ohlcv_data),
            "iwm_return_5d": [0.025] * len(sample_ohlcv_data),
            "vix_level": [14.2] * len(sample_ohlcv_data),
            "vix_change_5d": [-0.05] * len(sample_ohlcv_data),
        },
        index=sample_ohlcv_data.index.normalize(),
    )

    fund_data = {
        "float_shares": 8_500_000,
        "shares_outstanding": 10_000_000,
        "short_pct_float": 0.22,
        "short_ratio": 3.4,
        "insider_pct": 0.12,
        "institution_pct": 0.35,
    }

    quote_data = {
        "rel_volume": 2.85,
        "spread_pct": 1.25,
        "day_range_pos": 0.88,
        "year_range_pos": 0.72,
    }

    news = [
        {
            "title": "BioCorp gets FDA approval for breakthrough oncology treatment",
            "description": "Company announces major phase 3 clinical trial success and FDA nod.",
            "published_at": "2024-01-02T10:00:00Z",
            "sentiment_positive": 0.92,
            "sentiment_negative": 0.02,
        }
    ]

    features = engine.compute_features(
        sample_ohlcv_data,
        macro_df=macro_data,
        fundamental_data=fund_data,
        quote_info=quote_data,
        news_articles=news,
        as_of_time=pd.Timestamp("2024-01-02 12:00:00", tz="UTC"),
    )

    # 1. Microstructure / Liquidity
    assert "rel_volume" in features.columns
    assert "spread_pct" in features.columns
    assert "day_range_pos" in features.columns
    assert "year_range_pos" in features.columns
    assert features["rel_volume"].iloc[-1] == 2.85

    # 2. Macro Regime
    assert "spy_return_1d" in features.columns
    assert "iwm_return_1d" in features.columns
    assert "vix_level" in features.columns
    assert "relative_strength_spy_1d" in features.columns
    assert "relative_strength_iwm_1d" in features.columns

    # 3. Fundamentals
    assert "float_shares" in features.columns
    assert features["float_ratio"].iloc[0] == 0.85
    assert features["short_pct_float"].iloc[0] == 0.22

    # 4. News & Catalysts
    assert "catalyst_fda" in features.columns
    assert "catalyst_dilution" in features.columns
    assert features["catalyst_fda"].iloc[-1] == 1.0
    assert features["catalyst_dilution"].iloc[-1] == 0.0
    assert "days_since_last_news" in features.columns


def test_lookahead_leakage_protection():
    """Verify that articles published in the future are strictly ignored."""
    from sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()

    future_news = [
        {
            "title": "Massive share dilution offering announced",
            "description": "Registered direct offering at deep discount",
            "published_at": "2024-01-05T12:00:00Z",
            "sentiment_positive": 0.05,
            "sentiment_negative": 0.95,
        }
    ]

    # As of Jan 2 (before the news was published on Jan 5):
    result_before = analyzer.compute_aggregate_sentiment(
        future_news,
        as_of_time=pd.Timestamp("2024-01-02 12:00:00", tz="UTC"),
    )

    # Must NOT see the future dilution catalyst or negative sentiment
    assert result_before["catalyst_dilution"] == 0
    assert result_before["news_count_3d"] == 0
    assert result_before["days_since_last_news"] == 30.0

    # As of Jan 6 (after news was published):
    result_after = analyzer.compute_aggregate_sentiment(
        future_news,
        as_of_time=pd.Timestamp("2024-01-06 12:00:00", tz="UTC"),
    )

    assert result_after["catalyst_dilution"] == 1
    assert result_after["news_count_3d"] == 1

