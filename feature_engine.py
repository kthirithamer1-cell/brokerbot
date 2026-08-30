"""
feature_engine.py — Technical Indicator & Feature Engineering
=============================================================
Transforms raw OHLCV + sentiment data into 50+ ML-ready features
for model training and prediction.
"""

import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Computes technical indicators and engineered features from OHLCV data."""

    def __init__(self):
        pass

    def compute_features(
        self,
        df: pd.DataFrame,
        sentiment_features: dict | None = None,
    ) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
            sentiment_features: Optional dict of sentiment scores to add

        Returns:
            DataFrame with all engineered features (NaN rows from lookback dropped)
        """
        if df.empty:
            return df

        feat = df.copy()

        # === TREND INDICATORS ===
        self._add_moving_averages(feat)
        self._add_macd(feat)
        self._add_adx(feat)

        # === MOMENTUM INDICATORS ===
        self._add_rsi(feat)
        self._add_stochastic(feat)
        self._add_williams_r(feat)

        # === VOLATILITY INDICATORS ===
        self._add_bollinger_bands(feat)
        self._add_atr(feat)

        # === VOLUME INDICATORS ===
        self._add_volume_features(feat)

        # === PRICE-DERIVED FEATURES ===
        self._add_returns(feat)
        self._add_price_features(feat)

        # === CALENDAR FEATURES ===
        self._add_calendar_features(feat)

        # === SENTIMENT FEATURES ===
        if sentiment_features:
            for key, value in sentiment_features.items():
                feat[key] = 0.0 if (value is None or pd.isna(value)) else float(value)

        # Replace infinite values with NaN before dropping
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop NaN rows from lookback periods
        feat.dropna(inplace=True)

        # Drop raw OHLCV (keep only features + Close for labeling)
        feature_cols = [c for c in feat.columns if c not in ["Open", "High", "Low", "Volume"]]
        feat = feat[feature_cols]

        logger.info(f"Engineered {len(feat.columns)} features, {len(feat)} rows")
        return feat

    # ──────────────────────────────────────────────
    #  TREND
    # ──────────────────────────────────────────────

    def _add_moving_averages(self, df: pd.DataFrame):
        """EMA and SMA with crossover signals."""
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = ta.trend.ema_indicator(df["Close"], window=period)
            df[f"sma_{period}"] = ta.trend.sma_indicator(df["Close"], window=period)

        # Price relative to MAs (normalized)
        for period in [9, 21, 50, 200]:
            df[f"close_vs_ema_{period}"] = (df["Close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]

        # Crossover signals
        df["ema_9_21_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_50_200_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)  # Golden/Death cross

    def _add_macd(self, df: pd.DataFrame):
        """MACD line, signal line, and histogram."""
        macd = ta.trend.MACD(df["Close"])
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        df["macd_cross"] = (df["macd_line"] > df["macd_signal"]).astype(int)

    def _add_adx(self, df: pd.DataFrame):
        """Average Directional Index — trend strength."""
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

    # ──────────────────────────────────────────────
    #  MOMENTUM
    # ──────────────────────────────────────────────

    def _add_rsi(self, df: pd.DataFrame):
        """RSI with overbought/oversold zones."""
        df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
        df["rsi_7"] = ta.momentum.rsi(df["Close"], window=7)

        # Zone encoding
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    def _add_stochastic(self, df: pd.DataFrame):
        """Stochastic Oscillator %K and %D."""
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

    def _add_williams_r(self, df: pd.DataFrame):
        """Williams %R."""
        df["williams_r"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])

    # ──────────────────────────────────────────────
    #  VOLATILITY
    # ──────────────────────────────────────────────

    def _add_bollinger_bands(self, df: pd.DataFrame):
        """Bollinger Bands width and %B."""
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
        df["bb_pct_b"] = bb.bollinger_pband()

    def _add_atr(self, df: pd.DataFrame):
        """Average True Range (absolute and normalized)."""
        df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["atr_pct"] = df["atr_14"] / df["Close"]  # Normalized ATR

        # Rolling volatility
        df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()

    # ──────────────────────────────────────────────
    #  VOLUME
    # ──────────────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame):
        """Volume-based indicators."""
        # On-Balance Volume
        df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["obv_change"] = df["obv"].pct_change()

        # Volume ratio vs average
        df["volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

        # Volume trend
        df["volume_change_5d"] = df["Volume"].pct_change(5)

        # VWAP proxy (daily)
        df["vwap"] = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
        df["close_vs_vwap"] = (df["Close"] - df["vwap"]) / df["vwap"]

    # ──────────────────────────────────────────────
    #  PRICE-DERIVED
    # ──────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame):
        """Multi-period returns."""
        for period in [1, 2, 3, 5, 10, 20]:
            df[f"return_{period}d"] = df["Close"].pct_change(period)

        # Return momentum (acceleration)
        df["return_momentum"] = df["return_5d"] - df["return_10d"]

    def _add_price_features(self, df: pd.DataFrame):
        """Price position relative to range."""
        # High-Low range
        df["daily_range_pct"] = (df["High"] - df["Low"]) / df["Close"]

        # Close position within daily range
        df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-10)

        # Distance from 52-week (252 trading days) high/low
        df["high_252d"] = df["High"].rolling(252, min_periods=20).max()
        df["low_252d"] = df["Low"].rolling(252, min_periods=20).min()
        df["pct_from_52w_high"] = (df["Close"] - df["high_252d"]) / df["high_252d"]
        df["pct_from_52w_low"] = (df["Close"] - df["low_252d"]) / df["low_252d"]

        # Gap detection
        df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # ──────────────────────────────────────────────
    #  CALENDAR
    # ──────────────────────────────────────────────

    def _add_calendar_features(self, df: pd.DataFrame):
        """Day-of-week and month seasonality."""
        if isinstance(df.index, pd.DatetimeIndex):
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter
            df["is_month_start"] = df.index.is_month_start.astype(int)
            df["is_month_end"] = df.index.is_month_end.astype(int)

    # ──────────────────────────────────────────────
    #  LABELING (for ML training)
    # ──────────────────────────────────────────────

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        horizon: int = 5,
        threshold_pct: float = 1.5,
        close_col: str = "Close",
    ) -> pd.Series:
        """
        Create forward-looking labels for ML training.

        Args:
            df: DataFrame containing the close price
            horizon: Number of days to look ahead
            threshold_pct: % move threshold for BUY/SELL

        Returns:
            Series with labels: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        threshold = threshold_pct / 100.0
        future_return = df[close_col].pct_change(horizon).shift(-horizon)

        labels = pd.Series(0, index=df.index, name="label")
        labels[future_return > threshold] = 1    # BUY
        labels[future_return < -threshold] = -1  # SELL

        return labels

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names (for reference)."""
        # Generate on a small dummy to enumerate
        dummy = pd.DataFrame({
            "Open": np.random.randn(300) + 100,
            "High": np.random.randn(300) + 101,
            "Low": np.random.randn(300) + 99,
            "Close": np.random.randn(300) + 100,
            "Volume": np.random.randint(1_000_000, 10_000_000, 300),
        }, index=pd.date_range("2020-01-01", periods=300, freq="B"))
        dummy["High"] = dummy[["Open", "High", "Close"]].max(axis=1) + 0.5
        dummy["Low"] = dummy[["Open", "Low", "Close"]].min(axis=1) - 0.5
        feat = self.compute_features(dummy)
        return list(feat.columns)


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = FeatureEngine()

    # List all feature names
    features = engine.get_feature_names()
    print(f"\n=== {len(features)} Features ===")
    for i, f in enumerate(features, 1):
        print(f"  {i:>3}. {f}")
