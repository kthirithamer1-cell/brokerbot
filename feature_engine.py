"""
feature_engine.py — Technical Indicator & Feature Engineering
=============================================================
Transforms raw OHLCV + sentiment data into 80+ ML-ready features
for model training and prediction.

Feature Categories:
    1. Trend Indicators (EMA, SMA, MACD, ADX, crossovers)
    2. Momentum Indicators (RSI, Stochastic, Williams %R)
    3. Volatility Indicators (Bollinger Bands, ATR, rolling vol)
    4. Volume Indicators (OBV, volume ratio, VWAP)
    5. Price-Derived (returns, range, gaps, 52-week position)
    6. Calendar/Cyclical (sin/cos encoding, session flags, OPEX)
    7. Microstructure (spread proxy, tick intensity, price efficiency)
    8. Regime Detection (volatility regime, trend strength, Hurst)
    9. Statistical (skewness, kurtosis, autocorrelation, z-score)
    10. Intraday Patterns (opening range, first-hour momentum)
"""

import logging
from typing import Optional

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
        higher_tf_data: dict | None = None,
    ) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
            sentiment_features: Optional dict of sentiment scores to add
            higher_tf_data: Optional dict with higher-timeframe context
                            e.g. {"daily_ema_trend": 1, "daily_rsi": 55, "weekly_momentum": 0.02}

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

        # === CALENDAR FEATURES (cyclical encoding) ===
        self._add_calendar_features(feat)

        # === NEW: MICROSTRUCTURE FEATURES ===
        self._add_microstructure_features(feat)

        # === NEW: REGIME DETECTION FEATURES ===
        self._add_regime_features(feat)

        # === NEW: STATISTICAL FEATURES ===
        self._add_statistical_features(feat)

        # === NEW: INTRADAY PATTERN FEATURES ===
        self._add_intraday_features(feat)

        # === NEW: MULTI-TIMEFRAME FEATURES ===
        if higher_tf_data:
            for key, value in higher_tf_data.items():
                feat[key] = 0.0 if (value is None or (isinstance(value, float) and np.isnan(value))) else float(value)

        # === SENTIMENT FEATURES ===
        if sentiment_features:
            for key, value in sentiment_features.items():
                feat[key] = 0.0 if (value is None or pd.isna(value)) else float(value)

        # Defragment DataFrame after multiple column additions
        feat = feat.copy()

        # Replace infinite values with NaN before dropping
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop NaN rows from lookback periods
        feat.dropna(inplace=True)

        # Drop raw OHLCV (keep only features + Close for labeling)
        feature_cols = [c for c in feat.columns if c not in ["Open", "High", "Low", "Volume"]]
        feat = feat[feature_cols].copy()

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
    #  CALENDAR (cyclical encoding)
    # ──────────────────────────────────────────────

    def _add_calendar_features(self, df: pd.DataFrame):
        """Day-of-week and month seasonality with cyclical encoding."""
        if isinstance(df.index, pd.DatetimeIndex):
            # Cyclical encoding (better than raw integers for tree models)
            dow = df.index.dayofweek
            df["sin_day_of_week"] = np.sin(2 * np.pi * dow / 5)
            df["cos_day_of_week"] = np.cos(2 * np.pi * dow / 5)

            month = df.index.month
            df["sin_month"] = np.sin(2 * np.pi * month / 12)
            df["cos_month"] = np.cos(2 * np.pi * month / 12)

            df["quarter"] = df.index.quarter
            df["is_month_start"] = df.index.is_month_start.astype(int)
            df["is_month_end"] = df.index.is_month_end.astype(int)

            # Intraday time features for sub-daily timeframes
            if hasattr(df.index, 'hour'):
                hour = df.index.hour
                minute = df.index.minute
                minutes_since_open = (hour - 9) * 60 + (minute - 30)  # Minutes since 9:30 AM

                df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
                df["cos_hour"] = np.cos(2 * np.pi * hour / 24)

                # Session flags for 15m bars
                df["is_first_30min"] = ((hour == 9) & (minute >= 30) | (hour == 10) & (minute == 0)).astype(int)
                df["is_last_30min"] = ((hour == 15) & (minute >= 30)).astype(int)
                df["is_power_hour"] = ((hour >= 15) & (hour < 16)).astype(int)
                df["is_lunch_hour"] = ((hour >= 12) & (hour < 13)).astype(int)

            # Days to monthly options expiration (3rd Friday)
            df["days_to_opex"] = df.index.to_series().apply(self._days_to_opex)

    @staticmethod
    def _days_to_opex(dt) -> int:
        """Compute trading days until next monthly options expiration (3rd Friday)."""
        import calendar
        year, month = dt.year, dt.month

        # Find the 3rd Friday of current month
        cal = calendar.monthcalendar(year, month)
        # Find all Fridays (index 4 in week)
        fridays = [week[4] for week in cal if week[4] != 0]
        third_friday_day = fridays[2] if len(fridays) >= 3 else fridays[-1]

        from datetime import datetime
        opex = datetime(year, month, third_friday_day)

        if dt.replace(tzinfo=None) if hasattr(dt, 'tzinfo') and dt.tzinfo else dt > opex:
            # Move to next month
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1
            cal = calendar.monthcalendar(year, month)
            fridays = [week[4] for week in cal if week[4] != 0]
            third_friday_day = fridays[2] if len(fridays) >= 3 else fridays[-1]
            opex = datetime(year, month, third_friday_day)

        dt_naive = dt.replace(tzinfo=None) if hasattr(dt, 'tzinfo') and dt.tzinfo else dt
        delta = (opex - dt_naive).days
        return max(0, min(delta, 30))  # Cap at 30

    # ──────────────────────────────────────────────
    #  MICROSTRUCTURE (NEW)
    # ──────────────────────────────────────────────

    def _add_microstructure_features(self, df: pd.DataFrame):
        """
        Microstructure proxies from OHLCV data.
        These approximate order flow and market quality without L2 data.
        """
        hl_range = df["High"] - df["Low"]

        # Spread estimate: High-Low as proxy for bid-ask spread (normalized)
        df["spread_estimate"] = hl_range / df["Close"]

        # Tick intensity: Volume per unit of price range — measures aggression
        df["tick_intensity"] = df["Volume"] / (hl_range + 1e-10)
        # Normalize to avoid scale issues
        df["tick_intensity_zscore"] = (
            (df["tick_intensity"] - df["tick_intensity"].rolling(20).mean())
            / (df["tick_intensity"].rolling(20).std() + 1e-10)
        )

        # Price efficiency: |Close-Open| / (High-Low) — 1.0 = strong directional bar
        df["price_efficiency"] = abs(df["Close"] - df["Open"]) / (hl_range + 1e-10)

        # Cumulative delta proxy: up-volume vs down-volume accumulation
        # If close > open, volume is "up"; otherwise "down"
        up_vol = df["Volume"].where(df["Close"] >= df["Open"], 0)
        down_vol = df["Volume"].where(df["Close"] < df["Open"], 0)
        df["cum_delta_proxy"] = (up_vol - down_vol).rolling(10).sum()
        df["cum_delta_proxy_norm"] = df["cum_delta_proxy"] / (df["Volume"].rolling(10).sum() + 1e-10)

        # Amihud illiquidity: |return| / dollar volume (measures price impact)
        dollar_volume = df["Close"] * df["Volume"]
        df["amihud_illiquidity"] = abs(df["Close"].pct_change()) / (dollar_volume + 1e-10)
        df["amihud_illiquidity_20d"] = df["amihud_illiquidity"].rolling(20).mean()

    # ──────────────────────────────────────────────
    #  REGIME DETECTION (NEW)
    # ──────────────────────────────────────────────

    def _add_regime_features(self, df: pd.DataFrame):
        """
        Market regime detection features.
        Classify the market as trending, ranging, or volatile.
        """
        # Volatility regime: ATR percentile rank over lookback
        atr_pct = df.get("atr_pct")
        if atr_pct is None:
            atr_pct = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            ) / df["Close"]

        df["volatility_percentile"] = atr_pct.rolling(50, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Volatility regime bucketed: 0=low, 1=medium, 2=high
        df["volatility_regime"] = pd.cut(
            df["volatility_percentile"],
            bins=[-0.01, 0.25, 0.75, 1.01],
            labels=[0, 1, 2],
        ).astype(float)

        # Trend strength: ADX-based (uses existing ADX if available)
        adx = df.get("adx")
        if adx is None:
            adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
            adx = adx_ind.adx()

        # Trend regime: 0=ranging (<20), 1=weak trend (20-30), 2=strong trend (>30)
        df["trend_regime"] = pd.cut(
            adx,
            bins=[-0.01, 20, 30, 100],
            labels=[0, 1, 2],
        ).astype(float)

        # Mean reversion indicator: RSI + Bollinger %B composite
        rsi = df.get("rsi_14", ta.momentum.rsi(df["Close"], window=14))
        bb_pct = df.get("bb_pct_b")
        if bb_pct is None:
            bb = ta.volatility.BollingerBands(df["Close"])
            bb_pct = bb.bollinger_pband()

        # Composite: 0 = strongly oversold, 1 = strongly overbought
        df["mean_reversion_score"] = (rsi / 100 + bb_pct) / 2

        # Simplified Hurst exponent estimate
        # H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending
        df["hurst_estimate"] = df["Close"].rolling(100, min_periods=50).apply(
            self._estimate_hurst, raw=True
        )

        # Regime composite: single feature combining trend + volatility info
        # Higher = stronger trending, lower = ranging/mean-reverting
        df["regime_composite"] = (
            df["volatility_regime"].fillna(1) * 0.4
            + df["trend_regime"].fillna(0) * 0.4
            + df["hurst_estimate"].fillna(0.5) * 0.2 * 4  # Scale hurst to ~0-2
        )

    @staticmethod
    def _estimate_hurst(prices: np.ndarray) -> float:
        """
        Simplified Hurst exponent estimation using R/S analysis.
        Returns value between 0 and 1.
        """
        if len(prices) < 20:
            return 0.5

        try:
            returns = np.diff(np.log(prices + 1e-10))
            n = len(returns)
            if n < 10:
                return 0.5

            # Split into halves for R/S calculation
            max_k = min(int(n / 4), 50)
            if max_k < 2:
                return 0.5

            rs_values = []
            ns = []
            for k in range(2, max_k + 1):
                sub_n = n // k
                if sub_n < 2:
                    break
                rs_list = []
                for i in range(k):
                    sub_returns = returns[i * sub_n:(i + 1) * sub_n]
                    mean_r = np.mean(sub_returns)
                    deviations = np.cumsum(sub_returns - mean_r)
                    r = np.max(deviations) - np.min(deviations)
                    s = np.std(sub_returns, ddof=1) if np.std(sub_returns, ddof=1) > 0 else 1e-10
                    rs_list.append(r / s)
                rs_values.append(np.mean(rs_list))
                ns.append(sub_n)

            if len(rs_values) < 3:
                return 0.5

            log_n = np.log(ns)
            log_rs = np.log(np.array(rs_values) + 1e-10)

            # Linear regression for Hurst exponent
            slope = np.polyfit(log_n, log_rs, 1)[0]
            return float(np.clip(slope, 0.0, 1.0))

        except Exception:
            return 0.5

    # ──────────────────────────────────────────────
    #  STATISTICAL FEATURES (NEW)
    # ──────────────────────────────────────────────

    def _add_statistical_features(self, df: pd.DataFrame):
        """Statistical distribution features of returns."""
        returns = df["Close"].pct_change()

        # Skewness: measures tail asymmetry (negative = more downside risk)
        df["return_skewness_20d"] = returns.rolling(20).skew()

        # Kurtosis: measures fat tails (high = more extreme moves)
        df["return_kurtosis_20d"] = returns.rolling(20).kurt()

        # Autocorrelation lag-1: positive = momentum, negative = mean-reversion
        df["autocorrelation_lag1"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
        )

        # Price z-score: how many standard deviations from 20d mean
        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std = df["Close"].rolling(20).std()
        df["zscore_20d"] = (df["Close"] - rolling_mean) / (rolling_std + 1e-10)

        # Entropy proxy: rolling uniqueness of returns
        df["return_entropy"] = returns.rolling(20).apply(
            self._compute_entropy_proxy, raw=True
        )

        # Momentum divergence: price momentum vs volume momentum
        price_mom = df["Close"].pct_change(5)
        vol_mom = df["Volume"].pct_change(5)
        df["price_volume_divergence"] = price_mom - vol_mom.clip(-1, 1)

    @staticmethod
    def _compute_entropy_proxy(returns: np.ndarray) -> float:
        """Compute an entropy-like measure of return distribution complexity."""
        try:
            # Bin returns into 5 buckets and compute Shannon entropy
            counts, _ = np.histogram(returns, bins=5)
            probs = counts / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs + 1e-10)))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────
    #  INTRADAY PATTERN FEATURES (NEW)
    # ──────────────────────────────────────────────

    def _add_intraday_features(self, df: pd.DataFrame):
        """
        Intraday pattern features for sub-daily timeframes.
        These capture opening range breakouts, first-hour momentum, etc.
        """
        # Opening range: first bar's range as % of ATR
        # (approximated: current bar's O-C range vs rolling ATR)
        atr = df.get("atr_14")
        if atr is None:
            atr = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            )

        df["bar_range_vs_atr"] = (df["High"] - df["Low"]) / (atr + 1e-10)

        # Body-to-wick ratio: strong body = conviction
        body = abs(df["Close"] - df["Open"])
        upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
        lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]
        total_wick = upper_wick + lower_wick + 1e-10
        df["body_to_wick_ratio"] = body / total_wick

        # Bar direction momentum: count of consecutive up/down bars
        is_up = (df["Close"] > df["Open"]).astype(int)
        df["consecutive_up_bars"] = is_up.groupby(
            (is_up != is_up.shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_up_bars"] = df["consecutive_up_bars"] * is_up  # Only count when up
        df["consecutive_down_bars"] = (1 - is_up).groupby(
            ((1 - is_up) != (1 - is_up).shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_down_bars"] = df["consecutive_down_bars"] * (1 - is_up)

        # VWAP deviation (rolling intraday proxy)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap_session"] = (
            (typical_price * df["Volume"]).rolling(26).sum()
            / (df["Volume"].rolling(26).sum() + 1e-10)
        )
        df["vwap_deviation"] = (df["Close"] - df["vwap_session"]) / (df["vwap_session"] + 1e-10)

        # First-hour momentum proxy: 4-bar return for 15m timeframe (~first hour)
        df["first_hour_momentum"] = df["Close"].pct_change(4)

        # Relative volume at time of day (high RV = unusual activity)
        df["relative_volume_short"] = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-10)

    # ──────────────────────────────────────────────
    #  LABELING (for ML training)
    # ──────────────────────────────────────────────

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        horizon: int = 5,
        threshold_pct: float = 1.5,
        close_col: str = "Close",
        mode: str = "fixed",
        atr_multiplier: float = 1.5,
    ) -> pd.Series:
        """
        Create forward-looking labels for ML training.

        Args:
            df: DataFrame containing the close price
            horizon: Number of bars to look ahead
            threshold_pct: % move threshold for BUY/SELL (used in 'fixed' mode)
            close_col: Column name for close price
            mode: "fixed" = static threshold | "atr_relative" = dynamic ATR-based threshold
            atr_multiplier: Multiplier for ATR-based threshold (used in 'atr_relative' mode)

        Returns:
            Series with labels: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        future_return = df[close_col].pct_change(horizon).shift(-horizon)

        if mode == "atr_relative":
            # Dynamic threshold based on each stock's current volatility
            # ATR as % of price
            if "atr_pct" in df.columns:
                atr_threshold = df["atr_pct"] * atr_multiplier
            else:
                # Compute ATR from scratch if not available
                if all(col in df.columns for col in ["High", "Low"]):
                    tr = pd.concat([
                        df["High"] - df["Low"],
                        abs(df["High"] - df[close_col].shift(1)),
                        abs(df["Low"] - df[close_col].shift(1)),
                    ], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    atr_threshold = (atr / df[close_col]) * atr_multiplier
                else:
                    # Fallback to fixed threshold
                    atr_threshold = threshold_pct / 100.0

            labels = pd.Series(0, index=df.index, name="label")
            labels[future_return > atr_threshold] = 1    # BUY
            labels[future_return < -atr_threshold] = -1  # SELL
        else:
            # Original fixed threshold mode
            threshold = threshold_pct / 100.0
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
