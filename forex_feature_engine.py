"""
forex_feature_engine.py — Forex Technical Indicator & Feature Engineering
=========================================================================
Transforms raw OHLCV + macro + sentiment data into 80+ ML-ready features
specifically tuned for forex pair trading.

Feature Categories:
    1. Trend Indicators (EMA, SMA, MACD, ADX, crossovers)
    2. Momentum Indicators (RSI, Stochastic, Williams %R)
    3. Volatility Indicators (Bollinger Bands, ATR in pips)
    4. Tick Volume Indicators (normalized, session-relative)
    5. Price-Derived (pip returns, range, gaps)
    6. Session Features (Tokyo/London/NY awareness, overlap flags)
    7. Currency Strength (base vs quote divergence)
    8. Regime Detection (volatility regime, trend strength, Hurst)
    9. Statistical (skewness, kurtosis, autocorrelation, z-score)
    10. Macro Context (DXY, yields, gold, oil, risk sentiment)
    11. Candlestick Patterns (pin bars, engulfing, doji)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class ForexFeatureEngine:
    """Computes technical indicators and engineered features from forex OHLCV data."""

    # JPY pairs use 2-decimal pip size; all others use 4-decimal
    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

    def __init__(self):
        pass

    def get_pip_size(self, pair: str) -> float:
        """Get pip size for a pair."""
        clean = pair.replace("=X", "").replace("/", "").upper()
        return 0.01 if clean in self.JPY_PAIRS else 0.0001

    def compute_features(
        self,
        df: pd.DataFrame,
        pair: str = "EURUSD",
        sentiment_features: Optional[dict] = None,
        macro_df: Optional[pd.DataFrame] = None,
        currency_strength: Optional[dict] = None,
        correlation_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute all forex features from OHLCV data.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
            pair: Forex pair name (e.g., 'EURUSD') for pip calculations
            sentiment_features: Optional dict of pre-aggregated sentiment scores
            macro_df: Optional DataFrame with DXY/Gold/Oil/VIX/Yield data
            currency_strength: Optional dict of currency strength values
            correlation_data: Optional dict of cross-pair correlation info

        Returns:
            DataFrame with all engineered features (NaN rows dropped)
        """
        if df.empty:
            return df

        pip_size = self.get_pip_size(pair)
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
        self._add_atr(feat, pip_size)

        # === TICK VOLUME FEATURES ===
        self._add_volume_features(feat)
        feat = feat.copy()

        # === PRICE-DERIVED (PIP-BASED) ===
        self._add_pip_returns(feat, pip_size)
        self._add_price_features(feat, pip_size)

        # === SESSION FEATURES (FOREX-SPECIFIC) ===
        self._add_session_features(feat)

        # === CURRENCY STRENGTH FEATURES ===
        self._add_currency_strength_features(feat, pair, currency_strength)

        # === CANDLESTICK PATTERN FEATURES ===
        self._add_candlestick_patterns(feat)

        # === MACRO CONTEXT FEATURES ===
        self._add_macro_features(feat, macro_df)

        # === REGIME DETECTION ===
        self._add_regime_features(feat)

        # === STATISTICAL FEATURES ===
        self._add_statistical_features(feat)
        feat = feat.copy()

        # === SENTIMENT/NEWS FEATURES ===
        self._add_sentiment_features(feat, sentiment_features)

        # === CROSS-PAIR CORRELATION FEATURES ===
        self._add_correlation_features(feat, pair, correlation_data)

        # Defragment DataFrame
        feat = feat.copy()

        # Replace infinite values with NaN
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop NaN rows from lookback periods
        feat.dropna(inplace=True)

        # Drop raw OHLCV (keep Close for labeling)
        feature_cols = [c for c in feat.columns if c not in ["Open", "High", "Low", "Volume"]]
        feat = feat[feature_cols].copy()

        logger.info(f"Engineered {len(feat.columns)} forex features, {len(feat)} rows")
        return feat

    # ──────────────────────────────────────────────
    #  TREND INDICATORS
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
        df["ema_50_200_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)

    def _add_macd(self, df: pd.DataFrame):
        """MACD with histogram."""
        macd = ta.trend.MACD(df["Close"])
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        df["macd_cross"] = (df["macd_line"] > df["macd_signal"]).astype(int)

    def _add_adx(self, df: pd.DataFrame, period: int = 14):
        """Average Directional Index aligned with MetaTrader 5 standard iADX indicator."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up = high - high.shift(1)
        down = low.shift(1) - low

        pdm = np.where((up > down) & (up > 0), up, 0.0)
        mdm = np.where((down > up) & (down > 0), down, 0.0)

        # MetaTrader 5 standard iADX uses Exponential Moving Average (EMA)
        tr_ema = tr.ewm(span=period, adjust=False).mean()
        pdm_ema = pd.Series(pdm, index=df.index).ewm(span=period, adjust=False).mean()
        mdm_ema = pd.Series(mdm, index=df.index).ewm(span=period, adjust=False).mean()

        pdi = 100 * (pdm_ema / (tr_ema + 1e-10))
        mdi = 100 * (mdm_ema / (tr_ema + 1e-10))

        dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        df["adx"] = adx
        df["adx_pos"] = pdi
        df["adx_neg"] = mdi

    # ──────────────────────────────────────────────
    #  MOMENTUM INDICATORS
    # ──────────────────────────────────────────────

    def _add_rsi(self, df: pd.DataFrame):
        """RSI with overbought/oversold zones."""
        df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
        df["rsi_7"] = ta.momentum.rsi(df["Close"], window=7)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    def _add_stochastic(self, df: pd.DataFrame):
        """Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

    def _add_williams_r(self, df: pd.DataFrame):
        """Williams %R."""
        df["williams_r"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])

    # ──────────────────────────────────────────────
    #  VOLATILITY INDICATORS
    # ──────────────────────────────────────────────

    def _add_bollinger_bands(self, df: pd.DataFrame):
        """Bollinger Bands width and %B."""
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
        df["bb_pct_b"] = bb.bollinger_pband()

    def _add_atr(self, df: pd.DataFrame, pip_size: float):
        """Average True Range in both price and pip terms."""
        df["atr_14"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=14
        )
        df["atr_pct"] = df["atr_14"] / df["Close"]

        # ATR in pips (forex-specific)
        df["atr_14_pips"] = df["atr_14"] / pip_size

        # Rolling volatility
        df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()

        # ATR ratio: current ATR vs longer-term ATR (volatility expansion/contraction)
        atr_50 = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=50
        )
        df["atr_ratio_14_50"] = df["atr_14"] / (atr_50 + 1e-10)

    # ──────────────────────────────────────────────
    #  TICK VOLUME FEATURES
    # ──────────────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame):
        """
        Tick volume features for forex.
        Note: yfinance forex volume is tick volume (number of price changes),
        not real traded volume. We normalize it relative to session averages.
        """
        if "Volume" not in df.columns or df["Volume"].sum() == 0:
            # No volume data — fill with neutral values
            df["tick_volume_ratio"] = 1.0
            df["tick_volume_sma_20"] = 0.0
            df["tick_volume_change_5"] = 0.0
            df["obv"] = 0.0
            df["obv_change"] = 0.0
            return

        # Tick volume ratio vs 20-bar average
        df["tick_volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["tick_volume_ratio"] = df["Volume"] / (df["tick_volume_sma_20"] + 1e-10)

        # Volume change over 5 bars
        df["tick_volume_change_5"] = df["Volume"].pct_change(5)

        # On-Balance Volume (still useful with tick volume)
        df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["obv_change"] = df["obv"].pct_change()

        # Volume-price divergence: high volume but no price movement = indecision
        price_move = abs(df["Close"] - df["Open"])
        bar_range = df["High"] - df["Low"] + 1e-10
        df["volume_price_efficiency"] = price_move / bar_range

    # ──────────────────────────────────────────────
    #  PIP-BASED RETURNS
    # ──────────────────────────────────────────────

    def _add_pip_returns(self, df: pd.DataFrame, pip_size: float):
        """Multi-period returns in both percentage and pip terms."""
        for period in [1, 2, 3, 5, 10, 20]:
            df[f"return_{period}"] = df["Close"].pct_change(period)
            # Pip-based returns (forex standard)
            df[f"pip_return_{period}"] = (df["Close"] - df["Close"].shift(period)) / pip_size

        # Return momentum (acceleration)
        df["return_momentum"] = df["return_5"] - df["return_10"]
        df["pip_momentum"] = df["pip_return_5"] - df["pip_return_10"]

    def _add_price_features(self, df: pd.DataFrame, pip_size: float):
        """Price position and range features."""
        # High-Low range in pips
        df["bar_range_pips"] = (df["High"] - df["Low"]) / pip_size

        # Close position within bar range
        df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-10)

        # Distance from recent high/low (lookback based on available data)
        lookback = min(500, len(df) - 1)  # Use up to 500 bars
        df["high_lookback"] = df["High"].rolling(lookback, min_periods=20).max()
        df["low_lookback"] = df["Low"].rolling(lookback, min_periods=20).min()
        df["pct_from_high"] = (df["Close"] - df["high_lookback"]) / df["high_lookback"]
        df["pct_from_low"] = (df["Close"] - df["low_lookback"]) / df["low_lookback"]

        # Gap detection (open vs previous close)
        df["gap_pips"] = (df["Open"] - df["Close"].shift(1)) / pip_size

    # ──────────────────────────────────────────────
    #  SESSION FEATURES (FOREX-SPECIFIC)
    # ──────────────────────────────────────────────

    def _add_session_features(self, df: pd.DataFrame):
        """
        Forex trading session awareness.
        Sessions (UTC):
            Tokyo:  00:00 - 09:00
            London: 07:00 - 16:00
            New York: 12:00 - 21:00
            London/NY Overlap: 12:00 - 16:00
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df["session_tokyo"] = 0
            df["session_london"] = 0
            df["session_new_york"] = 0
            df["session_overlap"] = 0
            df["sin_hour"] = 0.0
            df["cos_hour"] = 0.0
            df["day_of_week"] = 0.0
            df["sin_day_of_week"] = 0.0
            df["cos_day_of_week"] = 0.0
            return

        # Get hour in UTC (or convert from local)
        idx = df.index
        if idx.tz is not None:
            hour = idx.tz_convert("UTC").hour
        else:
            hour = idx.hour  # Assume UTC if no timezone

        # Session flags
        df["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
        df["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
        df["session_new_york"] = ((hour >= 12) & (hour < 21)).astype(int)
        df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)  # London/NY overlap

        # Number of active sessions (0-3)
        df["active_sessions"] = (
            df["session_tokyo"] + df["session_london"] + df["session_new_york"]
        )

        # Cyclical time encoding
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (0=Mon, 4=Fri)
        dow = idx.dayofweek
        df["day_of_week"] = dow.astype(float)
        df["sin_day_of_week"] = np.sin(2 * np.pi * dow / 5)
        df["cos_day_of_week"] = np.cos(2 * np.pi * dow / 5)

        # Friday flag (increased risk of weekend gaps)
        df["is_friday"] = (dow == 4).astype(int)

        # Monday opening flag (potential gap)
        df["is_monday_open"] = ((dow == 0) & (hour < 3)).astype(int)

    # ──────────────────────────────────────────────
    #  CURRENCY STRENGTH FEATURES
    # ──────────────────────────────────────────────

    def _add_currency_strength_features(
        self, df: pd.DataFrame, pair: str,
        currency_strength: Optional[dict] = None,
    ):
        """
        Add base and quote currency strength features.
        Divergence between base strength and quote strength is a strong signal.
        """
        clean_pair = pair.replace("=X", "").replace("/", "").upper()
        base = clean_pair[:3]
        quote = clean_pair[3:]

        if currency_strength and base in currency_strength and quote in currency_strength:
            base_strength = float(currency_strength[base])
            quote_strength = float(currency_strength[quote])
            df["base_currency_strength"] = base_strength
            df["quote_currency_strength"] = quote_strength
            df["currency_strength_divergence"] = base_strength - quote_strength
        else:
            df["base_currency_strength"] = 0.0
            df["quote_currency_strength"] = 0.0
            df["currency_strength_divergence"] = 0.0

    # ──────────────────────────────────────────────
    #  CANDLESTICK PATTERNS
    # ──────────────────────────────────────────────

    def _add_candlestick_patterns(self, df: pd.DataFrame):
        """Detect key candlestick patterns for forex."""
        body = df["Close"] - df["Open"]
        body_abs = abs(body)
        upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
        lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]
        total_range = df["High"] - df["Low"] + 1e-10

        # Doji: small body relative to range
        df["is_doji"] = (body_abs / total_range < 0.1).astype(int)

        # Pin bar (hammer/shooting star): long wick, small body
        df["is_pin_bar_bull"] = (
            (lower_wick > 2 * body_abs) & (upper_wick < body_abs) & (body > 0)
        ).astype(int)
        df["is_pin_bar_bear"] = (
            (upper_wick > 2 * body_abs) & (lower_wick < body_abs) & (body < 0)
        ).astype(int)

        # Engulfing pattern
        prev_body = body.shift(1)
        df["is_bullish_engulfing"] = (
            (body > 0) & (prev_body < 0) & (body_abs > abs(prev_body))
        ).astype(int)
        df["is_bearish_engulfing"] = (
            (body < 0) & (prev_body > 0) & (body_abs > abs(prev_body))
        ).astype(int)

        # Body-to-wick ratio
        total_wick = upper_wick + lower_wick + 1e-10
        df["body_to_wick_ratio"] = body_abs / total_wick

        # Bar direction streak
        is_up = (df["Close"] > df["Open"]).astype(int)
        df["consecutive_up_bars"] = is_up.groupby(
            (is_up != is_up.shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_up_bars"] = df["consecutive_up_bars"] * is_up
        df["consecutive_down_bars"] = (1 - is_up).groupby(
            ((1 - is_up) != (1 - is_up).shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_down_bars"] = df["consecutive_down_bars"] * (1 - is_up)

    # ──────────────────────────────────────────────
    #  MACRO CONTEXT FEATURES
    # ──────────────────────────────────────────────

    def _add_macro_features(self, df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None):
        """
        Add macro regime context features: DXY, yields, gold, oil, VIX, SPY.
        """
        macro_cols = [
            "dxy_level", "dxy_return_1d", "dxy_return_5d", "dxy_trend",
            "us10y_level", "us10y_change_5d",
            "gold_return_1d", "gold_return_5d",
            "oil_return_1d", "oil_return_5d",
            "vix_level", "vix_change_5d",
            "spy_return_1d", "spy_return_5d",
        ]

        if macro_df is not None and not macro_df.empty:
            try:
                m = macro_df.copy()
                if isinstance(m.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
                    df_idx = (df.index.tz_convert(None) if df.index.tz is not None else df.index).astype("datetime64[ns]")
                    m_idx = (m.index.tz_convert(None) if m.index.tz is not None else m.index).astype("datetime64[ns]")

                    m_clean = m.copy()
                    m_clean["_macro_dt"] = m_idx
                    m_clean = m_clean.sort_values("_macro_dt").drop_duplicates(subset=["_macro_dt"])

                    target_df = pd.DataFrame({"_target_dt": df_idx}, index=df.index)
                    merged = pd.merge_asof(
                        target_df.sort_values("_target_dt"),
                        m_clean,
                        left_on="_target_dt",
                        right_on="_macro_dt",
                        direction="backward",
                    )
                    merged.index = target_df.sort_values("_target_dt").index
                    merged = merged.reindex(df.index)

                    for col in macro_cols:
                        default_val = self._macro_default(col)
                        if col in merged.columns:
                            s = merged[col].ffill().bfill().fillna(default_val)
                            df[col] = s.values
                        else:
                            df[col] = default_val
                else:
                    for col in macro_cols:
                        df[col] = self._macro_default(col)
            except Exception as e:
                logger.warning(f"Macro feature join failed: {e}")
                for col in macro_cols:
                    df[col] = self._macro_default(col)
        else:
            for col in macro_cols:
                df[col] = self._macro_default(col)

    @staticmethod
    def _macro_default(col: str) -> float:
        """Default value for macro columns."""
        if "vix_level" in col:
            return 15.0
        elif "dxy_level" in col:
            return 100.0
        elif "us10y_level" in col:
            return 4.0
        return 0.0

    # ──────────────────────────────────────────────
    #  REGIME DETECTION
    # ──────────────────────────────────────────────

    def _add_regime_features(self, df: pd.DataFrame):
        """Regime detection features."""
        # Volatility percentile
        atr_pct = df.get("atr_pct")
        if atr_pct is None:
            atr_pct = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            ) / df["Close"]

        # Fast vectorized volatility percentile using rolling rank
        df["volatility_percentile"] = atr_pct.rolling(50, min_periods=20).rank(pct=True)

        # Volatility regime: 0=low, 1=medium, 2=high
        df["volatility_regime"] = pd.cut(
            df["volatility_percentile"],
            bins=[-0.01, 0.25, 0.75, 1.01],
            labels=[0, 1, 2],
        ).astype(float)

        # Trend regime from ADX
        adx = df.get("adx")
        if adx is None:
            adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
            adx = adx_ind.adx()

        df["trend_regime"] = pd.cut(
            adx,
            bins=[-0.01, 20, 30, 100],
            labels=[0, 1, 2],
        ).astype(float)

        # Mean reversion score
        rsi = df.get("rsi_14", ta.momentum.rsi(df["Close"], window=14))
        bb_pct = df.get("bb_pct_b")
        if bb_pct is None:
            bb = ta.volatility.BollingerBands(df["Close"])
            bb_pct = bb.bollinger_pband()
        df["mean_reversion_score"] = (rsi / 100 + bb_pct) / 2

        # Hurst exponent estimate (sampled every 40 bars for fast vectorization)
        hurst_raw = pd.Series(np.nan, index=df.index)
        close_arr = df["Close"].values
        step = 40  # Compute every 40th bar and interpolate
        for i in range(100, len(close_arr), step):
            hurst_raw.iloc[i] = self._estimate_hurst(close_arr[i-100:i])
        df["hurst_estimate"] = hurst_raw.interpolate(method="linear").ffill().bfill().fillna(0.5)

        # Regime composite
        df["regime_composite"] = (
            df["volatility_regime"].fillna(1) * 0.4
            + df["trend_regime"].fillna(0) * 0.4
            + df["hurst_estimate"].fillna(0.5) * 0.2 * 4
        )

    @staticmethod
    def _estimate_hurst(prices: np.ndarray) -> float:
        """Simplified Hurst exponent estimation using R/S analysis."""
        if len(prices) < 20:
            return 0.5
        try:
            returns = np.diff(np.log(prices + 1e-10))
            n = len(returns)
            if n < 10:
                return 0.5

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
            slope = np.polyfit(log_n, log_rs, 1)[0]
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return 0.5

    # ──────────────────────────────────────────────
    #  STATISTICAL FEATURES
    # ──────────────────────────────────────────────

    def _add_statistical_features(self, df: pd.DataFrame):
        """Statistical distribution features."""
        returns = df["Close"].pct_change()

        df["return_skewness_20"] = returns.rolling(20).skew()
        df["return_kurtosis_20"] = returns.rolling(20).kurt()

        # Autocorrelation lag-1 (vectorized via rolling cov/var)
        returns_shifted = returns.shift(1)
        rolling_cov = returns.rolling(20).cov(returns_shifted)
        rolling_var = returns.rolling(20).var()
        df["autocorrelation_lag1"] = (rolling_cov / (rolling_var + 1e-10)).fillna(0)

        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std = df["Close"].rolling(20).std()
        df["zscore_20"] = (df["Close"] - rolling_mean) / (rolling_std + 1e-10)

        # Momentum divergence
        price_mom = df["Close"].pct_change(5)
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            vol_mom = df["Volume"].pct_change(5)
            df["price_volume_divergence"] = price_mom - vol_mom.clip(-1, 1)
        else:
            df["price_volume_divergence"] = 0.0

    # ──────────────────────────────────────────────
    #  SENTIMENT FEATURES
    # ──────────────────────────────────────────────

    def _add_sentiment_features(
        self, df: pd.DataFrame, sentiment_features: Optional[dict] = None
    ):
        """Add news sentiment features."""
        defaults = {
            "sentiment_score": 0.0,
            "news_count_3d": 0.0,
            "sentiment_momentum": 0.0,
            "max_positive_score": 0.0,
            "max_negative_score": 0.0,
            "catalyst_rate_decision": 0.0,
            "catalyst_inflation": 0.0,
            "catalyst_employment": 0.0,
        }

        merged = dict(defaults)
        if sentiment_features:
            for k, v in sentiment_features.items():
                if k in merged:
                    merged[k] = 0.0 if (v is None or pd.isna(v)) else float(v)

        for col, val in merged.items():
            df[col] = val

    # ──────────────────────────────────────────────
    #  CORRELATION FEATURES
    # ──────────────────────────────────────────────

    def _add_correlation_features(
        self, df: pd.DataFrame, pair: str,
        correlation_data: Optional[dict] = None,
    ):
        """Add cross-pair correlation features."""
        if correlation_data and pair in correlation_data:
            df["avg_pair_correlation"] = float(correlation_data[pair].get("avg_correlation", 0.5))
            df["max_pair_correlation"] = float(correlation_data[pair].get("max_correlation", 0.7))
        else:
            df["avg_pair_correlation"] = 0.5
            df["max_pair_correlation"] = 0.7

    # ──────────────────────────────────────────────
    #  LABELING (for ML training)
    # ──────────────────────────────────────────────

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        pair: str = "EURUSD",
        horizon: int = 24,
        threshold_pips: float = 30.0,
        close_col: str = "Close",
        mode: str = "fixed",
        atr_multiplier: float = 1.0,
    ) -> pd.Series:
        """
        Create forward-looking labels for forex ML training.

        Args:
            df: DataFrame containing close price
            pair: Forex pair name (for pip calculation)
            horizon: Number of bars to look ahead (24 = 1 day for H1)
            threshold_pips: Pip threshold for BUY/SELL in 'fixed' mode
            close_col: Column name for close price
            mode: "fixed" = static pip threshold | "atr_relative" = dynamic ATR-based
            atr_multiplier: Multiplier for ATR-based threshold

        Returns:
            Series with labels: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        # Determine pip size
        clean = pair.replace("=X", "").replace("/", "").upper()
        jpy_pairs = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}
        pip_size = 0.01 if clean in jpy_pairs else 0.0001

        future_return = df[close_col].pct_change(horizon).shift(-horizon)

        if mode == "atr_relative":
            # Dynamic threshold based on current volatility
            if "atr_pct" in df.columns:
                atr_pct = df["atr_pct"]
            elif "atr_14" in df.columns:
                atr_pct = df["atr_14"] / df[close_col]
            else:
                atr_pct = df[close_col].pct_change().rolling(14).std() * np.sqrt(horizon)

            threshold = atr_pct * atr_multiplier
            threshold = threshold.clip(lower=0.001)  # Min 0.1% threshold

            labels = pd.Series(0, index=df.index, dtype=int)
            labels[future_return > threshold] = 1
            labels[future_return < -threshold] = -1
        else:
            # Fixed pip threshold
            threshold_pct = (threshold_pips * pip_size) / df[close_col]
            labels = pd.Series(0, index=df.index, dtype=int)
            labels[future_return > threshold_pct] = 1
            labels[future_return < -threshold_pct] = -1

        # Clear labels for last `horizon` bars (no future data)
        labels.iloc[-horizon:] = np.nan

        return labels


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from forex_data_loader import ForexDataLoader

    loader = ForexDataLoader()
    engine = ForexFeatureEngine()

    # Fetch EUR/USD data
    df = loader.fetch_forex_data("EURUSD", period="60d", interval="1h")
    if not df.empty:
        print(f"\nRaw data: {len(df)} bars, {list(df.columns)}")

        # Compute features
        features = engine.compute_features(df, pair="EURUSD")
        print(f"Features: {len(features)} rows, {len(features.columns)} columns")
        print(f"\nFeature list:")
        for i, col in enumerate(sorted(features.columns)):
            print(f"  {i+1:3d}. {col}")

        # Create labels
        labels = engine.create_labels(
            features, pair="EURUSD", horizon=24,
            mode="atr_relative", atr_multiplier=1.0
        )
        print(f"\nLabel distribution:")
        print(labels.value_counts().sort_index())
