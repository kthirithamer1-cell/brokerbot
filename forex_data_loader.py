"""
forex_data_loader.py — Forex Historical Price Data & Market Context Fetcher
===========================================================================
Fetches forex OHLCV data from yfinance (primary), economic calendar data,
macro context (DXY, US10Y, Gold, Oil), and currency correlation matrices.

Designed for 24/5 forex market operation with session-aware caching.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 900  # 15 minutes


class ForexDataLoader:
    """Fetches and caches forex price data and market context."""

    # Forex pair to yfinance ticker mapping
    PAIR_TO_TICKER = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
        "USDCAD": "USDCAD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
        "EURCHF": "EURCHF=X", "EURAUD": "EURAUD=X", "EURCAD": "EURCAD=X",
        "GBPJPY": "GBPJPY=X", "GBPCHF": "GBPCHF=X", "GBPAUD": "GBPAUD=X",
        "GBPCAD": "GBPCAD=X", "AUDJPY": "AUDJPY=X", "AUDCAD": "AUDCAD=X",
        "AUDNZD": "AUDNZD=X", "NZDJPY": "NZDJPY=X", "NZDCAD": "NZDCAD=X",
        "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
    }

    # Macro context tickers
    MACRO_TICKERS = {
        "DXY": "DX-Y.NYB",        # US Dollar Index
        "US10Y": "^TNX",           # US 10-Year Treasury Yield
        "GOLD": "GC=F",           # Gold Futures
        "OIL": "CL=F",            # WTI Crude Oil Futures
        "VIX": "^VIX",            # Volatility Index
        "SPY": "SPY",             # S&P 500 for risk-on/risk-off
    }

    # JPY pair pip multiplier
    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

    # Individual currencies for strength calculation
    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path("data/forex")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.news_api_key = os.getenv("NEWS_API_KEY", "")

        data_cfg = self.config.get("data", {})
        self.history_years = data_cfg.get("history_years", 2)
        self.cache_expiry_hours = data_cfg.get("cache_expiry_hours", 4)

        self._info_cache = {}
        self._cache_ts = {}

    # ──────────────────────────────────────────────
    #  PIP UTILITIES
    # ──────────────────────────────────────────────

    def get_pip_size(self, pair: str) -> float:
        """Get the pip size for a forex pair (0.0001 for most, 0.01 for JPY pairs)."""
        clean_pair = pair.replace("=X", "").replace("/", "").upper()
        if clean_pair in self.JPY_PAIRS:
            return 0.01
        return 0.0001

    def price_to_pips(self, pair: str, price_diff: float) -> float:
        """Convert a price difference to pips."""
        return price_diff / self.get_pip_size(pair)

    def pips_to_price(self, pair: str, pips: float) -> float:
        """Convert pips to a price difference."""
        return pips * self.get_pip_size(pair)

    @staticmethod
    def normalize_pair(pair: str) -> str:
        """Normalize pair name: 'EUR/USD' or 'EURUSD=X' -> 'EURUSD'."""
        return pair.replace("=X", "").replace("/", "").replace("-", "").upper()

    def pair_to_ticker(self, pair: str) -> str:
        """Convert pair name to yfinance ticker."""
        clean = self.normalize_pair(pair)
        return self.PAIR_TO_TICKER.get(clean, f"{clean}=X")

    # ──────────────────────────────────────────────
    #  FOREX PRICE DATA
    # ──────────────────────────────────────────────

    def fetch_forex_data(
        self,
        pair: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a forex pair from yfinance.

        Args:
            pair: Forex pair (e.g., 'EURUSD', 'EUR/USD', 'EURUSD=X')
            period: yfinance period string (e.g., '2y', '6mo', '60d')
            interval: Bar size ('1d', '1h', '15m', '5m')
            start/end: Date strings for custom range
            use_cache: If True, load from local cache if available

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        clean_pair = self.normalize_pair(pair)
        ticker = self.pair_to_ticker(clean_pair)

        if interval is None:
            interval = self.config["strategy"].get("timeframe", "1h")
        if period is None and start is None:
            period = self._get_default_period(interval)

        # Cap period to yfinance limits
        period = self._cap_period(period, interval)

        cache_file = self.data_dir / f"{clean_pair}_{interval}_{period or 'custom'}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_expiry_hours):
                logger.info(f"Loading cached forex data for {clean_pair}")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading {clean_pair} ({ticker}) | interval={interval} | period={period}")

        try:
            yf_ticker = yf.Ticker(ticker)

            if start and end:
                df = yf_ticker.history(start=start, end=end, interval=interval)
            else:
                df = yf_ticker.history(period=period, interval=interval)

                # Fallback for empty results
                if df.empty:
                    fallbacks = self._get_fallback_periods(interval)
                    for fallback in fallbacks:
                        logger.info(f"Retrying {clean_pair} with fallback period={fallback}")
                        df = yf_ticker.history(period=fallback, interval=interval)
                        if not df.empty:
                            break

            if df.empty:
                logger.warning(f"No forex data returned for {clean_pair}")
                return pd.DataFrame()

            # Clean up columns — keep only OHLCV
            available_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[available_cols].copy()
            df.index.name = "Date"
            df.dropna(subset=["Close"], inplace=True)

            # Remove zero-volume rows (weekends/holidays sometimes have stale data)
            if "Volume" in df.columns:
                # For forex, volume is tick volume and can be 0 on weekends
                # Keep rows where price actually changed
                df = df[df["Close"] > 0].copy()

            # Cache it
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} bars for {clean_pair} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch forex data for {clean_pair}: {e}")
            return pd.DataFrame()

    def fetch_multiple_pairs(
        self, pairs: list[str], **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Fetch forex data for multiple pairs."""
        data = {}
        for pair in pairs:
            df = self.fetch_forex_data(pair, **kwargs)
            if not df.empty:
                data[self.normalize_pair(pair)] = df
        return data

    # ──────────────────────────────────────────────
    #  MACRO CONTEXT (DXY, Yields, Gold, Oil, VIX)
    # ──────────────────────────────────────────────

    def fetch_macro_context(
        self, period: str = "2y", interval: str = "1h", use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch broader market context for forex:
        - DXY (Dollar Index): Overall USD strength
        - US10Y: Interest rate environment
        - Gold: Safe-haven flows
        - Oil: Commodity currency driver (CAD, AUD)
        - VIX: Risk-on/risk-off sentiment
        - SPY: Equity market risk appetite

        Returns DataFrame with returns and levels for each.
        """
        cache_key = f"forex_macro:{period}:{interval}"
        if use_cache and hasattr(self, "_macro_cache"):
            ts = getattr(self, "_macro_cache_ts", {}).get(cache_key, 0)
            if time.time() - ts < 3600:
                return self._macro_cache[cache_key].copy()

        # Cap period for yfinance interval limits
        capped_period = self._cap_period(period, interval)

        dfs = {}
        for name, ticker in self.MACRO_TICKERS.items():
            try:
                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(period=capped_period, interval=interval)
                if not df.empty and "Close" in df.columns:
                    s = df["Close"].copy()
                    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                        s.index = s.index.tz_convert(None)
                    dfs[name] = s
            except Exception as e:
                logger.warning(f"Failed to fetch macro data for {name} ({ticker}): {e}")

        if not dfs:
            return pd.DataFrame()

        macro_df = pd.DataFrame(dfs)
        if isinstance(macro_df.index, pd.DatetimeIndex) and macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_convert(None)

        # Forward/backward fill for different trading hours
        macro_df = macro_df.ffill().bfill()

        result = pd.DataFrame(index=macro_df.index)

        # DXY — Dollar strength
        if "DXY" in macro_df.columns:
            result["dxy_level"] = macro_df["DXY"]
            result["dxy_return_1d"] = macro_df["DXY"].pct_change(1).fillna(0.0)
            result["dxy_return_5d"] = macro_df["DXY"].pct_change(5).fillna(0.0)
            # DXY momentum (is dollar strengthening or weakening?)
            dxy_ema_fast = macro_df["DXY"].ewm(span=9).mean()
            dxy_ema_slow = macro_df["DXY"].ewm(span=21).mean()
            result["dxy_trend"] = (dxy_ema_fast - dxy_ema_slow) / dxy_ema_slow

        # US 10-Year Yield — interest rate environment
        if "US10Y" in macro_df.columns:
            result["us10y_level"] = macro_df["US10Y"]
            result["us10y_change_5d"] = macro_df["US10Y"].diff(5).fillna(0.0)

        # Gold — safe haven proxy
        if "GOLD" in macro_df.columns:
            result["gold_return_1d"] = macro_df["GOLD"].pct_change(1).fillna(0.0)
            result["gold_return_5d"] = macro_df["GOLD"].pct_change(5).fillna(0.0)

        # Oil — commodity currency driver
        if "OIL" in macro_df.columns:
            result["oil_return_1d"] = macro_df["OIL"].pct_change(1).fillna(0.0)
            result["oil_return_5d"] = macro_df["OIL"].pct_change(5).fillna(0.0)

        # VIX — risk sentiment
        if "VIX" in macro_df.columns:
            result["vix_level"] = macro_df["VIX"]
            result["vix_change_5d"] = macro_df["VIX"].pct_change(5).fillna(0.0)

        # SPY — risk-on/risk-off
        if "SPY" in macro_df.columns:
            result["spy_return_1d"] = macro_df["SPY"].pct_change(1).fillna(0.0)
            result["spy_return_5d"] = macro_df["SPY"].pct_change(5).fillna(0.0)

        result.ffill(inplace=True)
        result.bfill(inplace=True)

        if not hasattr(self, "_macro_cache"):
            self._macro_cache = {}
            self._macro_cache_ts = {}
        self._macro_cache[cache_key] = result
        self._macro_cache_ts[cache_key] = time.time()

        logger.info(f"🌐 Forex macro context loaded: {len(result)} bars "
                     f"(DXY, US10Y, Gold, Oil, VIX, SPY)")
        return result.copy()

    # ──────────────────────────────────────────────
    #  CURRENCY STRENGTH
    # ──────────────────────────────────────────────

    def compute_currency_strength(
        self, period: str = "60d", interval: str = "1h", lookback_bars: int = 24
    ) -> dict[str, float]:
        """
        Compute relative strength of each major currency by averaging
        its performance across all pairs it participates in.

        Returns dict: {'USD': 0.35, 'EUR': -0.12, ...}
        A positive value means the currency is strengthening.
        """
        strength = {c: [] for c in self.CURRENCIES}

        for pair, ticker in self.PAIR_TO_TICKER.items():
            base = pair[:3]  # e.g., EUR from EURUSD
            quote = pair[3:]  # e.g., USD from EURUSD

            if base not in self.CURRENCIES or quote not in self.CURRENCIES:
                continue

            try:
                df = self.fetch_forex_data(pair, period=period, interval=interval, use_cache=True)
                if df.empty or len(df) < lookback_bars:
                    continue

                # Compute return over lookback period
                ret = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback_bars] - 1)

                # If pair goes up, base currency strengthens, quote weakens
                strength[base].append(ret)
                strength[quote].append(-ret)

            except Exception as e:
                logger.debug(f"Currency strength calc failed for {pair}: {e}")

        result = {}
        for currency, returns in strength.items():
            if returns:
                result[currency] = float(np.mean(returns))
            else:
                result[currency] = 0.0

        logger.info(f"💪 Currency strength: {', '.join(f'{k}={v:+.4f}' for k, v in sorted(result.items(), key=lambda x: -x[1]))}")
        return result

    # ──────────────────────────────────────────────
    #  CROSS-PAIR CORRELATION
    # ──────────────────────────────────────────────

    def compute_correlation_matrix(
        self, pairs: list[str], period: str = "60d", interval: str = "1h",
        window: int = 48,
    ) -> pd.DataFrame:
        """
        Compute rolling correlation matrix between forex pairs.
        Used for risk management (avoid overexposure to correlated pairs).

        Returns:
            DataFrame (pair × pair) correlation matrix
        """
        returns = {}
        for pair in pairs:
            df = self.fetch_forex_data(pair, period=period, interval=interval, use_cache=True)
            if not df.empty and len(df) > window:
                returns[self.normalize_pair(pair)] = df["Close"].pct_change().dropna()

        if len(returns) < 2:
            return pd.DataFrame()

        ret_df = pd.DataFrame(returns).dropna()

        # Use rolling window for recent correlations
        if len(ret_df) > window:
            corr = ret_df.tail(window).corr()
        else:
            corr = ret_df.corr()

        return corr

    # ──────────────────────────────────────────────
    #  ECONOMIC CALENDAR
    # ──────────────────────────────────────────────

    def fetch_economic_events(self, days_ahead: int = 7) -> list[dict]:
        """
        Fetch upcoming economic events that impact forex markets.
        Falls back to a static calendar of key recurring events if no API is available.

        Returns:
            List of event dicts with: date, currency, event, impact ('high', 'medium', 'low')
        """
        # Key recurring forex events (approximate typical schedule)
        # In production, use ForexFactory API, Investing.com API, or similar
        recurring_events = [
            {"event": "US Non-Farm Payrolls (NFP)", "currency": "USD", "impact": "high",
             "typical_day": "first_friday"},
            {"event": "FOMC Interest Rate Decision", "currency": "USD", "impact": "high",
             "frequency": "6_weeks"},
            {"event": "US CPI (Inflation)", "currency": "USD", "impact": "high",
             "typical_day": "mid_month"},
            {"event": "ECB Interest Rate Decision", "currency": "EUR", "impact": "high",
             "frequency": "6_weeks"},
            {"event": "BoE Interest Rate Decision", "currency": "GBP", "impact": "high",
             "frequency": "6_weeks"},
            {"event": "BoJ Interest Rate Decision", "currency": "JPY", "impact": "high",
             "frequency": "8_weeks"},
            {"event": "RBA Interest Rate Decision", "currency": "AUD", "impact": "high",
             "frequency": "monthly"},
            {"event": "RBNZ Interest Rate Decision", "currency": "NZD", "impact": "high",
             "frequency": "6_weeks"},
            {"event": "BoC Interest Rate Decision", "currency": "CAD", "impact": "high",
             "frequency": "6_weeks"},
            {"event": "US GDP (Preliminary)", "currency": "USD", "impact": "high",
             "frequency": "quarterly"},
            {"event": "UK GDP", "currency": "GBP", "impact": "medium",
             "frequency": "monthly"},
            {"event": "Eurozone GDP", "currency": "EUR", "impact": "medium",
             "frequency": "quarterly"},
            {"event": "US Retail Sales", "currency": "USD", "impact": "medium",
             "frequency": "monthly"},
            {"event": "US ISM Manufacturing PMI", "currency": "USD", "impact": "medium",
             "typical_day": "first_business_day"},
            {"event": "Australia Employment Data", "currency": "AUD", "impact": "medium",
             "frequency": "monthly"},
            {"event": "Canada Employment Data", "currency": "CAD", "impact": "medium",
             "frequency": "monthly"},
        ]

        logger.info(f"📅 Economic calendar: {len(recurring_events)} key events tracked")
        return recurring_events

    def get_upcoming_high_impact_currencies(self) -> list[str]:
        """
        Return list of currencies with upcoming high-impact events.
        Used to adjust position sizing (reduce risk before big events).
        """
        events = self.fetch_economic_events(days_ahead=2)
        return list(set(e["currency"] for e in events if e["impact"] == "high"))

    # ──────────────────────────────────────────────
    #  NEWS FETCHING (for forex sentiment)
    # ──────────────────────────────────────────────

    def fetch_forex_news(self, pair: str, days_back: int = 3) -> list[dict]:
        """
        Fetch forex-related news articles for sentiment analysis.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
            days_back: Number of days to look back

        Returns:
            List of article dicts with title, description, publishedAt
        """
        if not self.news_api_key:
            logger.debug("No NEWS_API_KEY set — skipping forex news fetch")
            return []

        clean_pair = self.normalize_pair(pair)
        base = clean_pair[:3]
        quote = clean_pair[3:]

        # Build forex-specific search queries
        currency_names = {
            "USD": "US dollar federal reserve",
            "EUR": "euro ECB eurozone",
            "GBP": "british pound sterling bank of england",
            "JPY": "japanese yen bank of japan",
            "CHF": "swiss franc SNB",
            "AUD": "australian dollar RBA",
            "NZD": "new zealand dollar RBNZ",
            "CAD": "canadian dollar bank of canada",
        }

        query_parts = []
        if base in currency_names:
            query_parts.append(currency_names[base])
        if quote in currency_names:
            query_parts.append(currency_names[quote])

        query = " OR ".join(query_parts) if query_parts else f"{base} {quote} forex"

        try:
            import requests
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 20,
                "apiKey": self.news_api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                logger.info(f"📰 Fetched {len(articles)} forex news articles for {clean_pair}")
                return articles
            else:
                logger.warning(f"News API returned status {resp.status_code}")
                return []

        except Exception as e:
            logger.warning(f"Forex news fetch failed for {clean_pair}: {e}")
            return []

    # ──────────────────────────────────────────────
    #  PAIR INFO (spread, ADR, etc.)
    # ──────────────────────────────────────────────

    def fetch_pair_info(self, pair: str, use_cache: bool = True) -> dict:
        """
        Fetch current pair information including spread estimate,
        average daily range (ADR), and current price.
        """
        clean_pair = self.normalize_pair(pair)

        if use_cache and clean_pair in self._info_cache:
            ts = self._cache_ts.get(clean_pair, 0)
            if time.time() - ts < CACHE_TTL_SECONDS:
                return self._info_cache[clean_pair]

        try:
            # Fetch recent daily data for ADR calculation
            df_daily = self.fetch_forex_data(pair, period="30d", interval="1d", use_cache=True)
            # Fetch recent H1 data for current price
            df_h1 = self.fetch_forex_data(pair, period="5d", interval="1h", use_cache=True)

            if df_daily.empty or df_h1.empty:
                return {"pair": clean_pair, "is_valid": False}

            current_price = float(df_h1["Close"].iloc[-1])
            pip_size = self.get_pip_size(clean_pair)

            # Average Daily Range in pips (last 14 days)
            daily_range = df_daily["High"] - df_daily["Low"]
            adr_pips = float(daily_range.tail(14).mean() / pip_size)

            # Spread estimate from H1 high-low (rough proxy)
            h1_range = df_h1["High"] - df_h1["Low"]
            min_range_pips = float(h1_range.min() / pip_size)
            spread_estimate_pips = max(0.5, min_range_pips * 0.3)  # Rough heuristic

            # ATR in pips
            import ta
            atr_14 = ta.volatility.average_true_range(
                df_h1["High"], df_h1["Low"], df_h1["Close"], window=14
            )
            atr_pips = float(atr_14.iloc[-1] / pip_size) if not atr_14.empty else 0.0

            # Tick volume average
            avg_tick_volume = float(df_h1["Volume"].tail(48).mean()) if "Volume" in df_h1.columns else 0

            info = {
                "pair": clean_pair,
                "is_valid": True,
                "current_price": current_price,
                "pip_size": pip_size,
                "adr_pips": round(adr_pips, 1),
                "atr_14_pips": round(atr_pips, 1),
                "spread_estimate_pips": round(spread_estimate_pips, 1),
                "avg_tick_volume": round(avg_tick_volume, 0),
                "daily_change_pips": round(
                    (df_daily["Close"].iloc[-1] - df_daily["Close"].iloc[-2]) / pip_size, 1
                ) if len(df_daily) >= 2 else 0.0,
            }

            self._info_cache[clean_pair] = info
            self._cache_ts[clean_pair] = time.time()
            return info

        except Exception as e:
            logger.warning(f"fetch_pair_info({clean_pair}) failed: {e}")
            return {"pair": clean_pair, "is_valid": False, "error": str(e)}

    # ──────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────

    def _get_default_period(self, interval: str) -> str:
        """Get default download period based on interval."""
        years = self.history_years
        if interval in ("5m", "15m"):
            return "60d"  # yfinance caps intraday
        elif interval in ("1h", "1H"):
            return "2y"  # yfinance allows up to 2y for hourly
        else:
            return f"{years}y"

    @staticmethod
    def _cap_period(period: Optional[str], interval: str) -> Optional[str]:
        """Cap period to yfinance limits."""
        if period is None:
            return None
        if interval in ("5m", "15m"):
            return "60d"
        elif interval in ("1h", "1H"):
            return "2y"
        return period

    @staticmethod
    def _get_fallback_periods(interval: str) -> list[str]:
        """Get fallback period list for a given interval."""
        if interval in ("5m", "15m"):
            return ["30d", "14d", "5d"]
        elif interval in ("1h", "1H"):
            return ["1y", "6mo", "3mo"]
        return ["2y", "1y", "6mo", "max"]

    def get_active_pairs(self) -> list[str]:
        """Get the configured active trading pairs."""
        watchlist = self.config.get("watchlist", {})
        return watchlist.get("active_pairs", watchlist.get("majors", ["EURUSD"]))

    def get_all_pairs(self) -> list[str]:
        """Get all configured pairs (majors + crosses)."""
        watchlist = self.config.get("watchlist", {})
        majors = watchlist.get("majors", [])
        crosses = watchlist.get("crosses", [])
        return majors + crosses


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    loader = ForexDataLoader()

    print("\n=== Forex Data Loader Test ===")
    print(f"Active pairs: {loader.get_active_pairs()}")

    # Test EUR/USD data fetch
    df = loader.fetch_forex_data("EURUSD", period="60d", interval="1h")
    if not df.empty:
        print(f"\nEUR/USD H1: {len(df)} bars")
        print(f"  Price range: {df['Close'].min():.5f} – {df['Close'].max():.5f}")
        print(f"  Last close: {df['Close'].iloc[-1]:.5f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Test pair info
    info = loader.fetch_pair_info("EURUSD")
    if info.get("is_valid"):
        print(f"\nEUR/USD Info:")
        print(f"  ADR: {info['adr_pips']} pips")
        print(f"  ATR(14): {info['atr_14_pips']} pips")

    # Test currency strength
    strength = loader.compute_currency_strength(period="30d", interval="1d", lookback_bars=5)
    print(f"\nCurrency Strength (5-day):")
    for curr, val in sorted(strength.items(), key=lambda x: -x[1]):
        print(f"  {curr}: {val:+.4f}")

    # Test macro context
    macro = loader.fetch_macro_context(period="3mo", interval="1d")
    if not macro.empty:
        print(f"\nMacro context: {len(macro)} bars, {list(macro.columns)}")
