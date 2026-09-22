"""
data_loader.py — Historical Price Data & News Fetcher
=====================================================
Fetches OHLCV data from Alpaca (primary, 7+ years intraday),
yfinance (fallback / fundamentals / news), or IBKR (live trading).
"""

import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import requests
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 900  # 15 minutes


class DataLoader:
    """Fetches and caches historical price data and financial news."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

        # Alpaca credentials (free from https://alpaca.markets)
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_api_secret = os.getenv("ALPACA_API_SECRET", "")

        # Data source config
        data_cfg = self.config.get("data", {})
        self.default_source = data_cfg.get("price_source", "auto")
        self.alpaca_history_years = data_cfg.get("alpaca_history_years", 2)
        self.cache_expiry_hours = data_cfg.get("cache_expiry_hours", 18)
        self._info_cache = {}

    # ──────────────────────────────────────────────
    #  PRICE DATA
    # ──────────────────────────────────────────────

    def fetch_price_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for a symbol.

        Uses Alpaca (7+ years intraday) or Yahoo Finance (60-day intraday cap)
        depending on the `source` parameter and available API keys.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            period: yfinance period string (e.g., '5y', '1y', '6mo')
            interval: Bar size ('1d', '1h', '15m', '5m')
            start/end: Date strings for custom range
            use_cache: If True, load from local cache if available
            source: Data source — 'alpaca', 'yahoo', or 'auto' (default from config)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if interval is None:
            interval = self.config["strategy"].get("timeframe", "1d")
        if source is None:
            source = self.default_source

        # Route to the appropriate data source
        if source == "alpaca":
            return self._fetch_alpaca(symbol, interval=interval, use_cache=use_cache)
        elif source == "auto":
            # Try Alpaca first if API keys are configured
            if self._has_alpaca_keys():
                df = self._fetch_alpaca(symbol, interval=interval, use_cache=use_cache)
                if not df.empty:
                    return df
                logger.warning(f"Alpaca fetch failed for {symbol}, falling back to Yahoo Finance")
            # Fall through to Yahoo — cap period to Yahoo's intraday limits
            yahoo_period = self._cap_yahoo_period(period, interval)
            return self._fetch_yahoo(symbol, period=yahoo_period, interval=interval,
                                     start=start, end=end, use_cache=use_cache)
        else:  # "yahoo" or anything else
            return self._fetch_yahoo(symbol, period=period, interval=interval,
                                     start=start, end=end, use_cache=use_cache)

    def _has_alpaca_keys(self) -> bool:
        """Check if Alpaca API keys are configured."""
        return (
            bool(self.alpaca_api_key)
            and self.alpaca_api_key != "your_alpaca_api_key"
            and bool(self.alpaca_api_secret)
            and self.alpaca_api_secret != "your_alpaca_api_secret"
        )

    @staticmethod
    def _cap_yahoo_period(period: str | None, interval: str) -> str | None:
        """Cap the period to Yahoo Finance's intraday data limits.

        Yahoo enforces:
            - 15m / 5m: max 60 days
            - 1h: max 2 years (730 days)
            - 1d: unlimited
        """
        if interval in ("15m", "5m"):
            return "60d"
        elif interval in ("1h", "1H"):
            return "2y"
        return period

    # ──────────────────────────────────────────────
    #  ALPACA DATA SOURCE (7+ years intraday)
    # ──────────────────────────────────────────────

    def _fetch_alpaca(
        self,
        symbol: str,
        interval: str = "15m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Alpaca Markets (free tier: 7+ years intraday).

        Args:
            symbol: Ticker symbol
            interval: Bar size ('1d', '1h', '15m', '5m')
            use_cache: Use local parquet cache

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        history_years = self.alpaca_history_years
        cache_tag = f"{history_years}y"
        cache_file = self.data_dir / f"{symbol}_{interval}_{cache_tag}_alpaca.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_expiry_hours):
                logger.info(f"Loading cached Alpaca data for {symbol}")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading {symbol} from Alpaca | interval={interval} | history={history_years}y")

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            # Map our interval strings to Alpaca TimeFrame
            timeframe_map = {
                "1m": TimeFrame(1, TimeFrameUnit.Minute),
                "5m": TimeFrame(5, TimeFrameUnit.Minute),
                "15m": TimeFrame(15, TimeFrameUnit.Minute),
                "30m": TimeFrame(30, TimeFrameUnit.Minute),
                "1h": TimeFrame(1, TimeFrameUnit.Hour),
                "1H": TimeFrame(1, TimeFrameUnit.Hour),
                "1d": TimeFrame(1, TimeFrameUnit.Day),
                "1D": TimeFrame(1, TimeFrameUnit.Day),
            }

            tf = timeframe_map.get(interval)
            if tf is None:
                logger.error(f"Unsupported interval for Alpaca: {interval}")
                return pd.DataFrame()

            client = StockHistoricalDataClient(
                api_key=self.alpaca_api_key,
                secret_key=self.alpaca_api_secret,
            )

            # Calculate date range
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=history_years * 365)

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
            )

            bars = client.get_stock_bars(request_params)
            df = bars.df

            if df.empty:
                logger.warning(f"No Alpaca data returned for {symbol}")
                return pd.DataFrame()

            # Alpaca returns MultiIndex (symbol, timestamp) — flatten it
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")

            # Normalize column names to match Yahoo format
            col_map = {
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            }
            df.rename(columns=col_map, inplace=True)

            # Keep only OHLCV columns
            ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[ohlcv_cols].copy()
            df.index.name = "Date"
            df.dropna(inplace=True)

            # Convert timezone-aware index to timezone-naive (match Yahoo format)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)

            # Cache it
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} Alpaca bars for {symbol} ({history_years}y of {interval})")
            return df

        except ImportError:
            logger.error("alpaca-py not installed. Run: pip install alpaca-py")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Alpaca fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    #  YAHOO FINANCE DATA SOURCE (fallback)
    # ──────────────────────────────────────────────

    def _fetch_yahoo(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance (yfinance).

        Note: Intraday data (15m, 5m) is capped at ~60 days by Yahoo.

        Args:
            symbol: Ticker symbol
            period: yfinance period string
            interval: Bar size
            start/end: Date strings for custom range
            use_cache: Use local parquet cache

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if interval is None:
            interval = self.config["strategy"].get("timeframe", "1d")
        if period is None and start is None:
            years = self.config["strategy"].get("lookback_years", 5)
            period = f"{years}y"

        cache_file = self.data_dir / f"{symbol}_{interval}_{period or 'custom'}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_expiry_hours):
                logger.info(f"Loading cached data for {symbol}")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading {symbol} from Yahoo | interval={interval} | period={period}")
        try:
            ticker = yf.Ticker(symbol)
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
                # If period is empty (e.g. recent IPO or intraday limit), fallback gracefully
                if df.empty:
                    fallbacks = ["30d", "14d", "5d"] if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h") else ["2y", "1y", "6mo", "max"]
                    for fallback in fallbacks:
                        logger.info(f"Retrying {symbol} with fallback period={fallback}")
                        df = ticker.history(period=fallback, interval=interval)
                        if not df.empty:
                            break

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Clean up columns
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            df.dropna(inplace=True)

            # Cache it
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple(self, symbols: list[str], **kwargs) -> dict[str, pd.DataFrame]:
        """Fetch price data for multiple symbols."""
        data = {}
        for symbol in symbols:
            df = self.fetch_price_data(symbol, **kwargs)
            if not df.empty:
                data[symbol] = df
        return data

    # ──────────────────────────────────────────────
    #  STOCK SCREENING DATA (for penny scanner)
    # ──────────────────────────────────────────────

    def fetch_stock_info(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Fetch stock info via fast_info — fast, but only fast_info fields
        (no sector/industry/float; use fetch_full_info() for those on a
        shortlist). Includes relative volume and spread, which fast/basic
        scanners typically miss.
        """
        self._ensure_cache_attrs()

        if use_cache and symbol in self._info_cache and self._cache_valid(symbol):
            return self._info_cache[symbol]

        for attempt in range(2):
            try:
                ticker = yf.Ticker(symbol)
                fi = ticker.fast_info

                price = float(getattr(fi, "last_price", 0.0) or getattr(fi, "previous_close", 0.0) or 0.0)
                prev_close = float(getattr(fi, "previous_close", 0.0) or 0.0)
                avg_vol = float(
                    getattr(fi, "three_month_average_volume", 0)
                    or getattr(fi, "ten_day_average_volume", 0)
                    or 0
                )
                last_vol = float(getattr(fi, "last_volume", 0) or getattr(fi, "regular_market_volume", 0) or 0)
                mcap = float(getattr(fi, "market_cap", 0) or 0)
                bid = float(getattr(fi, "bid", 0) or 0)
                ask = float(getattr(fi, "ask", 0) or 0)
                day_high = float(getattr(fi, "day_high", 0) or 0)
                day_low = float(getattr(fi, "day_low", 0) or 0)
                year_high = float(getattr(fi, "year_high", 0) or 0)
                year_low = float(getattr(fi, "year_low", 0) or 0)
                shares_out = float(getattr(fi, "shares", 0) or 0)
                exchange = getattr(fi, "exchange", "") or ""

                if price <= 0:
                    # Halted, delisted, or bad ticker — don't cache garbage as if it were real
                    logger.debug(f"{symbol}: no valid price from fast_info")
                    return {"symbol": symbol, "is_valid": False, "error": "no_price"}

                # --- Relative volume: THE key penny-stock momentum signal ---
                rel_volume = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0.0

                # --- Spread %: liquidity/execution-risk check ---
                spread_pct = round(((ask - bid) / ask) * 100, 2) if (bid > 0 and ask > 0) else None

                # --- % change vs previous close ---
                change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0.0

                # --- Position within day/52w range (0 = at low, 1 = at high) ---
                day_range_pos = (
                    round((price - day_low) / (day_high - day_low), 2)
                    if day_high > day_low else None
                )
                year_range_pos = (
                    round((price - year_low) / (year_high - year_low), 2)
                    if year_high > year_low else None
                )

                info = {
                    "symbol": symbol,
                    "is_valid": True,
                    "price": price,
                    "prev_close": prev_close,
                    "change_pct": change_pct,
                    "avg_volume": avg_vol,
                    "last_volume": last_vol,
                    "rel_volume": rel_volume,          # >1 = trading above average (trending)
                    "market_cap": mcap,
                    "shares_outstanding": shares_out,
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": spread_pct,           # None if no quote; watch for >3-5% = illiquid
                    "day_high": day_high,
                    "day_low": day_low,
                    "day_range_pos": day_range_pos,
                    "year_high": year_high,
                    "year_low": year_low,
                    "year_range_pos": year_range_pos,
                    "exchange": exchange,
                    # Placeholder tier — real values only via fetch_full_info()
                    "sector": None,
                    "industry": None,
                    "name": symbol,
                    "info_tier": "fast",
                }
                self._cache_set(symbol, info)
                return info

            except Exception as e:
                logger.debug(f"fetch_stock_info({symbol}) attempt {attempt + 1}/2 failed: {e}")
                time.sleep(0.5)

        return {"symbol": symbol, "is_valid": False, "error": "fetch_failed"}

    def fetch_stock_info_batch(
        self, symbols: list[str], max_workers: int = 8, use_cache: bool = True
    ) -> dict[str, dict]:
        """
        Parallel version of fetch_stock_info for scanning a full universe.
        Bounded thread pool avoids hammering Yahoo hard enough to trigger 429s,
        while still being far faster than the sequential single-symbol loop
        you'd get calling fetch_stock_info() in a plain for-loop.
        """
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.fetch_stock_info, sym, use_cache): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    results[sym] = future.result()
                except Exception as e:
                    logger.warning(f"Batch fetch failed for {sym}: {e}")
                    results[sym] = {"symbol": sym, "is_valid": False, "error": str(e)}

        valid_count = sum(1 for r in results.values() if r.get("is_valid"))
        logger.info(f"Batch fetched {valid_count}/{len(symbols)} valid quotes")
        return results

    def fetch_full_info(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Slow, accurate tier — real sector/industry/float/PE/beta from yfinance's
        full .info endpoint. Call this ONLY on your final shortlist after
        fast_info-based filtering, not on the whole universe — .info is
        meaningfully slower and heavier per-symbol than fast_info.
        """
        self._ensure_cache_attrs()
        cache_key = f"{symbol}:full"

        if use_cache and cache_key in self._info_cache and self._cache_valid(cache_key):
            return self._info_cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.info or {}

            info = {
                "symbol": symbol,
                "is_valid": bool(raw.get("regularMarketPrice") or raw.get("currentPrice")),
                "name": raw.get("shortName") or raw.get("longName") or symbol,
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                "float_shares": raw.get("floatShares"),
                "shares_outstanding": raw.get("sharesOutstanding"),
                "short_ratio": raw.get("shortRatio"),
                "short_pct_float": raw.get("shortPercentOfFloat"),
                "pe_ratio": raw.get("trailingPE"),
                "forward_pe": raw.get("forwardPE"),
                "beta": raw.get("beta"),
                "insider_pct": raw.get("heldPercentInsiders"),
                "institution_pct": raw.get("heldPercentInstitutions"),
                "target_mean_price": raw.get("targetMeanPrice"),
                "recommendation": raw.get("recommendationKey"),
                "info_tier": "full",
            }
            self._cache_set(cache_key, info)
            return info
        except Exception as e:
            logger.warning(f"fetch_full_info({symbol}) failed: {e}")
            return {"symbol": symbol, "is_valid": False, "error": str(e), "info_tier": "full"}

    def fetch_macro_context(
        self, period: str = "2y", interval: str = "1d", use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch broader market regime context (SPY, QQQ, IWM, VIX).
        Provides macro trend, growth appetite, small-cap relative strength (IWM),
        and market-wide volatility/risk sentiment.
        """
        cache_key = f"macro:{period}:{interval}"
        if use_cache and hasattr(self, "_macro_cache") and cache_key in self._macro_cache:
            ts = getattr(self, "_macro_cache_ts", {}).get(cache_key, 0)
            if time.time() - ts < 3600:  # 1 hour cache for macro
                return self._macro_cache[cache_key].copy()

        symbols = ["SPY", "QQQ", "IWM", "^VIX"]
        dfs = {}
        for sym in symbols:
            try:
                df = self.fetch_price_data(sym, period=period, interval=interval, use_cache=use_cache)
                if not df.empty and "Close" in df.columns:
                    s = df["Close"].copy()
                    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                        s.index = s.index.tz_convert(None)
                    dfs[sym] = s
            except Exception as e:
                logger.warning(f"Failed to fetch macro data for {sym}: {e}")

        if not dfs:
            return pd.DataFrame()

        macro_df = pd.DataFrame(dfs)
        if isinstance(macro_df.index, pd.DatetimeIndex) and macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_convert(None)

        # Forward/backward fill individual symbol series first because NYSE (SPY/QQQ/IWM at :30)
        # and CBOE (^VIX at :00) trade on staggered hourly timestamps. Without ffill, pct_change
        # evaluates across interleaved NaNs and returns 100% NaN for returns!
        macro_df = macro_df.ffill().bfill()

        result = pd.DataFrame(index=macro_df.index)
        if "SPY" in macro_df.columns:
            result["spy_close"] = macro_df["SPY"]
            result["spy_return_1d"] = macro_df["SPY"].pct_change(1).fillna(0.0)
            result["spy_return_5d"] = macro_df["SPY"].pct_change(5).fillna(0.0)
        if "QQQ" in macro_df.columns:
            result["qqq_close"] = macro_df["QQQ"]
            result["qqq_return_1d"] = macro_df["QQQ"].pct_change(1).fillna(0.0)
            result["qqq_return_5d"] = macro_df["QQQ"].pct_change(5).fillna(0.0)
        if "IWM" in macro_df.columns:
            result["iwm_close"] = macro_df["IWM"]
            result["iwm_return_1d"] = macro_df["IWM"].pct_change(1).fillna(0.0)
            result["iwm_return_5d"] = macro_df["IWM"].pct_change(5).fillna(0.0)
        if "^VIX" in macro_df.columns:
            result["vix_level"] = macro_df["^VIX"]
            result["vix_change_5d"] = macro_df["^VIX"].pct_change(5).fillna(0.0)

        result.ffill(inplace=True)
        result.bfill(inplace=True)

        if not hasattr(self, "_macro_cache"):
            self._macro_cache = {}
            self._macro_cache_ts = {}
        self._macro_cache[cache_key] = result
        self._macro_cache_ts[cache_key] = time.time()

        return result.copy()

    def fetch_earnings_dates(self, symbol: str) -> dict:
        """Fetch days until next earnings date and days since last earnings date."""
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            now = datetime.now()
            next_date = None
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date") or cal.get("Earnings High")
                if dates and isinstance(dates, list) and len(dates) > 0:
                    next_date = pd.to_datetime(dates[0]).to_pydatetime().replace(tzinfo=None)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    next_date = pd.to_datetime(val).to_pydatetime().replace(tzinfo=None)

            if next_date:
                days_until = max(0, (next_date - now).days)
                return {"days_until_earnings": days_until, "next_earnings_date": str(next_date)}
        except Exception as e:
            logger.debug(f"Earnings lookup failed for {symbol}: {e}")

        return {"days_until_earnings": 999, "next_earnings_date": None}

    CACHE_TTL_SECONDS = 900  # 15 minutes

    def _ensure_cache_attrs(self):
        if not hasattr(self, "_info_cache"):
            self._info_cache = {}
        if not hasattr(self, "_cache_ts"):
            self._cache_ts = {}

    def _cache_set(self, key: str, data: dict) -> None:
        self._ensure_cache_attrs()
        self._info_cache[key] = data
        self._cache_ts[key] = time.time()

    def _cache_valid(self, key: str) -> bool:
        self._ensure_cache_attrs()
        ts = self._cache_ts.get(key)
        return ts is not None and (time.time() - ts) < CACHE_TTL_SECONDS

    def _get_yahoo_session(self):
        """
        Fixed version: your original called .text on the crumb response but
        never checked the status code, and never set a real browser UA on the
        crumb request. If Yahoo returns a 401/HTML error page here, `crumb`
        silently becomes garbage and every screener call after it 401s.
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        })
        try:
            session.get("https://fc.yahoo.com", timeout=5)  # sets cookies
            crumb_resp = session.get(
                "https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=5
            )
            crumb = crumb_resp.text.strip()
            if crumb_resp.status_code != 200 or not crumb or "<html" in crumb.lower():
                logger.warning(f"Yahoo crumb request failed (status {crumb_resp.status_code})")
                return session, None
            return session, crumb
        except Exception as e:
            logger.warning(f"Yahoo session/crumb setup failed: {e}")
            return session, None

    def _fetch_yahoo_screener(self, session, crumb, scr_id, min_price, max_price, min_volume, count=100):
        """Fetch one Yahoo predefined screener with retry/backoff. Returns raw quote dicts."""
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "formatted": "false",
            "lang": "en-US",
            "region": "US",
            "scrIds": scr_id,
            "count": count,
        }
        if crumb:
            params["crumb"] = crumb

        for attempt in range(3):
            try:
                resp = session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("finance", {}).get("result") or [{}]
                    return result[0].get("quotes", [])
                if resp.status_code == 401:
                    logger.warning(f"{scr_id}: 401 Unauthorized (crumb invalid/expired) — not retrying")
                    return []
                logger.warning(f"{scr_id}: HTTP {resp.status_code} (attempt {attempt + 1}/3)")
            except Exception as e:
                logger.warning(f"{scr_id}: request error {e} (attempt {attempt + 1}/3)")
            time.sleep(2 ** attempt)
        return []

    def _fetch_nasdaq_universe(self) -> list[str]:
        """Pull the full current NASDAQ + NYSE/AMEX ticker list (free, no auth)."""
        symbols = set()
        headers = {"User-Agent": "Mozilla/5.0"}
        urls = [
            "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        ]
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                lines = resp.text.splitlines()
                header = lines[0].split("|")
                sym_idx = header.index("Symbol") if "Symbol" in header else 0
                for line in lines[1:-1]:  # last line is a file-generation footer
                    parts = line.split("|")
                    if len(parts) <= sym_idx:
                        continue
                    sym = parts[sym_idx].strip()
                    if sym and sym.isascii() and "$" not in sym and "." not in sym and len(sym) <= 5:
                        symbols.add(sym)
            except Exception as e:
                logger.warning(f"Could not fetch {url}: {e}")
        logger.info(f"Exchange sweep: {len(symbols)} raw symbols from NASDAQ/NYSE/AMEX")
        return sorted(symbols)

    def _batch_filter_by_price_volume(self, symbols, min_price, max_price, min_volume, batch_size=150) -> list[str]:
        """Bulk-fetch last price/volume for a large symbol list and keep only penny-range hits."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — skipping exchange-sweep price filter")
            return []

        matched = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                data = yf.download(
                    tickers=" ".join(batch),
                    period="1d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as e:
                logger.warning(f"Batch {i // batch_size} quote fetch failed: {e}")
                continue

            for sym in batch:
                try:
                    row = data[sym] if len(batch) > 1 else data
                    if row.empty:
                        continue
                    price = float(row["Close"].iloc[-1])
                    vol = float(row["Volume"].iloc[-1])
                    if min_price <= price <= max_price and vol >= min_volume:
                        matched.append(sym)
                        if not self._cache_valid(sym):
                            self._cache_set(sym, {
                                "symbol": sym,
                                "name": sym,
                                "price": price,
                                "avg_volume": vol,
                                "market_cap": 0,
                                "pe_ratio": None,
                                "beta": 1.5,
                                "sector": "Unknown",
                                "industry": "Unknown",
                                "52w_high": price,
                                "52w_low": price,
                                "change_pct": 0,
                            })
                except Exception:
                    continue
        logger.info(f"Exchange sweep price/volume filter matched {len(matched)} symbols")
        return matched

    def get_penny_stock_universe(
        self,
        max_price: float = 10.0,
        min_price: float = 0.50,
        min_volume: int = 500_000,
    ) -> list[str]:
        """
        Get a broad list of penny stock tickers to screen, including live trending
        and active stocks. Queries live Yahoo Finance screeners (properly
        authenticated with crumb/session), falls back to a full NASDAQ/NYSE/AMEX
        exchange sweep if the live screeners under-deliver, and only falls back
        to a small static safety net if BOTH live sources fail.
        """
        self._ensure_cache_attrs()
        logger.info("Discovering live trending and active penny stocks...")
        screener_symbols = set()

        # Method 1: Live Yahoo Finance Predefined Screeners (properly authenticated)
        session, crumb = self._get_yahoo_session()
        if crumb is None:
            logger.warning("Proceeding without a Yahoo crumb — screener calls will likely 401")

        scr_ids = [
            "day_gainers",
            "most_actives",
            "small_cap_gainers",
            "undervalued_growth_stocks",
            "growth_technology_stocks",
            "aggressive_small_caps",
        ]
        for scr_id in scr_ids:
            quotes = self._fetch_yahoo_screener(session, crumb, scr_id, min_price, max_price, min_volume)
            for q in quotes:
                sym = q.get("symbol", "")
                price = q.get("regularMarketPrice", 0) or 0
                vol = q.get("regularMarketVolume", 0) or q.get("averageDailyVolume3Month", 0) or 0
                # Filter for penny/small-cap range and liquid volume
                if sym and min_price <= price <= max_price and vol >= min_volume:
                    screener_symbols.add(sym)
                    # Cache basic info directly to save network calls later
                    self._cache_set(sym, {
                        "symbol": sym,
                        "name": q.get("shortName", sym),
                        "price": price,
                        "avg_volume": vol,
                        "market_cap": q.get("marketCap", 0) or 0,
                        "pe_ratio": q.get("trailingPE"),
                        "beta": q.get("beta", 1.5),
                        "sector": "Trending Small-Cap",
                        "industry": "Trending",
                        "52w_high": q.get("fiftyTwoWeekHigh", price),
                        "52w_low": q.get("fiftyTwoWeekLow", price),
                        "change_pct": q.get("regularMarketChangePercent", 0),
                    })
        logger.info(f"Discovered {len(screener_symbols)} live trending/active penny stocks from Yahoo Finance")

        # Method 2: Broad NASDAQ/NYSE/AMEX exchange sweep if screeners under-delivered (< 8)
        if len(screener_symbols) < 8:
            logger.info("Live screener yield low — sweeping full NASDAQ/NYSE/AMEX listings...")
            full_universe = self._fetch_nasdaq_universe()
            if full_universe:
                swept = self._batch_filter_by_price_volume(full_universe, min_price, max_price, min_volume)
                screener_symbols.update(swept)
                logger.info(f"Exchange sweep added {len(swept)} more candidates")

        # Method 3: Static safety net — last resort only, clearly logged (not silent)
        if len(screener_symbols) < 5:
            logger.warning(
                "Both live screeners and exchange sweep failed/near-empty — "
                "using static safety net (results may be stale)"
            )
            expanded_pennies = [
                "PDSB", "SNDL", "CLNE", "GEVO", "MVIS", "BLNK", "DNA", "TELL", "GSAT",
                "BTBT", "BNGO", "WKHS", "BARK", "PSFE", "FCEL", "ZOM", "CTRM",
                "SENS", "AEVA", "OUST", "CAN", "HUT", "BITF", "WULF",
            ]
            screener_symbols.update(expanded_pennies)

        # Always include user's watchlist
        watchlist = self.config.get("watchlist", {}).get("symbols", [])
        screener_symbols.update([s.upper() for s in watchlist])

        # Always include PDSB
        screener_symbols.add("PDSB")

        logger.info(f"Total penny universe for scanning: {len(screener_symbols)} symbols")
        return sorted(list(screener_symbols))

    # ──────────────────────────────────────────────
    #  NEWS DATA
    # ──────────────────────────────────────────────


    def _parse_any_date(self, value) -> Optional[datetime]:
        """Best-effort parse of a date coming from any of the three sources."""
        if not value:
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except Exception:
                return None
        if isinstance(value, str):
            # NewsAPI: ISO 8601, e.g. 2026-09-10T14:23:00Z
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
            # RSS pubDate: RFC 2822, e.g. "Wed, 10 Sep 2026 14:23:00 GMT"
            try:
                return parsedate_to_datetime(value).replace(tzinfo=None)
            except Exception:
                pass
        return None

    def _within_lookback(self, published_at, days_back: int) -> bool:
        """Return True if the article's date is within the lookback window, or
        if the date couldn't be parsed at all (fail open rather than silently
        dropping everything when a source's date format is unexpected)."""
        dt = self._parse_any_date(published_at)
        if dt is None:
            return True
        cutoff = datetime.now() - timedelta(days=days_back)
        return dt >= cutoff

    def _dedup_articles(self, articles: list[dict]) -> list[dict]:
        """Collapse near-duplicate wire-service stories syndicated across outlets."""
        seen = set()
        deduped = []
        for a in articles:
            title = (a.get("title") or "").lower().strip()
            # Normalize: strip punctuation/whitespace so minor title variants collapse
            key = re.sub(r"[^a-z0-9 ]", "", title)
            key = re.sub(r"\s+", " ", key)[:80]  # first 80 chars is enough to catch dupes
            if key and key not in seen:
                seen.add(key)
                deduped.append(a)
        return deduped

    def fetch_news(
        self,
        query: str,
        days_back: Optional[int] = None,
        max_articles: int = 50,
    ) -> list[dict]:

        """
        Fetch financial news articles, merging every source available:
        NewsAPI (if key set) + Google News RSS (always, free) + Yahoo scrape
        (fallback if the others come up short). All sources are filtered to
        the same days_back window and deduplicated before returning.

        Returns:
            List of dicts with keys: title, description, source, url, published_at
        """
        if days_back is None:
            days_back = self.config["sentiment"].get("lookback_days", 7)

        all_articles: list[dict] = []

        # --- Source 1: NewsAPI (best relevance/coverage, but rate-limited) ---
        if self.news_api_key and self.news_api_key != "your_newsapi_key_here":
            all_articles.extend(self._fetch_newsapi(query, days_back, max_articles))
        else:
            logger.info("No NewsAPI key set — skipping NewsAPI source")

        # --- Source 2: Google News RSS (free, keyless, no quota) ---
        try:
            all_articles.extend(self._fetch_google_news_rss(query, days_back, max_articles))
        except Exception as e:
            logger.warning(f"Google News RSS failed for '{query}': {e}")

        # --- Source 3: Yahoo scrape (fallback if the above returned little) ---
        if len(all_articles) < 5:
            all_articles.extend(self._scrape_news_fallback(query, days_back))

        # Enforce the lookback window uniformly (NewsAPI already filters via
        # `from_date`, but RSS/Yahoo need it applied here) and dedup.
        filtered = [a for a in all_articles if self._within_lookback(a.get("published_at"), days_back)]
        deduped = self._dedup_articles(filtered)

        logger.info(f"fetch_news('{query}'): {len(deduped)} unique articles after merge/dedup")
        return deduped[:max_articles]


    def _fetch_newsapi(self, query: str, days_back: int, max_articles: int) -> list[dict]:
        """NewsAPI /v2/everything source."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            import requests as req

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f'"{query}" AND (stock OR shares OR trading OR earnings)',
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": min(max_articles, 100),
                "language": "en",
                "apiKey": self.news_api_key,
            }
            resp = req.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("NewsAPI rate limit hit (429) — daily quota likely exhausted")
                return []
            resp.raise_for_status()
            data = resp.json()

            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                })
            logger.info(f"NewsAPI: {len(articles)} articles for '{query}'")
            return articles
        except Exception as e:
            logger.error(f"NewsAPI error for '{query}': {e}")
            return []


    def _fetch_google_news_rss(self, query: str, days_back: int, max_articles: int) -> list[dict]:
        """
        Free, keyless news source via Google News RSS. No API key, no daily
        quota — good coverage boost for thinly-covered small/penny-cap tickers
        that NewsAPI's free tier often misses entirely.
        """
        import requests as req

        search_term = f"{query} stock"
        url = f"https://news.google.com/rss/search?q={quote_plus(search_term)}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        resp = req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        articles = []
        for item in root.findall(".//item")[: max_articles]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else "Google News"
            description = (item.findtext("description") or title).strip()

            if title:
                articles.append({
                    "title": title,
                    "description": description,
                    "source": source,
                    "url": link,
                    "published_at": pub_date,
                })

        logger.info(f"Google News RSS: {len(articles)} articles for '{query}'")
        return articles


    def _scrape_news_fallback(
        self, query: str, days_back: Optional[int] = None
    ) -> list[dict]:
        """
        Fallback: scrape Yahoo Finance news for a ticker.
        Fixed: days_back is now actually applied — previously accepted but ignored.
        """
        if days_back is None:
            days_back = self.config["sentiment"].get("lookback_days", 7)

        try:
            ticker = yf.Ticker(query)
            news = ticker.news or []

            articles = []
            for item in news[:20]:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") if isinstance(item.get("content"), dict) else item
                title = item.get("title") or content.get("title", "")
                description = item.get("description") or content.get("summary") or content.get("description") or title
                publisher = item.get("publisher") or (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict) else ""
                )
                url = item.get("link") or (
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict) else ""
                )

                pub_time = item.get("providerPublishTime") or content.get("pubDate") or content.get("providerPublishTime")
                published_at = ""
                if pub_time:
                    if isinstance(pub_time, (int, float)):
                        try:
                            published_at = datetime.fromtimestamp(pub_time).isoformat()
                        except Exception:
                            published_at = ""
                    elif isinstance(pub_time, str):
                        published_at = pub_time

                # Fix: actually apply the days_back filter (previously ignored)
                if not self._within_lookback(published_at, days_back):
                    continue

                articles.append({
                    "title": title,
                    "description": description,
                    "source": publisher,
                    "url": url,
                    "published_at": published_at,
                })

            logger.info(f"Scraped {len(articles)} articles for '{query}' within {days_back}d (fallback)")
            return articles

        except Exception as e:
            logger.error(f"News scrape fallback failed for '{query}': {e}")
            return []


    def fetch_news_for_symbols(self, symbols: list[str]) -> dict[str, list[dict]]:
        """Fetch news for multiple symbols (sequential — see note below for parallelizing)."""
        all_news = {}
        for symbol in symbols:
            articles = self.fetch_news(symbol)
            if articles:
                all_news[symbol] = articles
        return all_news


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()

    # Test price data
    df = loader.fetch_price_data("AAPL", period="1y")
    print(f"\nAAPL — {len(df)} bars loaded")
    print(df.tail())

    # Test news
    news = loader.fetch_news("AAPL", days_back=3)
    print(f"\nAAPL News — {len(news)} articles")
    for n in news[:3]:
        print(f"  • {n['title'][:80]}")
