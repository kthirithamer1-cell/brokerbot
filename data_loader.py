"""
data_loader.py — Historical Price Data & News Fetcher
=====================================================
Fetches OHLCV data from Alpaca (primary, 7+ years intraday),
yfinance (fallback / fundamentals / news), or IBKR (live trading).
"""

import os
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

    def fetch_stock_info(self, symbol: str) -> dict:
        """Fetch basic stock info (price, volume, market cap, exchange) ultra-fast via fast_info."""
        if symbol in self._info_cache:
            return self._info_cache[symbol]
        try:
            ticker = yf.Ticker(symbol)
            fi = ticker.fast_info
            price = float(getattr(fi, "last_price", 0.0) or getattr(fi, "previous_close", 0.0) or 0.0)
            avg_vol = float(getattr(fi, "three_month_average_volume", 0) or getattr(fi, "ten_day_average_volume", 0) or 0)
            mcap = float(getattr(fi, "market_cap", 0) or 0)
            info = {
                "symbol": symbol,
                "price": price,
                "avg_volume": avg_vol,
                "market_cap": mcap,
                "exchange": getattr(fi, "exchange", ""),
                "sector": "Trending Small-Cap",
                "industry": "Momentum",
                "name": symbol,
            }
            self._info_cache[symbol] = info
            return info
        except Exception as e:
            logger.debug(f"Failed to fetch fast_info for {symbol}: {e}")
            return {}

    def get_penny_stock_universe(self, max_price: float = 10.0, min_price: float = 0.50, min_volume: int = 500_000) -> list[str]:
        """
        Get a broad list of penny stock tickers to screen, including live trending and active stocks.
        Queries live Yahoo Finance screeners (day_gainers, most_actives) and merges with curated universe.
        """
        logger.info("Discovering live trending and active penny stocks...")
        screener_symbols = set()

        # Method 1: Live Yahoo Finance Predefined Screeners (Day Gainers & Most Active)
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            for scr_id in ["day_gainers", "most_actives"]:
                url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds={scr_id}&count=100"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    quotes = resp.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
                    for q in quotes:
                        sym = q.get("symbol", "")
                        price = q.get("regularMarketPrice", 0) or 0
                        vol = q.get("regularMarketVolume", 0) or q.get("averageDailyVolume3Month", 0) or 0
                        # Filter for penny/small-cap range and liquid volume
                        if sym and min_price <= price <= max_price and vol >= min_volume:
                            screener_symbols.add(sym)
                            # Cache basic info directly to save network calls later
                            self._info_cache[sym] = {
                                "symbol": sym,
                                "name": q.get("shortName", sym),
                                "price": price,
                                "avg_volume": vol,
                                "market_cap": q.get("marketCap", 0) or 0,
                                "pe_ratio": q.get("trailingPE"),
                                "beta": 1.5,
                                "sector": "Trending Small-Cap",
                                "industry": "Trending",
                                "52w_high": q.get("fiftyTwoWeekHigh", price),
                                "52w_low": q.get("fiftyTwoWeekLow", price),
                                "change_pct": q.get("regularMarketChangePercent", 0),
                            }
            logger.info(f"Discovered {len(screener_symbols)} live trending/active penny stocks from Yahoo Finance")
        except Exception as e:
            logger.warning(f"Could not fetch live trending screeners: {e}")

        # If live screener returned few or no symbols, fallback to expanded list
        if len(screener_symbols) < 10:
            expanded_pennies = [
                "PDSB", "SNDL", "CLNE", "GEVO", "MVIS", "BLNK", "DNA", "TELL", "GSAT",
                "BTBT", "BNGO", "WKHS", "BARK", "PSFE", "FCEL", "ZOM", "CTRM",
                "SENS", "AEVA", "OUST", "CAN", "HUT", "BITF", "WULF",
                "PLUG", "SOFI", "NIO", "LCID", "MARA", "RIOT", "OPEN", "CLOV",
                "BB", "NOK", "QS", "LAZR", "ACHR", "JOBY", "RIVN", "GRAB"
            ]
            screener_symbols.update(expanded_pennies)

        # Always include user's watchlist
        watchlist = self.config.get("watchlist", {}).get("symbols", [])
        screener_symbols.update([s.upper() for s in watchlist])

        # Also always include PDSB
        screener_symbols.add("PDSB")

        logger.info(f"Total penny universe for scanning: {len(screener_symbols)} symbols")
        return sorted(list(screener_symbols))

    # ──────────────────────────────────────────────
    #  NEWS DATA
    # ──────────────────────────────────────────────

    def fetch_news(
        self,
        query: str,
        days_back: Optional[int] = None,
        max_articles: int = 50,
    ) -> list[dict]:
        """
        Fetch financial news articles via NewsAPI.

        Args:
            query: Search query (ticker symbol or company name)
            days_back: How many days of news to fetch
            max_articles: Maximum number of articles

        Returns:
            List of dicts with keys: title, description, source, url, published_at
        """
        if not self.news_api_key or self.news_api_key == "your_newsapi_key_here":
            logger.warning("No NewsAPI key set — falling back to scraper")
            return self._scrape_news_fallback(query, days_back)

        if days_back is None:
            days_back = self.config["sentiment"].get("lookback_days", 7)

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

            response = req.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                })

            logger.info(f"Fetched {len(articles)} articles for '{query}'")
            return articles

        except Exception as e:
            logger.error(f"NewsAPI error for '{query}': {e}")
            return self._scrape_news_fallback(query, days_back)

    def _scrape_news_fallback(
        self, query: str, days_back: Optional[int] = None
    ) -> list[dict]:
        """
        Fallback: scrape Yahoo Finance news for a ticker.
        Used when NewsAPI key is not available.
        """
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
                publisher = item.get("publisher") or (content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else "")
                url = item.get("link") or (content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else "")
                
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

                articles.append({
                    "title": title,
                    "description": description,
                    "source": publisher,
                    "url": url,
                    "published_at": published_at,
                })

            logger.info(f"Scraped {len(articles)} articles for '{query}' (fallback)")
            return articles

        except Exception as e:
            logger.error(f"News scrape fallback failed for '{query}': {e}")
            return []

    def fetch_news_for_symbols(self, symbols: list[str]) -> dict[str, list[dict]]:
        """Fetch news for multiple symbols."""
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
