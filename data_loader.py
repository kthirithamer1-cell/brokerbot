"""
data_loader.py — Historical Price Data & News Fetcher
=====================================================
Fetches OHLCV data from yfinance (or IBKR) and financial news
headlines from NewsAPI for sentiment analysis.
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
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for a symbol via yfinance.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            period: yfinance period string (e.g., '5y', '1y', '6mo')
            interval: Bar size ('1d', '1h', '15m', '5m')
            start/end: Date strings for custom range
            use_cache: If True, load from local cache if available

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if interval is None:
            interval = self.config["strategy"].get("timeframe", "1d")
        if period is None and start is None:
            years = self.config["strategy"].get("lookback_years", 5)
            period = f"{years}y"

        cache_file = self.data_dir / f"{symbol}_{interval}_{period or 'custom'}.parquet"

        # Check cache (< 1 day old)
        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=18):
                logger.info(f"Loading cached data for {symbol}")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading {symbol} | interval={interval} | period={period}")
        try:
            ticker = yf.Ticker(symbol)
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

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
        """Fetch basic stock info (price, volume, market cap, exchange)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "avg_volume": info.get("averageDailyVolume10Day", 0),
                "market_cap": info.get("marketCap", 0),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "name": info.get("shortName", symbol),
            }
        except Exception as e:
            logger.error(f"Failed to fetch info for {symbol}: {e}")
            return {}

    def get_penny_stock_universe(self) -> list[str]:
        """
        Get a broad list of penny stock tickers to screen.
        Uses yfinance screener for stocks under $5 with decent volume.
        """
        logger.info("Building penny stock universe...")
        # We'll use a curated approach: fetch from known small-cap lists
        # and filter by price/volume criteria
        try:
            # Fetch a broad set of small-cap tickers from known ETF holdings
            # and popular penny stock lists
            screener_symbols = set()

            # Method 1: Screen popular small-cap / micro-cap tickers
            small_cap_etfs = ["IWC", "SCHA", "VB"]  # Micro/small cap ETFs
            for etf_symbol in small_cap_etfs:
                try:
                    etf = yf.Ticker(etf_symbol)
                    holdings = etf.info.get("holdings", [])
                    if holdings:
                        for h in holdings:
                            if "symbol" in h:
                                screener_symbols.add(h["symbol"])
                except Exception:
                    pass

            # Method 2: Use yfinance screener query
            # Fetch a list of active US stocks and filter
            try:
                from yfinance import Screener
                sc = Screener()
                sc.set_default_body({
                    "query": {
                        "operator": "AND",
                        "operands": [
                            {"operator": "LT", "operands": ["regularmarketprice", 5.0]},
                            {"operator": "GT", "operands": ["regularmarketprice", 0.5]},
                            {"operator": "GT", "operands": ["avgdailyvol10day", 500000]},
                        ],
                    },
                    "size": 250,
                    "offset": 0,
                    "sortField": "avgdailyvol10day",
                    "sortType": "DESC",
                })
                result = sc.response
                if "quotes" in result:
                    for q in result["quotes"]:
                        screener_symbols.add(q["symbol"])
            except Exception as e:
                logger.warning(f"yfinance Screener not available: {e}")

            # Method 3: Fallback — curated active sub-$10 tickers
            fallback_pennies = [
                "SOFI", "PLUG", "NIO", "LCID", "MARA", "RIOT", "OPEN", "CLOV",
                "DNA", "TELL", "GSAT", "SNDL", "BTBT", "BNGO", "WKHS", "BARK",
                "PSFE", "BB", "NOK", "FCEL", "ZOM", "CTRM", "SENS", "MVIS",
                "CLNE", "GEVO", "BLNK", "QS", "LAZR", "AEVA", "OUST", "ACHR",
                "JOBY", "RIVN", "GRAB", "WBD", "PENN", "RUN", "PLUG", "SEDG"
            ]
            screener_symbols.update(fallback_pennies)

            logger.info(f"Penny universe: {len(screener_symbols)} candidates")
            return list(screener_symbols)

        except Exception as e:
            logger.error(f"Failed to build penny universe: {e}")
            return []

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
