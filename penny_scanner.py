"""
penny_scanner.py — Penny Stock Scanner & Daily Picks
=====================================================
Scans the market for penny stocks ($0.50–$5) with the highest
short-term upside potential using multi-factor scoring and
ML confidence overlay.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from data_loader import DataLoader
from feature_engine import FeatureEngine
from sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class PennyScanner:
    """
    Scans and ranks penny stocks by short-term upside potential.

    Scoring factors:
        1. Volume Surge (25%)    — Today's vol vs 20-day avg
        2. Price Momentum (20%)  — 5-day and 10-day returns
        3. Technical Setup (20%) — RSI, MACD, EMA signals
        4. News Sentiment (20%)  — FinBERT scored headlines
        5. Relative Strength (15%) — vs SPY benchmark
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.scanner_cfg = self.config.get("penny_scanner", {})
        self.weights = self.scanner_cfg.get("weights", {
            "volume_surge": 0.25,
            "momentum": 0.20,
            "technical": 0.20,
            "sentiment": 0.20,
            "relative_strength": 0.15,
        })
        self.top_n = self.scanner_cfg.get("top_n_picks", 10)

        self.data_loader = DataLoader(config_path)
        self.feature_engine = FeatureEngine()
        self.sentiment_analyzer = None  # Lazy load

        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def scan(self, ml_model=None) -> pd.DataFrame:
        """
        Run the full penny stock scan pipeline.

        Args:
            ml_model: Optional trained MLModel instance for confidence overlay

        Returns:
            DataFrame with ranked penny stock picks
        """
        logger.info("=" * 60)
        logger.info("🔍 PENNY STOCK SCANNER — Starting Scan")
        logger.info("=" * 60)

        # Step 1: Get candidate universe
        candidates = self._get_candidates()
        if candidates.empty:
            logger.warning("No penny stock candidates found")
            return pd.DataFrame()

        logger.info(f"Found {len(candidates)} candidates after filtering")

        # Step 2: Score each candidate
        scored = self._score_candidates(candidates, ml_model)

        # Step 3: Rank and select top picks
        scored = scored.sort_values("total_score", ascending=False)
        top_picks = scored.head(self.top_n).reset_index(drop=True)
        top_picks["rank"] = range(1, len(top_picks) + 1)

        # Step 4: Save report
        self._save_report(top_picks)
        self._print_report(top_picks)

        return top_picks

    def _get_candidates(self) -> pd.DataFrame:
        """Filter the universe for valid penny stock candidates."""
        min_price = self.scanner_cfg.get("min_price", 0.50)
        max_price = self.scanner_cfg.get("max_price", 5.00)
        min_volume = self.scanner_cfg.get("min_avg_volume", 500_000)
        min_mcap = self.scanner_cfg.get("min_market_cap", 10_000_000)
        max_mcap = self.scanner_cfg.get("max_market_cap", 500_000_000)

        # Get broad universe of tickers (including live trending & active gainers)
        symbols = self.data_loader.get_penny_stock_universe(
            max_price=max_price,
            min_price=min_price,
            min_volume=min_volume
        )

        candidates = []
        for symbol in symbols:
            try:
                info = self.data_loader.fetch_stock_info(symbol)
                if not info:
                    continue

                price = info.get("price", 0)
                avg_vol = info.get("avg_volume", 0)
                mcap = info.get("market_cap", 0)

                # Apply filters
                if not (min_price <= price <= max_price):
                    continue
                if avg_vol < min_volume:
                    continue
                if mcap > 0 and not (min_mcap <= mcap <= max_mcap):
                    continue

                candidates.append(info)
                logger.debug(f"  ✓ {symbol}: ${price:.2f}, vol={avg_vol:,.0f}")

            except Exception as e:
                logger.debug(f"  ✗ {symbol}: {e}")
                continue

        if not candidates:
            return pd.DataFrame()

        # Prioritize top active gainers and cap at 12 to run lean on 4GB RAM
        candidates = sorted(candidates, key=lambda x: (x.get("change_pct", 0) or 0, x.get("avg_volume", 0) or 0), reverse=True)[:12]

        return pd.DataFrame(candidates)

    def _score_candidates(
        self, candidates: pd.DataFrame, ml_model=None
    ) -> pd.DataFrame:
        """Score each candidate on all factors."""
        scores = []

        # Get SPY benchmark data for relative strength
        spy_data = self.data_loader.fetch_price_data("SPY", period="3mo", interval="1d")
        spy_return_5d = 0.0
        if not spy_data.empty and len(spy_data) >= 5:
            spy_return_5d = (
                spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[-5] - 1
            ) * 100

        for _, row in candidates.iterrows():
            symbol = row["symbol"]
            try:
                score = self._score_single(symbol, row, spy_return_5d, ml_model)
                if score:
                    scores.append(score)
            except Exception as e:
                logger.debug(f"Scoring failed for {symbol}: {e}")
                continue

        if not scores:
            return pd.DataFrame()

        df = pd.DataFrame(scores)
        return df

    def _score_single(
        self, symbol: str, info: dict, spy_return_5d: float, ml_model=None
    ) -> dict | None:
        """Score a single penny stock candidate."""

        # Fetch 3 months of daily data
        df = self.data_loader.fetch_price_data(symbol, period="3mo", interval="1d")
        if df.empty or len(df) < 20:
            return None

        close = df["Close"]
        volume = df["Volume"]

        # ── 1. VOLUME SURGE SCORE (0–100) ──
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0
        vol_score = min(100, vol_ratio * 33.3)  # 3x avg = 100

        # ── 2. MOMENTUM SCORE (0–100) ──
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        ret_10d = (close.iloc[-1] / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0
        # Positive momentum = higher score
        momentum_score = min(100, max(0, 50 + ret_5d * 5 + ret_10d * 2.5))

        # ── 3. TECHNICAL SETUP SCORE (0–100) ──
        tech_score = self._compute_technical_score(df)

        # ── 4. SENTIMENT SCORE (0–100) ──
        sent_score = self._compute_sentiment_score(symbol)

        # ── 5. RELATIVE STRENGTH SCORE (0–100) ──
        stock_ret_5d = ret_5d
        rs = stock_ret_5d - spy_return_5d
        rs_score = min(100, max(0, 50 + rs * 5))

        # ── WEIGHTED TOTAL ──
        total = (
            vol_score * self.weights["volume_surge"]
            + momentum_score * self.weights["momentum"]
            + tech_score * self.weights["technical"]
            + sent_score * self.weights["sentiment"]
            + rs_score * self.weights["relative_strength"]
        )

        # ── ML CONFIDENCE OVERLAY ──
        ml_confidence = 0.0
        if ml_model is not None:
            try:
                features = self.feature_engine.compute_features(df)
                if not features.empty:
                    pred = ml_model.predict(features.iloc[[-1]])
                    ml_confidence = pred["prob_buy"].iloc[0]
            except Exception as e:
                logger.debug(f"ML prediction failed for {symbol}: {e}")

        # Determine key catalyst
        catalyst = self._determine_catalyst(vol_ratio, ret_5d, tech_score, sent_score)

        return {
            "symbol": symbol,
            "name": info.get("name", symbol),
            "price": round(info.get("price", close.iloc[-1]), 2),
            "sector": info.get("sector", "Unknown"),
            "volume_surge_score": round(vol_score, 1),
            "momentum_score": round(momentum_score, 1),
            "technical_score": round(tech_score, 1),
            "sentiment_score": round(sent_score, 1),
            "relative_strength_score": round(rs_score, 1),
            "total_score": round(total, 1),
            "ml_confidence": round(ml_confidence, 3),
            "volume_ratio": round(vol_ratio, 2),
            "return_5d_pct": round(ret_5d, 2),
            "return_10d_pct": round(ret_10d, 2),
            "key_catalyst": catalyst,
            "suggested_stop_loss": round(close.iloc[-1] * 0.92, 2),  # 8% below
            "suggested_target": round(close.iloc[-1] * 1.15, 2),    # 15% above
        }

    def _compute_technical_score(self, df: pd.DataFrame) -> float:
        """Compute technical setup score (0–100)."""
        import ta as ta_lib

        close = df["Close"]
        score = 50.0  # Start neutral

        try:
            # RSI
            rsi = ta_lib.momentum.rsi(close, window=14).iloc[-1]
            if rsi < 30:
                score += 15  # Oversold bounce potential
            elif rsi < 45:
                score += 8
            elif rsi > 70:
                score -= 10  # Overbought

            # MACD
            macd = ta_lib.trend.MACD(close)
            macd_hist = macd.macd_diff().iloc[-1]
            macd_hist_prev = macd.macd_diff().iloc[-2] if len(macd.macd_diff()) > 1 else 0
            if macd_hist > 0 and macd_hist_prev <= 0:
                score += 15  # Fresh bullish crossover
            elif macd_hist > 0:
                score += 5

            # EMA crossover
            ema_9 = ta_lib.trend.ema_indicator(close, window=9).iloc[-1]
            ema_21 = ta_lib.trend.ema_indicator(close, window=21).iloc[-1]
            if ema_9 > ema_21:
                score += 10  # Short-term bullish
            if close.iloc[-1] > ema_9:
                score += 5   # Price above short EMA

            # Bollinger Band position
            bb = ta_lib.volatility.BollingerBands(close)
            bb_pct = bb.bollinger_pband().iloc[-1]
            if bb_pct < 0.2:
                score += 10  # Near lower band — potential reversal
            elif bb_pct > 0.9:
                score -= 5

        except Exception:
            pass

        return min(100, max(0, score))

    def _compute_sentiment_score(self, symbol: str) -> float:
        """Compute sentiment score (0–100) from news using fast financial lexicon (low RAM)."""
        if not self.data_loader.news_api_key:
            return 50.0  # Instant neutral score without slow network scraping
        try:
            articles = self.data_loader.fetch_news(symbol, days_back=3)
            if not articles:
                return 50.0  # Neutral if no news

            bullish_words = {
                "surge", "gain", "soar", "jump", "rally", "profit", "beat", "growth",
                "buy", "bullish", "approval", "fda", "partner", "contract", "record", "high", "upgrade"
            }
            bearish_words = {
                "drop", "fall", "plunge", "sink", "loss", "miss", "sell", "bearish",
                "dilution", "warning", "debt", "lawsuit", "investigation", "low", "downgrade"
            }

            pos_count = 0
            neg_count = 0
            for a in articles:
                text = (str(a.get("title", "")) + " " + str(a.get("description", ""))).lower()
                pos_count += sum(1 for w in bullish_words if w in text)
                neg_count += sum(1 for w in bearish_words if w in text)

            total = pos_count + neg_count
            if total == 0:
                return 50.0

            net = (pos_count - neg_count) / total
            return float(min(100.0, max(0.0, 50.0 + net * 50.0)))

        except Exception as e:
            logger.debug(f"Sentiment scoring failed for {symbol}: {e}")
            return 50.0

    def _determine_catalyst(
        self, vol_ratio: float, ret_5d: float, tech_score: float, sent_score: float
    ) -> str:
        """Determine the primary catalyst for the pick."""
        catalysts = []

        if vol_ratio > 3:
            catalysts.append("Volume surge")
        elif vol_ratio > 2:
            catalysts.append("High volume")

        if ret_5d > 10:
            catalysts.append("Strong momentum")
        elif ret_5d > 5:
            catalysts.append("Momentum")

        if tech_score > 75:
            catalysts.append("Bullish setup")
        elif tech_score > 60:
            catalysts.append("Technical breakout")

        if sent_score > 70:
            catalysts.append("Positive news")
        elif sent_score < 30:
            catalysts.append("Contrarian (neg. sentiment)")

        if not catalysts:
            catalysts.append("Multi-factor")

        return " + ".join(catalysts[:2])

    def _print_report(self, picks: pd.DataFrame):
        """Print a formatted report to console."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console(legacy_windows=False)
            today = datetime.now().strftime("%b %d, %Y")

            table = Table(
                title=f"PENNY STOCK PICKS — {today}",
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
            )

            table.add_column("Rank", style="bold", justify="center", width=5)
            table.add_column("Ticker", style="bold cyan", width=7)
            table.add_column("Price", justify="right", width=7)
            table.add_column("Score", justify="center", width=7)
            table.add_column("ML Conf.", justify="center", width=8)
            table.add_column("Vol Ratio", justify="center", width=9)
            table.add_column("5d Ret%", justify="right", width=8)
            table.add_column("Catalyst", width=22)
            table.add_column("Stop-Loss", justify="right", width=9)
            table.add_column("Target", justify="right", width=9)

            for _, row in picks.iterrows():
                score_color = (
                    "green" if row["total_score"] >= 70
                    else "yellow" if row["total_score"] >= 50
                    else "red"
                )
                ml_pct = f"{row['ml_confidence']*100:.0f}%" if row["ml_confidence"] > 0 else "N/A"

                table.add_row(
                    str(row["rank"]),
                    row["symbol"],
                    f"${row['price']:.2f}",
                    f"[{score_color}]{row['total_score']:.0f}[/{score_color}]",
                    ml_pct,
                    f"{row['volume_ratio']:.1f}x",
                    f"{row['return_5d_pct']:+.1f}%",
                    row["key_catalyst"],
                    f"${row['suggested_stop_loss']:.2f}",
                    f"${row['suggested_target']:.2f}",
                )

            console.print()
            console.print(table)
            console.print()
            console.print(Panel(
                "[yellow]Penny stocks are HIGH RISK. Use strict stop-losses and "
                "never risk more than 1-2% of your account per trade.[/yellow]",
                border_style="yellow",
            ))

        except Exception:
            # Fallback without rich
            print(f"\n{'='*70}")
            print(f"  PENNY STOCK PICKS — {datetime.now().strftime('%b %d, %Y')}")
            print(f"{'='*70}")
            for _, row in picks.iterrows():
                print(
                    f"  #{row['rank']} {row['symbol']:>6} | "
                    f"${row['price']:.2f} | "
                    f"Score: {row['total_score']:.0f} | "
                    f"{row['key_catalyst']}"
                )
            print(f"{'='*70}\n")

    def _save_report(self, picks: pd.DataFrame):
        """Save picks to CSV and JSON."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        csv_path = self.report_dir / f"penny_picks_{date_str}.csv"
        json_path = self.report_dir / f"penny_picks_{date_str}.json"

        picks.to_csv(csv_path, index=False)
        picks.to_json(json_path, orient="records", indent=2)

        logger.info(f"Report saved: {csv_path}")


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scanner = PennyScanner()
    picks = scanner.scan()
    if not picks.empty:
        print(f"\nTop {len(picks)} picks generated!")
    else:
        print("\nNo picks found — check filters or data sources")
