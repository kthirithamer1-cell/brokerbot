"""
penny_scanner.py — Penny Stock Scanner & Daily Picks
=====================================================
Scans the market for penny stocks ($0.50–$5) with the highest
short-term upside potential using multi-factor scoring and
ML confidence overlay.
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup UTF-8 safe streams on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

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
        min_price = self.scanner_cfg.get("min_price", 0.1)
        max_price = self.scanner_cfg.get("max_price", 5.00)
        min_volume = self.scanner_cfg.get("min_avg_volume", 100_000)
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
                if not info or not info.get("is_valid", True):
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

        # Fetch macro context once for ML model feature engineering (1y for 200 SMA indicator warm-up)
        macro_df = self.data_loader.fetch_macro_context(period="1y", interval="1d")

        for _, row in candidates.iterrows():
            symbol = row["symbol"]
            try:
                score = self._score_single(symbol, row, spy_return_5d, ml_model, macro_df=macro_df)
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
        self, symbol: str, info: dict, spy_return_5d: float, ml_model=None, macro_df=None
    ) -> dict | None:
        """Score a single penny stock candidate for 5-day momentum setups."""

        # Fetch 1 year of daily data (requires >=200 bars for SMA 200 / ATR / MACD warm-up)
        df = self.data_loader.fetch_price_data(symbol, period="1y", interval="1d")
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

        # ── 4. SENTIMENT & CATALYST SCORE (0–100) ──
        sent_score, catalyst_flags, articles = self._compute_sentiment_and_catalysts(symbol)

        # ── 5. RELATIVE STRENGTH SCORE (0–100) ──
        stock_ret_5d = ret_5d
        rs = stock_ret_5d - spy_return_5d
        rs_score = min(100, max(0, 50 + rs * 5))

        # ── 6. FUNDAMENTAL / SHORT SQUEEZE SCORE ──
        full_info = self.data_loader.fetch_full_info(symbol)
        float_shares = full_info.get("float_shares") or 0.0
        short_pct = full_info.get("short_pct_float") or 0.0

        # Penalize dilution/offering heavily (major penny stock risk)
        if catalyst_flags.get("catalyst_dilution") or catalyst_flags.get("catalyst_offering"):
            sent_score = max(0.0, sent_score - 30.0)

        # Boost if low float and short interest
        if 0 < float_shares < 15_000_000 and short_pct > 0.15:
            vol_score = min(100.0, vol_score + 10.0)

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
                features = self.feature_engine.compute_features(
                    df,
                    macro_df=macro_df,
                    quote_info=info,
                    fundamental_data=full_info,
                    news_articles=articles,
                )
                if not features.empty:
                    pred_features = features.drop(columns=["Close"], errors="ignore")
                    pred = ml_model.predict(pred_features.iloc[[-1]])
                    ml_confidence = pred["prob_buy"].iloc[0]
            except Exception as e:
                logger.debug(f"ML prediction failed for {symbol}: {e}")

        # Determine key catalyst
        catalyst = self._determine_catalyst(vol_ratio, ret_5d, tech_score, sent_score, catalyst_flags)

        price_val = round(info.get("price", close.iloc[-1]), 2)
        shares_30 = int(30.0 / price_val) if price_val > 0 else 0
        stop_loss = round(close.iloc[-1] * 0.92, 2)           # -8% risk stop
        target_25 = round(close.iloc[-1] * 1.25, 2)           # +25% 5-day base target
        target_40 = round(close.iloc[-1] * 1.40, 2)           # +40% 5-day runner target
        est_profit_30 = round(shares_30 * (target_25 - price_val), 2)

        return {
            "symbol": symbol,
            "name": info.get("name", symbol),
            "price": price_val,
            "sector": full_info.get("sector") or info.get("sector", "Unknown"),
            "volume_surge_score": round(vol_score, 1),
            "momentum_score": round(momentum_score, 1),
            "technical_score": round(tech_score, 1),
            "sentiment_score": round(sent_score, 1),
            "relative_strength_score": round(rs_score, 1),
            "total_score": round(total, 1),
            "ml_confidence": round(ml_confidence, 3),
            "volume_ratio": round(vol_ratio, 2),
            "rel_volume": info.get("rel_volume", round(vol_ratio, 2)),
            "spread_pct": info.get("spread_pct"),
            "return_5d_pct": round(ret_5d, 2),
            "return_10d_pct": round(ret_10d, 2),
            "key_catalyst": catalyst,
            "suggested_stop_loss": stop_loss,
            "suggested_target": target_25,
            "extended_target": target_40,
            "shares_on_30": shares_30,
            "est_profit_30": est_profit_30,
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

            # Resistance breakout (20-day high breakout like ORBS / TNON)
            if len(df) >= 21:
                high_20 = df["High"].iloc[-21:-1].max()
                if close.iloc[-1] >= high_20:
                    score += 15  # Fresh 20-day breakout!

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

    def _compute_sentiment_and_catalysts(self, symbol: str) -> tuple[float, dict, list]:
        """Compute sentiment score (0–100) and detect catalyst flags from news."""
        try:
            articles = self.data_loader.fetch_news(symbol, days_back=3)
            if not articles:
                return 50.0, {}, []

            catalyst_flags = {}
            from sentiment import SentimentAnalyzer
            combined_texts = " ".join(
                (str(a.get("title", "")) + " " + str(a.get("description", ""))).lower()
                for a in articles
            )
            for cat_name, pattern in SentimentAnalyzer.CATALYST_PATTERNS.items():
                catalyst_flags[cat_name] = 1 if bool(re.search(pattern, combined_texts, re.IGNORECASE)) else 0

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
                return 50.0, catalyst_flags, articles

            net = (pos_count - neg_count) / total
            score = float(min(100.0, max(0.0, 50.0 + net * 50.0)))
            return score, catalyst_flags, articles

        except Exception as e:
            logger.debug(f"Sentiment scoring failed for {symbol}: {e}")
            return 50.0, {}, []

    def _determine_catalyst(
        self,
        vol_ratio: float,
        ret_5d: float,
        tech_score: float,
        sent_score: float,
        catalyst_flags: Optional[dict] = None,
    ) -> str:
        """Determine the primary catalyst for the pick."""
        catalysts = []
        flags = catalyst_flags or {}

        # High-priority fundamental catalyst flags
        if flags.get("catalyst_dilution"):
            catalysts.append("⚠️ Dilution Alert")
        if flags.get("catalyst_fda"):
            catalysts.append("💊 FDA Catalyst")
        if flags.get("catalyst_earnings"):
            catalysts.append("📊 Earnings")
        if flags.get("catalyst_merger"):
            catalysts.append("🤝 M&A")

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

        if sent_score > 70 and not flags.get("catalyst_fda") and not flags.get("catalyst_earnings"):
            catalysts.append("Positive news")

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
                title=f"🚀 5-DAY PENNY STOCK MOMENTUM SCANNER — {today}",
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
            )

            table.add_column("Rank", style="bold", justify="center", width=4)
            table.add_column("Ticker", style="bold cyan", width=6)
            table.add_column("Price", justify="right", width=6)
            table.add_column("Score", justify="center", width=5)
            table.add_column("ML Conf.", justify="center", width=8)
            table.add_column("5d Ret%", justify="right", width=8)
            table.add_column("Vol", justify="center", width=5)
            table.add_column("Catalyst", width=20)
            table.add_column("Stop (-8%)", justify="right", width=8)
            table.add_column("5d Target (+25%)", justify="right", width=12)
            table.add_column("Runner (+40%)", justify="right", width=10)
            table.add_column("Shares ($30)", justify="center", width=9)
            table.add_column("Est Gain", justify="right", style="bold green", width=9)

            for _, row in picks.iterrows():
                score_color = (
                    "green" if row["total_score"] >= 70
                    else "yellow" if row["total_score"] >= 50
                    else "red"
                )
                ml_pct = f"{row['ml_confidence']*100:.0f}%" if row.get("ml_confidence", 0) > 0 else "N/A"
                shares = row.get("shares_on_30", 0)
                est_p = row.get("est_profit_30", 0.0)
                est_str = f"+${est_p:.2f}" if est_p > 0 else "$0.00"

                table.add_row(
                    str(row["rank"]),
                    row["symbol"],
                    f"${row['price']:.2f}",
                    f"[{score_color}]{row['total_score']:.0f}[/{score_color}]",
                    ml_pct,
                    f"{row['return_5d_pct']:+.1f}%",
                    f"{row['volume_ratio']:.1f}x",
                    row["key_catalyst"],
                    f"${row['suggested_stop_loss']:.2f}",
                    f"${row['suggested_target']:.2f}",
                    f"${row.get('extended_target', row['suggested_target']):.2f}",
                    str(shares),
                    est_str,
                )

            console.print()
            console.print(table)
            console.print()
            console.print(Panel(
                "[bold cyan]🎯 $30 ➔ $60 GROWTH BLUEPRINT (Volume Breakouts):[/bold cyan]\n"
                "• [bold]Strategy:[/bold] Enter high-RVOL volume breakouts (like ORBS / TNON) targeting +20% to +30%.\n"
                "• [bold]Golden Exit Rule:[/bold] Take profit immediately when target is reached. Do NOT hold blindly for 5 days (e.g. ORBS surged +26% on Day 1 then faded!).\n"
                "• [bold]Cash Account Rule:[/bold] Hold until target hit (1 to 4 days) -> cash settles under T+1 -> rotate 100% into next runner.\n"
                "• [bold]Fee Advantage:[/bold] IBKR Tiered is only ~$0.35/order ($0.70 round-trip). 3–4 trades = ~$2.50 fees total.\n"
                "• [bold]Risk Rule:[/bold] Cut losses immediately at -8% stop loss ($2.40 max loss on $30 capital). Never hold a failing breakout.",
                border_style="green",
            ))

        except Exception as e:
            # Fallback without rich
            logger.warning(f"Rich console table render error: {e}")
            print(f"\n{'='*70}")
            print(f"  5-DAY PENNY STOCK PICKS -- {datetime.now().strftime('%b %d, %Y')}")
            print(f"{'='*70}")
            for _, row in picks.iterrows():
                cat = str(row.get("key_catalyst", "")).encode("ascii", errors="replace").decode("ascii")
                print(
                    f"  #{row['rank']} {row['symbol']:>6} | "
                    f"${row['price']:.2f} | "
                    f"Score: {row['total_score']:.0f} | "
                    f"5d Target: ${row.get('suggested_target', 0):.2f} (+25%) | "
                    f"{cat}"
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
