"""
sentiment.py — Financial News Sentiment Analysis
=================================================
Uses FinBERT (a finance-domain BERT model) to classify news
headlines/articles as Positive, Negative, or Neutral with
confidence scores.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.

    FinBERT is a BERT model fine-tuned on financial text.
    It classifies text into: Positive, Negative, Neutral.
    """

    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier for FinBERT
        """
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy-load the model (only when first needed)."""
        if self._model is not None:
            return

        logger.info(f"Loading sentiment model: {self.model_name}")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Sentiment model loaded successfully")

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: The text to analyze (headline or article snippet)

        Returns:
            Dict with keys:
                - label: 'positive', 'negative', or 'neutral'
                - score: Confidence score (0–1) for the predicted label
                - scores: Dict of all label scores
        """
        self._load_model()

        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 0.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = probs.cpu().numpy()[0]

            # Map to labels
            scores = {label: float(probs[i]) for i, label in enumerate(self.LABELS)}
            predicted_label = self.LABELS[np.argmax(probs)]
            confidence = float(np.max(probs))

            return {
                "label": predicted_label,
                "score": confidence,
                "scores": scores,
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "label": "neutral",
                "score": 0.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment result dicts
        """
        self._load_model()
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Filter empty strings
            batch = [t if t and t.strip() else "neutral" for t in batch]

            try:
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                probs = probs.cpu().numpy()

                for j, p in enumerate(probs):
                    scores = {label: float(p[k]) for k, label in enumerate(self.LABELS)}
                    predicted_label = self.LABELS[np.argmax(p)]
                    confidence = float(np.max(p))
                    results.append({
                        "text": batch[j][:100],
                        "label": predicted_label,
                        "score": confidence,
                        "scores": scores,
                    })

            except Exception as e:
                logger.error(f"Batch sentiment failed: {e}")
                for t in batch:
                    results.append({
                        "text": t[:100],
                        "label": "neutral",
                        "score": 0.0,
                        "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                    })

        return results

    def score_articles(self, articles: list[dict]) -> list[dict]:
        """
        Score a list of news articles (from data_loader).

        Args:
            articles: List of article dicts with 'title' and 'description' keys

        Returns:
            Articles enriched with sentiment scores
        """
        texts = []
        for article in articles:
            # Combine title + description for richer context
            title = article.get("title", "")
            desc = article.get("description", "")
            combined = f"{title}. {desc}" if desc else title
            texts.append(combined)

        sentiments = self.analyze_batch(texts)

        enriched = []
        for article, sentiment in zip(articles, sentiments):
            enriched.append({
                **article,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "sentiment_positive": sentiment["scores"]["positive"],
                "sentiment_negative": sentiment["scores"]["negative"],
                "sentiment_neutral": sentiment["scores"]["neutral"],
            })

        return enriched

    CATALYST_PATTERNS = {
        "catalyst_earnings": r"\b(earnings|quarterly|revenue|eps|guidance|financial results)\b",
        "catalyst_fda": r"\b(fda|approval|clinical trial|phase [123]|orphan drug|fast track|nda|bla)\b",
        "catalyst_merger": r"\b(merger|acquisition|buyout|takeover|definitive agreement|tender offer)\b",
        "catalyst_offering": r"\b(offering|public offering|direct offering|underwritten|prospectus|shelf)\b",
        "catalyst_dilution": r"\b(dilution|warrant|convertible note|registered direct|atm offering|reverse split)\b",
    }

    def compute_aggregate_sentiment(
        self,
        scored_articles: list[dict],
        days_windows: list[int] = [1, 3, 7],
        as_of_time: Optional[pd.Timestamp] = None,
    ) -> dict:
        """
        Compute aggregate sentiment and catalyst features from scored articles.
        Enforces strict timestamp alignment (no lookahead leakage).

        Args:
            scored_articles: List of article dicts with sentiment scores and published_at
            days_windows: Rolling lookback windows in days
            as_of_time: Maximum allowable timestamp for articles (lookahead protection)

        Returns:
            Dict with sentiment scores, news volumes, recency, and catalyst binary flags.
        """
        catalyst_defaults = {cat: 0 for cat in self.CATALYST_PATTERNS}
        if not scored_articles:
            features = {}
            for d in days_windows:
                features[f"avg_sentiment_{d}d"] = 0.0
                features[f"news_count_{d}d"] = 0
            features["sentiment_score"] = 0.0
            features["days_since_last_news"] = 30.0
            features["max_positive_score"] = 0.0
            features["max_negative_score"] = 0.0
            features["sentiment_momentum"] = 0.0
            features.update(catalyst_defaults)
            return features

        df = pd.DataFrame(scored_articles)

        # Convert published_at to datetime
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            df["published_at"] = df["published_at"].fillna(pd.Timestamp.now(tz="UTC"))
        else:
            df["published_at"] = pd.Timestamp.now(tz="UTC")

        # Normalize UTC timezone
        try:
            if df["published_at"].dt.tz is None:
                df["published_at"] = df["published_at"].dt.tz_localize("UTC")
            else:
                df["published_at"] = df["published_at"].dt.tz_convert("UTC")
        except Exception:
            df["published_at"] = pd.Timestamp.now(tz="UTC")

        if as_of_time is not None:
            now = pd.to_datetime(as_of_time)
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            else:
                now = now.tz_convert("UTC")
        else:
            now = pd.Timestamp.now(tz="UTC")

        # CRITICAL LEAKAGE FILTER: only articles published at or before now
        df = df[df["published_at"] <= now]

        if df.empty:
            features = {}
            for d in days_windows:
                features[f"avg_sentiment_{d}d"] = 0.0
                features[f"news_count_{d}d"] = 0
            features["sentiment_score"] = 0.0
            features["days_since_last_news"] = 30.0
            features["max_positive_score"] = 0.0
            features["max_negative_score"] = 0.0
            features["sentiment_momentum"] = 0.0
            features.update(catalyst_defaults)
            return features

        # Days since last news event
        latest_pub = df["published_at"].max()
        days_since = max(0.0, round((now - latest_pub).total_seconds() / 86400.0, 2))

        # Compute net sentiment: positive - negative
        if "sentiment_positive" in df.columns and "sentiment_negative" in df.columns:
            df["net_sentiment"] = df["sentiment_positive"] - df["sentiment_negative"]
        else:
            df["net_sentiment"] = 0.0

        features = {}
        sentiment_by_window = {}

        for d in days_windows:
            cutoff = now - pd.Timedelta(days=d)
            window_df = df[df["published_at"] >= cutoff]

            if len(window_df) > 0 and not window_df["net_sentiment"].isna().all():
                avg_sent = float(window_df["net_sentiment"].mean())
            else:
                avg_sent = 0.0

            if pd.isna(avg_sent):
                avg_sent = 0.0

            features[f"avg_sentiment_{d}d"] = round(avg_sent, 4)
            features[f"news_count_{d}d"] = len(window_df)
            sentiment_by_window[d] = avg_sent

        # Primary sentiment_score alias (3-day window)
        features["sentiment_score"] = features.get("avg_sentiment_3d", 0.0)
        features["days_since_last_news"] = days_since

        # Max scores (safe computation)
        pos_max = df["sentiment_positive"].max() if ("sentiment_positive" in df.columns and len(df) > 0) else 0.0
        neg_max = df["sentiment_negative"].max() if ("sentiment_negative" in df.columns and len(df) > 0) else 0.0

        features["max_positive_score"] = round(float(pos_max), 4) if pd.notna(pos_max) else 0.0
        features["max_negative_score"] = round(float(neg_max), 4) if pd.notna(neg_max) else 0.0

        # Sentiment momentum: recent vs older
        if 3 in sentiment_by_window and 7 in sentiment_by_window:
            momentum = sentiment_by_window[3] - sentiment_by_window[7]
            features["sentiment_momentum"] = round(float(momentum), 4) if pd.notna(momentum) else 0.0
        else:
            features["sentiment_momentum"] = 0.0

        # Catalyst detection across all articles in window (up to 7 days)
        recent_cutoff = now - pd.Timedelta(days=7)
        recent_df = df[df["published_at"] >= recent_cutoff]
        combined_texts = " ".join(
            (recent_df.get("title", pd.Series(dtype=str)).fillna("") + " " +
             recent_df.get("description", pd.Series(dtype=str)).fillna("")).str.lower()
        )

        for cat_name, pattern in self.CATALYST_PATTERNS.items():
            features[cat_name] = 1 if bool(re.search(pattern, combined_texts, re.IGNORECASE)) else 0

        # Guarantee no NaN exists
        for k, v in features.items():
            if v is None or pd.isna(v):
                features[k] = 0.0

        return features


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SentimentAnalyzer()

    test_headlines = [
        "Apple reports record quarterly revenue, beating analyst expectations",
        "Tesla shares plunge 15% after disappointing earnings and weak guidance",
        "Federal Reserve holds interest rates steady, markets react cautiously",
        "NVIDIA announces groundbreaking AI chip, stock surges to all-time high",
        "Bankrupt crypto firm files for Chapter 11, investors lose billions",
    ]

    print("\n=== Sentiment Analysis Results ===\n")
    results = analyzer.analyze_batch(test_headlines)
    for r in results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[r["label"]]
        print(f"{emoji} [{r['label'].upper():>8}] ({r['score']:.2f}) {r['text']}")
