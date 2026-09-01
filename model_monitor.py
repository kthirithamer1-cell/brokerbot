"""
model_monitor.py — Model Performance Monitoring
================================================
Tracks live model health, prediction accuracy, signal quality,
feature drift, and auto-retrain triggers.

Designed to answer: "Is the model still working well enough to trade?"
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Live model performance monitoring system.

    Features:
        - Rolling accuracy tracker over recent predictions
        - Model decay detection (accuracy vs training baseline)
        - Signal quality scoring by confidence bucket
        - Auto-retrain trigger when accuracy degrades
        - Feature drift detection (live vs training distributions)
        - Trade journal with full prediction + outcome logging
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        monitoring_cfg = self.config.get("monitoring", {})
        self.enabled = monitoring_cfg.get("enabled", True)
        self.rolling_window = monitoring_cfg.get("rolling_accuracy_window", 50)
        self.decay_threshold_pct = monitoring_cfg.get("decay_threshold_pct", 10)
        self.auto_retrain = monitoring_cfg.get("auto_retrain", True)
        self.feature_drift_threshold = monitoring_cfg.get("feature_drift_threshold", 2.0)

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.journal_path = self.log_dir / "prediction_journal.jsonl"
        self.health_path = self.log_dir / "model_health.json"

        # In-memory prediction journal
        self._predictions = []
        self._training_feature_stats = {}  # {feature_name: {mean, std}}
        self._training_accuracy = 0.0

        # Load existing journal
        self._load_journal()

    def _load_journal(self):
        """Load existing prediction journal from disk."""
        if self.journal_path.exists():
            try:
                with open(self.journal_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._predictions.append(json.loads(line))
                logger.debug(f"Loaded {len(self._predictions)} predictions from journal")
            except Exception as e:
                logger.warning(f"Failed to load prediction journal: {e}")

    def set_training_baseline(
        self,
        accuracy: float,
        feature_stats: dict[str, dict] | None = None,
    ):
        """
        Set the training accuracy baseline for decay detection.

        Args:
            accuracy: Training accuracy from walk-forward validation
            feature_stats: Dict of {feature_name: {"mean": float, "std": float}}
        """
        self._training_accuracy = accuracy
        if feature_stats:
            self._training_feature_stats = feature_stats

        logger.info(f"Training baseline set: accuracy={accuracy:.4f}, "
                     f"features={len(feature_stats) if feature_stats else 0}")

    def log_prediction(
        self,
        symbol: str,
        predicted_signal: int,
        confidence: float,
        prob_buy: float,
        prob_sell: float,
        prob_hold: float,
        actual_outcome: int | None = None,
        pnl: float | None = None,
        regime: str = "unknown",
        features: dict | None = None,
    ):
        """
        Log a prediction for tracking.

        Args:
            symbol: Ticker symbol
            predicted_signal: Model's prediction (1=BUY, -1=SELL, 0=HOLD)
            confidence: Model's confidence score
            prob_buy/sell/hold: Individual class probabilities
            actual_outcome: Actual outcome (1=profit, -1=loss, 0=flat, None=pending)
            pnl: Realized P&L if trade was closed
            regime: Market regime at time of prediction
            features: Optional feature values for drift detection
        """
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "predicted_signal": predicted_signal,
            "confidence": round(confidence, 4),
            "prob_buy": round(prob_buy, 4),
            "prob_sell": round(prob_sell, 4),
            "prob_hold": round(prob_hold, 4),
            "actual_outcome": actual_outcome,
            "pnl": round(pnl, 4) if pnl is not None else None,
            "regime": regime,
        }

        self._predictions.append(entry)

        # Append to journal file
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write prediction journal: {e}")

        # Check for feature drift
        if features:
            self._check_feature_drift(features)

    def update_outcome(
        self,
        symbol: str,
        actual_outcome: int,
        pnl: float,
        timestamp: str | None = None,
    ):
        """
        Update the outcome of a pending prediction.

        Finds the most recent prediction for the symbol that has no outcome
        and fills in the actual result.
        """
        for entry in reversed(self._predictions):
            if entry["symbol"] == symbol and entry["actual_outcome"] is None:
                entry["actual_outcome"] = actual_outcome
                entry["pnl"] = round(pnl, 4)
                break

    def get_rolling_accuracy(self) -> dict:
        """
        Compute rolling accuracy over the last N predictions.

        Returns:
            Dict with accuracy metrics for different signal types
        """
        # Filter to predictions with known outcomes
        completed = [
            p for p in self._predictions[-self.rolling_window * 2:]
            if p["actual_outcome"] is not None and p["predicted_signal"] != 0
        ]

        if len(completed) < 5:
            return {"status": "insufficient_data", "count": len(completed)}

        recent = completed[-self.rolling_window:]

        # Overall accuracy (did the signal direction match the outcome?)
        correct = sum(
            1 for p in recent
            if p["predicted_signal"] == p["actual_outcome"]
        )
        accuracy = correct / len(recent)

        # BUY signal accuracy
        buys = [p for p in recent if p["predicted_signal"] == 1]
        buy_accuracy = (
            sum(1 for p in buys if p["actual_outcome"] == 1) / len(buys)
            if buys else 0
        )

        # SELL signal accuracy
        sells = [p for p in recent if p["predicted_signal"] == -1]
        sell_accuracy = (
            sum(1 for p in sells if p["actual_outcome"] == -1) / len(sells)
            if sells else 0
        )

        # Average PnL per signal
        buy_pnls = [p["pnl"] for p in buys if p["pnl"] is not None]
        sell_pnls = [p["pnl"] for p in sells if p["pnl"] is not None]

        return {
            "status": "ok",
            "count": len(recent),
            "overall_accuracy": round(accuracy, 4),
            "buy_accuracy": round(buy_accuracy, 4),
            "sell_accuracy": round(sell_accuracy, 4),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "avg_buy_pnl": round(np.mean(buy_pnls), 4) if buy_pnls else 0,
            "avg_sell_pnl": round(np.mean(sell_pnls), 4) if sell_pnls else 0,
        }

    def get_signal_quality_by_confidence(self) -> dict:
        """
        Break down signal quality by confidence buckets.

        Answers: "Are high-confidence signals actually better?"
        """
        completed = [
            p for p in self._predictions
            if p["actual_outcome"] is not None and p["predicted_signal"] != 0
        ]

        if len(completed) < 10:
            return {"status": "insufficient_data"}

        # Bucket by confidence
        buckets = {
            "low (< 0.30)": [],
            "medium (0.30-0.45)": [],
            "high (0.45-0.60)": [],
            "very_high (> 0.60)": [],
        }

        for p in completed:
            conf = p["confidence"]
            if conf < 0.30:
                buckets["low (< 0.30)"].append(p)
            elif conf < 0.45:
                buckets["medium (0.30-0.45)"].append(p)
            elif conf < 0.60:
                buckets["high (0.45-0.60)"].append(p)
            else:
                buckets["very_high (> 0.60)"].append(p)

        result = {}
        for bucket_name, predictions in buckets.items():
            if not predictions:
                result[bucket_name] = {"count": 0, "accuracy": 0, "avg_pnl": 0}
                continue

            accuracy = sum(
                1 for p in predictions
                if p["predicted_signal"] == p["actual_outcome"]
            ) / len(predictions)

            pnls = [p["pnl"] for p in predictions if p["pnl"] is not None]
            avg_pnl = np.mean(pnls) if pnls else 0

            result[bucket_name] = {
                "count": len(predictions),
                "accuracy": round(accuracy, 4),
                "avg_pnl": round(avg_pnl, 4),
                "win_rate": round(
                    sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0, 1
                ),
            }

        return result

    def detect_model_decay(self) -> dict:
        """
        Detect if the model's live performance has degraded from training.

        Returns:
            Dict with decay status and recommendation
        """
        rolling = self.get_rolling_accuracy()
        if rolling.get("status") != "ok":
            return {"status": "insufficient_data", "needs_retrain": False}

        live_accuracy = rolling["overall_accuracy"]
        training_accuracy = self._training_accuracy

        if training_accuracy <= 0:
            return {"status": "no_training_baseline", "needs_retrain": False}

        decay_pct = (1 - live_accuracy / training_accuracy) * 100

        needs_retrain = decay_pct >= self.decay_threshold_pct

        status = {
            "status": "degraded" if needs_retrain else "healthy",
            "live_accuracy": live_accuracy,
            "training_accuracy": training_accuracy,
            "decay_pct": round(decay_pct, 2),
            "threshold_pct": self.decay_threshold_pct,
            "needs_retrain": needs_retrain,
        }

        if needs_retrain:
            logger.warning(
                f"⚠️ MODEL DECAY DETECTED: Live accuracy {live_accuracy:.2%} is "
                f"{decay_pct:.1f}% below training ({training_accuracy:.2%}). "
                f"{'Auto-retrain triggered.' if self.auto_retrain else 'Manual retrain recommended.'}"
            )

        return status

    def _check_feature_drift(self, features: dict):
        """
        Check if live feature values have drifted from training distribution.

        Uses z-score: if a feature's live value is >2 std from training mean,
        flag it as drifted.
        """
        if not self._training_feature_stats:
            return

        drifted = []
        for name, value in features.items():
            if name in self._training_feature_stats:
                stats = self._training_feature_stats[name]
                train_mean = stats.get("mean", 0)
                train_std = stats.get("std", 1)

                if train_std > 0:
                    zscore = abs(value - train_mean) / train_std
                    if zscore > self.feature_drift_threshold:
                        drifted.append({
                            "feature": name,
                            "live_value": round(value, 4),
                            "train_mean": round(train_mean, 4),
                            "zscore": round(zscore, 2),
                        })

        if drifted:
            logger.warning(
                f"⚠️ Feature drift detected: {len(drifted)} features significantly "
                f"different from training. Top: {drifted[0]['feature']} (z={drifted[0]['zscore']:.1f})"
            )

    def get_health_report(self) -> dict:
        """
        Generate a comprehensive model health report.

        Returns:
            Dict with all health metrics combined
        """
        rolling = self.get_rolling_accuracy()
        decay = self.detect_model_decay()
        quality = self.get_signal_quality_by_confidence()

        # Overall status
        if decay.get("needs_retrain"):
            overall_status = "DEGRADED — Retrain recommended"
        elif rolling.get("overall_accuracy", 0) < 0.35:
            overall_status = "POOR — Model underperforming"
        elif rolling.get("overall_accuracy", 0) >= 0.50:
            overall_status = "HEALTHY — Model performing well"
        else:
            overall_status = "OK — Model performing adequately"

        report = {
            "overall_status": overall_status,
            "generated_at": datetime.now().isoformat(),
            "total_predictions": len(self._predictions),
            "rolling_accuracy": rolling,
            "model_decay": decay,
            "signal_quality": quality,
        }

        # Save to disk
        try:
            with open(self.health_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save health report: {e}")

        return report

    def get_regime_performance(self) -> dict:
        """Break down model performance by market regime."""
        completed = [
            p for p in self._predictions
            if p["actual_outcome"] is not None and p["predicted_signal"] != 0
        ]

        if len(completed) < 10:
            return {"status": "insufficient_data"}

        regimes = {}
        for p in completed:
            regime = p.get("regime", "unknown")
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(p)

        result = {}
        for regime, predictions in regimes.items():
            correct = sum(
                1 for p in predictions
                if p["predicted_signal"] == p["actual_outcome"]
            )
            pnls = [p["pnl"] for p in predictions if p["pnl"] is not None]

            result[regime] = {
                "count": len(predictions),
                "accuracy": round(correct / len(predictions), 4),
                "avg_pnl": round(np.mean(pnls), 4) if pnls else 0,
                "total_pnl": round(sum(pnls), 4) if pnls else 0,
            }

        return result

    def print_health_report(self):
        """Print a formatted health report to console."""
        report = self.get_health_report()

        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console(legacy_windows=False)

            # Overall status
            status = report["overall_status"]
            status_color = "green" if "HEALTHY" in status else "yellow" if "OK" in status else "red"
            console.print(Panel(
                f"[{status_color}]{status}[/{status_color}]\n"
                f"Total predictions: {report['total_predictions']}",
                title="🧠 Model Health Report",
                border_style=status_color,
            ))

            # Rolling accuracy
            rolling = report["rolling_accuracy"]
            if rolling.get("status") == "ok":
                table = Table(title="Rolling Accuracy", border_style="blue")
                table.add_column("Metric", style="bold")
                table.add_column("Value", justify="right")
                table.add_row("Overall", f"{rolling['overall_accuracy']:.2%}")
                table.add_row("BUY signals", f"{rolling['buy_accuracy']:.2%} ({rolling['buy_count']} trades)")
                table.add_row("SELL signals", f"{rolling['sell_accuracy']:.2%} ({rolling['sell_count']} trades)")
                table.add_row("Avg BUY PnL", f"${rolling['avg_buy_pnl']:+.2f}")
                table.add_row("Avg SELL PnL", f"${rolling['avg_sell_pnl']:+.2f}")
                console.print(table)

            # Decay detection
            decay = report["model_decay"]
            if decay.get("status") not in ("insufficient_data", "no_training_baseline"):
                decay_color = "red" if decay["needs_retrain"] else "green"
                console.print(f"\n[{decay_color}]Model Decay: {decay['decay_pct']:.1f}% "
                              f"(threshold: {decay['threshold_pct']}%)[/{decay_color}]")

            # Signal quality by confidence
            quality = report["signal_quality"]
            if quality.get("status") != "insufficient_data":
                table = Table(title="Signal Quality by Confidence", border_style="magenta")
                table.add_column("Confidence Bucket")
                table.add_column("Count", justify="right")
                table.add_column("Accuracy", justify="right")
                table.add_column("Win Rate", justify="right")
                table.add_column("Avg PnL", justify="right")

                for bucket, stats in quality.items():
                    if isinstance(stats, dict) and stats.get("count", 0) > 0:
                        table.add_row(
                            bucket,
                            str(stats["count"]),
                            f"{stats['accuracy']:.2%}",
                            f"{stats['win_rate']:.0f}%",
                            f"${stats['avg_pnl']:+.2f}",
                        )
                console.print(table)

        except ImportError:
            print(f"\n{'='*50}")
            print(f"  MODEL HEALTH REPORT")
            print(f"{'='*50}")
            print(f"  Status: {report['overall_status']}")
            print(f"  Predictions: {report['total_predictions']}")
            if report["rolling_accuracy"].get("status") == "ok":
                print(f"  Accuracy: {report['rolling_accuracy']['overall_accuracy']:.2%}")
            print(f"{'='*50}\n")


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = ModelMonitor()
    monitor.set_training_baseline(accuracy=0.55)

    # Simulate some predictions
    np.random.seed(42)
    for i in range(100):
        signal = np.random.choice([-1, 1])
        confidence = np.random.uniform(0.25, 0.65)
        outcome = signal if np.random.random() < 0.52 else -signal
        pnl = np.random.uniform(-2, 3) if outcome == signal else np.random.uniform(-3, 1)

        monitor.log_prediction(
            symbol="TEST",
            predicted_signal=signal,
            confidence=confidence,
            prob_buy=confidence if signal == 1 else (1 - confidence) / 2,
            prob_sell=confidence if signal == -1 else (1 - confidence) / 2,
            prob_hold=(1 - confidence),
            actual_outcome=outcome,
            pnl=pnl,
            regime=np.random.choice(["trending", "ranging", "volatile"]),
        )

    # Print health report
    monitor.print_health_report()

    # Regime breakdown
    regime_perf = monitor.get_regime_performance()
    print(f"\nRegime Performance: {json.dumps(regime_perf, indent=2)}")
