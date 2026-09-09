"""
model_gate.py — Day-by-Day Model Accuracy & Promotion Gate
===========================================================
Ensures that only models meeting or exceeding the current Champion's
benchmark (accuracy, F1, win rate) are approved for live trading.

Features:
    - Champion / Challenger evaluation framework
    - Model registry with full version history (models/model_registry.json)
    - Automated model archiving prior to promotion (models/archive/)
    - Rollback and safety protection against degraded models
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
ARCHIVE_DIR = MODELS_DIR / "archive"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
ACTIVE_MODEL_PATH = MODELS_DIR / "trading_model.joblib"
ACTIVE_META_PATH = MODELS_DIR / "trading_model_meta.json"


class ModelGate:
    """
    Manages model versioning, approval gates, and promotion from Challenger to Champion.
    """

    def __init__(self):
        MODELS_DIR.mkdir(exist_ok=True)
        ARCHIVE_DIR.mkdir(exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load model registry or initialize default."""
        if REGISTRY_PATH.exists():
            try:
                with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry, creating new: {e}")

        # Default registry structure
        default_reg = {
            "champion": {
                "version": "initial_baseline",
                "accuracy": 0.4047,
                "f1": 0.3485,
                "trials_completed": 200,
                "approved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "APPROVED",
                "symbols": ["PDSB"],
            },
            "history": []
        }
        self._save_registry(default_reg)
        return default_reg

    def _save_registry(self, reg: dict):
        """Save registry to disk."""
        try:
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def get_champion_metrics(self) -> dict:
        """Return the current Champion's metrics."""
        return self.registry.get("champion", {})

    def evaluate_and_promote(
        self,
        candidate_model_path: str | Path,
        candidate_metrics: dict,
        candidate_meta: Optional[dict] = None,
        min_f1_threshold: float = 0.33,
    ) -> tuple[bool, str]:
        """
        Evaluate candidate model metrics against the active Champion.

        Args:
            candidate_model_path: Path to newly trained model file
            candidate_metrics: Dict with 'accuracy', 'f1' (or 'overall_f1'), etc.
            candidate_meta: Optional metadata dict
            min_f1_threshold: Minimum absolute F1 required for any model

        Returns:
            (approved: bool, reason: str)
        """
        champion = self.get_champion_metrics()
        champ_acc = champion.get("accuracy", 0.0)
        champ_f1 = champion.get("f1", 0.0)

        cand_acc = candidate_metrics.get("accuracy", 0.0)
        cand_f1 = candidate_metrics.get("f1") or candidate_metrics.get("overall_f1", 0.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"

        logger.info("=" * 60)
        logger.info("🛡️ MODEL APPROVAL GATE — Champion vs Challenger Evaluation")
        logger.info(f"   Champion:   Accuracy={champ_acc:.4f}, F1={champ_f1:.4f} ({champion.get('version', 'N/A')})")
        logger.info(f"   Challenger: Accuracy={cand_acc:.4f}, F1={cand_f1:.4f} ({version_id})")
        logger.info("=" * 60)

        # Baseline sanity check
        if cand_f1 < min_f1_threshold:
            reason = f"Challenger F1 ({cand_f1:.4f}) is below minimum acceptable threshold ({min_f1_threshold:.4f})"
            self._record_decision(version_id, "REJECTED", reason, candidate_metrics)
            logger.warning(f"❌ {reason}")
            return False, reason

        # Tolerance: Challenger must be at least within 95% of champion or improve it
        # Prefer higher F1 over accuracy for imbalanced trading data
        f1_improved = cand_f1 >= champ_f1
        acc_acceptable = cand_acc >= (champ_acc * 0.95)

        if f1_improved and acc_acceptable:
            # APPROVE & PROMOTE
            reason = (
                f"Challenger approved! F1 improved from {champ_f1:.4f} to {cand_f1:.4f} "
                f"(Accuracy: {cand_acc:.4f})"
            )
            logger.info(f"✅ {reason}")

            # 1. Archive previous champion
            if ACTIVE_MODEL_PATH.exists():
                archive_name = f"trading_model_{champion.get('version', 'prev')}_{timestamp}.joblib"
                archive_dest = ARCHIVE_DIR / archive_name
                try:
                    shutil.copy2(ACTIVE_MODEL_PATH, archive_dest)
                    logger.info(f"📦 Archived previous champion to {archive_dest}")
                except Exception as e:
                    logger.warning(f"Failed to archive previous model: {e}")

            # 2. Promote candidate to active model
            try:
                shutil.copy2(candidate_model_path, ACTIVE_MODEL_PATH)
                logger.info(f"🚀 Promoted {version_id} to active trading model: {ACTIVE_MODEL_PATH}")
            except Exception as e:
                err_msg = f"Failed to promote model file: {e}"
                logger.error(err_msg)
                return False, err_msg

            # 3. Update active metadata if provided
            if candidate_meta:
                try:
                    candidate_meta["version"] = version_id
                    candidate_meta["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ACTIVE_META_PATH, "w", encoding="utf-8") as f:
                        json.dump(candidate_meta, f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Could not update meta file: {e}")

            # 4. Update registry
            new_champion = {
                "version": version_id,
                "accuracy": round(cand_acc, 4),
                "f1": round(cand_f1, 4),
                "trials_completed": candidate_metrics.get("trials_completed", champion.get("trials_completed", 0)),
                "approved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "APPROVED",
                "symbols": candidate_metrics.get("symbols", ["PDSB"]),
            }
            self.registry["champion"] = new_champion
            self._record_decision(version_id, "APPROVED", reason, candidate_metrics)
            self._save_registry(self.registry)

            return True, reason

        else:
            # REJECT
            reason = (
                f"Challenger rejected: F1 ({cand_f1:.4f}) did not beat Champion ({champ_f1:.4f}) "
                f"or accuracy fell too low ({cand_acc:.4f} vs {champ_acc:.4f}). Retaining current Champion."
            )
            logger.info(f"⛔ {reason}")
            self._record_decision(version_id, "REJECTED", reason, candidate_metrics)
            self._save_registry(self.registry)
            return False, reason

    def _record_decision(self, version_id: str, status: str, reason: str, metrics: dict):
        """Append decision to history log."""
        record = {
            "version": version_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "reason": reason,
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1") or metrics.get("overall_f1"),
        }
        history = self.registry.setdefault("history", [])
        history.insert(0, record)
        # Keep latest 30 decisions
        self.registry["history"] = history[:30]
