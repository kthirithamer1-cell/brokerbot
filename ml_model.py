"""
ml_model.py — Machine Learning Model Training & Prediction
==========================================================
Trains XGBoost / LightGBM on engineered features using
walk-forward validation. Supports hyperparameter tuning
with Optuna.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class MLModel:
    """
    ML model for stock signal prediction.

    Supports:
        - XGBoost and LightGBM
        - Walk-forward (expanding window) validation
        - Hyperparameter tuning via Optuna
        - Probability-based predictions with confidence thresholds
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config["strategy"].get("model_type", "xgboost")
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.training_metrics = {}

    def _create_model(self, params: dict | None = None):
        """Create a fresh model instance with given or default params."""
        if self.model_type == "xgboost":
            import xgboost as xgb

            default_params = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "random_state": 42,
                "n_jobs": -1,
            }
            if params:
                default_params.update(params)
            self.model = xgb.XGBClassifier(**default_params)

        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            default_params = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "multiclass",
                "num_class": 3,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            if params:
                default_params.update(params)
            self.model = lgb.LGBMClassifier(**default_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # ──────────────────────────────────────────────
    #  TRAINING
    # ──────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict | None = None,
    ) -> dict:
        """
        Train the model on the full dataset.

        Args:
            X: Feature DataFrame
            y: Label Series (-1, 0, 1)
            params: Optional model hyperparameters

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training {self.model_type} on {len(X)} samples, {len(X.columns)} features")

        self.feature_names = list(X.columns)

        # Encode labels: -1, 0, 1 → 0, 1, 2
        y_encoded = self.label_encoder.fit_transform(y)

        self._create_model(params)
        self.model.fit(X, y_encoded)

        # Training accuracy
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y_encoded, y_pred)

        self.training_metrics = {
            "train_accuracy": round(train_acc, 4),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "model_type": self.model_type,
            "label_distribution": dict(pd.Series(y).value_counts().to_dict()),
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"Training accuracy: {train_acc:.4f}")
        return self.training_metrics

    # ──────────────────────────────────────────────
    #  WALK-FORWARD VALIDATION
    # ──────────────────────────────────────────────

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        min_train_size: int = 252,  # ~1 year of daily bars
        params: dict | None = None,
    ) -> dict:
        """
        Walk-forward (expanding window) cross-validation.

        Always trains on past data and tests on future data — no lookahead bias.

        Args:
            X: Feature DataFrame (time-ordered)
            y: Label Series
            n_splits: Number of walk-forward splits
            min_train_size: Minimum training window size
            params: Model hyperparameters

        Returns:
            Dict with per-fold and aggregate metrics
        """
        logger.info(f"Walk-forward validation: {n_splits} splits, min_train={min_train_size}")

        self.feature_names = list(X.columns)
        total_size = len(X)
        step_size = (total_size - min_train_size) // n_splits

        if step_size < 20:
            logger.warning("Not enough data for walk-forward validation")
            return {"error": "Insufficient data"}

        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for fold in range(n_splits):
            train_end = min_train_size + fold * step_size
            test_end = min(train_end + step_size, total_size)

            if test_end <= train_end:
                break

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # Encode labels
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            # Train
            self._create_model(params)
            self.model.fit(X_train, y_train_enc)

            # Predict
            y_pred_enc = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)

            # Metrics
            acc = accuracy_score(y_test_enc, y_pred_enc)
            f1 = f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0)

            fold_info = {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
            }
            fold_metrics.append(fold_info)
            logger.info(f"  Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f} "
                        f"(train={len(X_train)}, test={len(X_test)})")

            all_y_true.extend(y_test_enc.tolist())
            all_y_pred.extend(y_pred_enc.tolist())

        # Store the label encoder from the last fold for consistency
        self.label_encoder = le

        # Aggregate metrics
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)

        # Win rate: % of BUY signals that were correct
        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        buy_label = le.transform([1])[0] if 1 in le.classes_ else None

        if buy_label is not None:
            buy_mask = y_pred_arr == buy_label
            if buy_mask.sum() > 0:
                win_rate = (y_true_arr[buy_mask] == buy_label).mean()
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0

        results = {
            "n_splits": n_splits,
            "overall_accuracy": round(overall_acc, 4),
            "overall_f1": round(overall_f1, 4),
            "buy_signal_win_rate": round(win_rate, 4),
            "fold_metrics": fold_metrics,
            "classification_report": classification_report(
                all_y_true, all_y_pred, output_dict=True, zero_division=0
            ),
        }

        self.training_metrics = results
        logger.info(f"Overall: acc={overall_acc:.4f}, f1={overall_f1:.4f}, "
                     f"buy_win_rate={win_rate:.4f}")
        return results

    # ──────────────────────────────────────────────
    #  HYPERPARAMETER TUNING
    # ──────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        n_splits: int = 3,
    ) -> dict:
        """
        Tune hyperparameters using Optuna with walk-forward validation.

        Returns:
            Best parameters dict
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"Starting Optuna tuning: {n_trials} trials")

        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                }
            else:  # lightgbm
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                }

            result = self.walk_forward_validate(X, y, n_splits=n_splits, params=params)
            return result.get("overall_f1", 0.0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best F1 score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        # Re-train with best params on full data
        self.train(X, y, params=best_params)

        return {"best_params": best_params, "best_f1": best_score}

    # ──────────────────────────────────────────────
    #  PREDICTION
    # ──────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate predictions with confidence scores.

        Args:
            X: Feature DataFrame
            confidence_threshold: Minimum probability to issue a signal

        Returns:
            DataFrame with columns: signal, confidence, prob_buy, prob_hold, prob_sell
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Align features
        X_aligned = X.copy()
        if self.feature_names:
            missing = set(self.feature_names) - set(X_aligned.columns)
            if missing:
                logger.warning(f"Missing features (filling with 0): {missing}")
                for col in missing:
                    X_aligned[col] = 0.0
            # Ensure exact column order
            X_aligned = X_aligned[self.feature_names]

        y_pred_enc = self.model.predict(X_aligned)
        y_prob = self.model.predict_proba(X_aligned)

        # Decode labels back to -1, 0, 1
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)

        # Build results DataFrame
        results = pd.DataFrame(index=X.index)

        # Map probabilities to labels
        classes = self.label_encoder.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}

        results["prob_sell"] = y_prob[:, class_to_idx.get(-1, 0)] if -1 in class_to_idx else 0
        results["prob_hold"] = y_prob[:, class_to_idx.get(0, 1)] if 0 in class_to_idx else 0
        results["prob_buy"] = y_prob[:, class_to_idx.get(1, 2)] if 1 in class_to_idx else 0

        results["raw_signal"] = y_pred
        results["confidence"] = np.max(y_prob, axis=1)

        # Apply confidence threshold
        results["signal"] = results.apply(
            lambda row: row["raw_signal"]
            if row["confidence"] >= confidence_threshold
            else 0,
            axis=1,
        )

        return results

    # ──────────────────────────────────────────────
    #  FEATURE IMPORTANCE
    # ──────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_
        names = self.feature_names or [f"f_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df.head(top_n).reset_index(drop=True)

    # ──────────────────────────────────────────────
    #  SAVE / LOAD
    # ──────────────────────────────────────────────

    def save(self, name: str = "trading_model"):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save")

        model_path = self.model_dir / f"{name}.joblib"
        meta_path = self.model_dir / f"{name}_meta.json"

        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }, model_path)

        with open(meta_path, "w") as f:
            json.dump({
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")

    def load(self, name: str = "trading_model"):
        """Load a trained model."""
        model_path = self.model_dir / f"{name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        data = joblib.load(model_path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.model_type = data["model_type"]

        logger.info(f"Model loaded from {model_path}")


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data for testing
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25]))

    model = MLModel()
    model._create_model = lambda params=None: setattr(model, "model_type", "xgboost") or model._create_model.__wrapped__(model, params) if hasattr(model._create_model, "__wrapped__") else None

    # Simple train test
    model = MLModel()
    metrics = model.train(X, y)
    print(f"\nTraining metrics: {metrics}")

    # Walk-forward
    wf = model.walk_forward_validate(X, y, n_splits=3)
    print(f"\nWalk-forward results:")
    print(f"  Accuracy: {wf['overall_accuracy']}")
    print(f"  F1 Score: {wf['overall_f1']}")
    print(f"  Buy Win Rate: {wf['buy_signal_win_rate']}")
