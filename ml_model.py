"""
ml_model.py — Machine Learning Model Training & Prediction
==========================================================
Supports single models (XGBoost, LightGBM, CatBoost) and
ensemble methods (voting, stacking) with walk-forward purged
cross-validation, feature selection, and adaptive confidence.
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
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

logger = logging.getLogger(__name__)


class MLModel:
    """
    ML model for stock signal prediction.

    Supports:
        - XGBoost, LightGBM, and CatBoost
        - Ensemble methods: soft voting and stacking
        - Walk-forward (expanding window) validation with purge gap
        - Hyperparameter tuning via Optuna
        - Feature selection via importance-based pruning
        - Adaptive confidence threshold
        - Probability-based predictions with confidence thresholds
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config["strategy"].get("model_type", "xgboost")
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.selected_features = None  # After feature selection
        self.feature_selector = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.training_metrics = {}

        # Ensemble config
        model_cfg = self.config.get("model", {})
        self.ensemble_method = model_cfg.get("ensemble_method", "single")
        self.base_model_types = model_cfg.get("base_models", ["xgboost", "lightgbm"])
        self.enable_feature_selection = model_cfg.get("feature_selection", True)
        self.max_features = model_cfg.get("max_features", 30)
        self.feature_selection_mode = model_cfg.get("feature_selection_mode", "global")
        self.purge_gap = model_cfg.get("purge_gap_bars", 30)
        self.embargo_bars = model_cfg.get("embargo_bars", 10)
        self.use_sample_weights = model_cfg.get("use_sample_weights", True)
        self.use_smote = model_cfg.get("use_smote", False)
        self.min_probability_gap = model_cfg.get("min_probability_gap", 0.12)
        self.confidence_percentile = model_cfg.get("confidence_percentile", None)

        # Adaptive confidence tracking
        self._prediction_log = []  # List of (predicted, actual, confidence) tuples
        self._adaptive_threshold = None

    # ──────────────────────────────────────────────
    #  MODEL FACTORIES
    # ──────────────────────────────────────────────

    def _create_single_model(self, model_type: str, params: dict | None = None):
        """Create a single model instance."""
        if model_type == "xgboost":
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
            return xgb.XGBClassifier(**default_params)

        elif model_type == "lightgbm":
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
            return lgb.LGBMClassifier(**default_params)

        elif model_type == "catboost":
            from catboost import CatBoostClassifier

            default_params = {
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_seed": 42,
                "verbose": 0,
                "task_type": "CPU",
                "loss_function": "MultiClass",
            }
            if params:
                default_params.update(params)
            return CatBoostClassifier(**default_params)

        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )

        elif model_type == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_model(self, params: dict | None = None):
        """Create the model based on ensemble configuration."""
        if self.ensemble_method == "single":
            self.model = self._create_single_model(self.model_type, params)

        elif self.ensemble_method == "voting":
            self.model = self._create_voting_ensemble(params)

        elif self.ensemble_method == "stacking":
            self.model = self._create_stacking_ensemble(params)

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _create_voting_ensemble(self, params: dict | None = None):
        """Create a soft-voting ensemble of base models."""
        estimators = []
        for model_type in self.base_model_types:
            try:
                model = self._create_single_model(model_type, params)
                estimators.append((model_type, model))
            except Exception as e:
                logger.warning(f"Failed to create {model_type} for ensemble: {e}")

        if len(estimators) < 2:
            logger.warning("Not enough models for ensemble, falling back to single model")
            return self._create_single_model(self.model_type, params)

        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )

    def _create_stacking_ensemble(self, params: dict | None = None):
        """Create a stacking ensemble with a meta-learner."""
        estimators = []
        for model_type in self.base_model_types:
            try:
                model = self._create_single_model(model_type, params)
                estimators.append((model_type, model))
            except Exception as e:
                logger.warning(f"Failed to create {model_type} for stacking: {e}")

        if len(estimators) < 2:
            logger.warning("Not enough models for stacking, falling back to single model")
            return self._create_single_model(self.model_type, params)

        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.0,
                random_state=42,
            ),
            cv=3,  # Internal CV for generating meta-features
            stack_method="predict_proba",
            n_jobs=-1,
        )

    # ──────────────────────────────────────────────
    #  FEATURE SELECTION
    # ──────────────────────────────────────────────

    def _select_features(self, X: pd.DataFrame, y: np.ndarray, force_refit: bool = False) -> pd.DataFrame:
        """
        Select top features using importance-based pruning.

        Trains a quick XGBoost model to rank features, then keeps the top N.
        This reduces noise and overfitting, especially on small datasets.
        """
        if not self.enable_feature_selection or len(X.columns) <= self.max_features:
            self.selected_features = list(X.columns)
            return X

        # In global mode, if features already selected and valid, reuse them
        if not force_refit and getattr(self, "feature_selection_mode", "global") == "global" and self.selected_features:
            valid_feats = [f for f in self.selected_features if f in X.columns]
            if len(valid_feats) == len(self.selected_features):
                return X[valid_feats]

        logger.info(f"Feature selection: {len(X.columns)} → max {self.max_features}")

        try:
            import xgboost as xgb

            # Quick model for feature ranking
            selector_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
            selector_model.fit(X, y)

            # Get importances and select top features
            importances = pd.Series(
                selector_model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False)

            top_features = importances.head(self.max_features).index.tolist()
            self.selected_features = top_features

            logger.info(f"Selected {len(top_features)} features: {top_features[:10]}...")
            return X[top_features]

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            self.selected_features = list(X.columns)
            return X

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
        logger.info(f"Training {self.ensemble_method} ({self.model_type}) on "
                     f"{len(X)} samples, {len(X.columns)} features")

        self.feature_names = list(X.columns)

        # Encode labels: -1, 0, 1 → 0, 1, 2
        y_encoded = self.label_encoder.fit_transform(y)

        # Feature selection (force fresh selection on full dataset)
        X_selected = self._select_features(X, y_encoded, force_refit=True)

        # SMOTE oversampling for minority classes
        X_fit, y_fit = X_selected, y_encoded
        if getattr(self, "use_smote", True):
            try:
                from imblearn.over_sampling import SMOTE
                counts = pd.Series(y_encoded).value_counts()
                if counts.min() > 5:
                    k_neighbors = min(5, counts.min() - 1)
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_fit, y_fit = smote.fit_resample(X_selected, y_encoded)
                    logger.info(f"SMOTE applied: {len(X_selected)} → {len(X_fit)} samples (balanced across all classes)")
            except Exception as e:
                logger.warning(f"SMOTE skipped: {e}")

        # Compute balanced sample weights
        sample_weights = None
        if getattr(self, "use_sample_weights", True):
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight("balanced", y_fit)
            logger.info("Using balanced class weights for training")

        self._create_model(params)
        
        # Fit model
        if sample_weights is not None:
            try:
                self.model.fit(X_fit, y_fit, sample_weight=sample_weights)
            except Exception as e:
                logger.warning(f"Fitting with sample_weight failed ({e}), falling back to standard fit")
                self.model.fit(X_fit, y_fit)
        else:
            self.model.fit(X_fit, y_fit)

        # Training accuracy on original real data
        y_pred = self.model.predict(X_selected)
        train_acc = accuracy_score(y_encoded, y_pred)
        train_f1 = f1_score(y_encoded, y_pred, average="weighted", zero_division=0)

        existing_wf = self.training_metrics.get("walk_forward")
        self.training_metrics = {
            "train_accuracy": round(train_acc, 4),
            "train_f1": round(train_f1, 4),
            "n_samples": len(X),
            "n_features_original": len(X.columns),
            "n_features_selected": len(X_selected.columns),
            "model_type": self.model_type,
            "ensemble_method": self.ensemble_method,
            "label_distribution": dict(pd.Series(y).value_counts().to_dict()),
            "trained_at": datetime.now().isoformat(),
        }
        if existing_wf:
            self.training_metrics["walk_forward"] = existing_wf

        logger.info(f"Training accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        if self.selected_features:
            logger.info(f"Using {len(self.selected_features)} selected features")
        return self.training_metrics

    # ──────────────────────────────────────────────
    #  WALK-FORWARD VALIDATION (PURGED)
    # ──────────────────────────────────────────────

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        min_train_size: int = 252,  # ~1 year of daily bars
        params: dict | None = None,
        purge_gap: int | None = None,
        embargo: int | None = None,
    ) -> dict:
        """
        Walk-forward (expanding window) cross-validation with purge gap.

        Always trains on past data and tests on future data — no lookahead bias.
        Includes a purge gap between train and test to prevent label leakage.

        Args:
            X: Feature DataFrame (time-ordered)
            y: Label Series
            n_splits: Number of walk-forward splits
            min_train_size: Minimum training window size
            params: Model hyperparameters
            purge_gap: Gap in bars between train and test (>= label horizon)
            embargo: Gap in bars after test set

        Returns:
            Dict with per-fold and aggregate metrics
        """
        effective_purge = purge_gap if purge_gap is not None else max(self.purge_gap, 26)
        effective_embargo = embargo if embargo is not None else max(self.embargo_bars, 10)

        logger.info(f"Walk-forward validation: {n_splits} splits, min_train={min_train_size}, "
                     f"purge_gap={effective_purge}, embargo={effective_embargo}")

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

            # Apply purge gap: skip `effective_purge` bars between train and test to prevent label leakage
            test_start = train_end + effective_purge
            test_end = min(test_start + step_size, total_size)

            # Apply embargo: skip `embargo_bars` after each test set
            # (only affects training of subsequent folds, handled by train_end)

            if test_end <= test_start or test_start >= total_size:
                break

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            if len(X_test) < 5:
                continue

            # Encode labels
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            # Feature selection
            X_train_sel = self._select_features(X_train, y_train_enc, force_refit=(fold == 0))
            X_test_sel = X_test[self.selected_features] if self.selected_features else X_test

            # Apply SMOTE & sample weights inside fold
            X_fit_fold, y_fit_fold = X_train_sel, y_train_enc
            if getattr(self, "use_smote", True):
                try:
                    from imblearn.over_sampling import SMOTE
                    counts = pd.Series(y_train_enc).value_counts()
                    if counts.min() > 5:
                        k_neighbors = min(5, counts.min() - 1)
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                        X_fit_fold, y_fit_fold = smote.fit_resample(X_train_sel, y_train_enc)
                except Exception:
                    pass

            sample_weights_fold = None
            if getattr(self, "use_sample_weights", True):
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weights_fold = compute_sample_weight("balanced", y_fit_fold)

            # Train
            self._create_model(params)
            if sample_weights_fold is not None:
                try:
                    self.model.fit(X_fit_fold, y_fit_fold, sample_weight=sample_weights_fold)
                except Exception:
                    self.model.fit(X_fit_fold, y_fit_fold)
            else:
                self.model.fit(X_fit_fold, y_fit_fold)

            # Predict
            y_pred_enc = self.model.predict(X_test_sel)
            y_prob = self.model.predict_proba(X_test_sel)

            # Metrics
            acc = accuracy_score(y_test_enc, y_pred_enc)
            f1 = f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0)

            # Precision for BUY signals specifically
            buy_label = le.transform([1])[0] if 1 in le.classes_ else None
            if buy_label is not None:
                buy_precision = precision_score(
                    y_test_enc, y_pred_enc, labels=[buy_label], average="micro", zero_division=0
                )
            else:
                buy_precision = 0.0

            fold_info = {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "purge_gap": self.purge_gap,
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
                "buy_precision": round(buy_precision, 4),
            }
            fold_metrics.append(fold_info)
            logger.info(f"  Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f}, "
                        f"buy_prec={buy_precision:.4f} "
                        f"(train={len(X_train)}, test={len(X_test)})")

            all_y_true.extend(y_test_enc.tolist())
            all_y_pred.extend(y_pred_enc.tolist())

        if not all_y_true:
            return {"error": "No valid folds completed"}

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
            "purge_gap_bars": effective_purge,
            "embargo_bars": effective_embargo,
            "overall_accuracy": round(overall_acc, 4),
            "overall_f1": round(overall_f1, 4),
            "buy_signal_win_rate": round(win_rate, 4),
            "fold_metrics": fold_metrics,
            "classification_report": classification_report(
                all_y_true, all_y_pred, output_dict=True, zero_division=0
            ),
        }

        self.training_metrics["walk_forward"] = results
        logger.info(f"Walk-forward results: acc={overall_acc:.4f}, f1={overall_f1:.4f}, "
                     f"buy_win_rate={win_rate:.4f}")
        return results

    # ──────────────────────────────────────────────
    #  HYPERPARAMETER TUNING
    # ──────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 30,
        n_splits: int = 3,
        study_name: str = "ibkr_penny_study",
    ) -> dict:
        """
        Tune hyperparameters using Optuna with walk-forward validation.
        Persists trials in SQLite db (models/optuna_study.db) so tuning accumulates across days.

        Returns:
            Dict with best parameters, best F1 score, and total trials count
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        db_path = (self.model_dir / "optuna_study.db").resolve()
        storage_url = f"sqlite:///{db_path}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
        )
        existing_trials = len(study.trials)
        logger.info(f"📊 Persistent Optuna study '{study_name}' loaded ({existing_trials} prior trials)")
        logger.info(f"🚀 Running {n_trials} additional tuning trials...")

        # Save original ensemble method and temporarily switch to single for speed
        original_ensemble = self.ensemble_method
        self.ensemble_method = "single"

        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                    "n_jobs": 2,
                }
            elif self.model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                    "n_jobs": 2,
                    "verbose": -1,
                }
            elif self.model_type == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 80, 400),
                    "depth": trial.suggest_int("depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                    "thread_count": 2,
                    "verbose": False,
                }
            else:
                params = {}

            result = self.walk_forward_validate(X, y, n_splits=n_splits, params=params)
            return result.get("overall_f1", 0.0)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value
        total_trials = len(study.trials)

        logger.info(f"🏆 Total cumulative trials across all runs: {total_trials}")
        logger.info(f"🏆 Best F1 score found: {best_score:.4f}")
        logger.info(f"🏆 Best params: {best_params}")

        # Restore ensemble method and re-train with best params on full data
        self.ensemble_method = original_ensemble
        self.train(X, y, params=best_params)

        return {"best_params": best_params, "best_f1": best_score, "total_trials": total_trials}

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

        # Apply feature selection if active
        if self.selected_features:
            missing = set(self.selected_features) - set(X_aligned.columns)
            if missing:
                logger.warning(f"Missing features (filling with 0): {missing}")
                for col in missing:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[self.selected_features]
        elif self.feature_names:
            missing = set(self.feature_names) - set(X_aligned.columns)
            if missing:
                logger.warning(f"Missing features (filling with 0): {missing}")
                for col in missing:
                    X_aligned[col] = 0.0
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

        # Calculate probability gap between highest and second-highest class
        sorted_probs = np.sort(y_prob, axis=1)
        results["prob_gap"] = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Use adaptive threshold if available, otherwise use provided threshold
        effective_threshold = self.get_adaptive_threshold() or confidence_threshold

        # If confidence_percentile is configured and valid, compute dynamic percentile cutoff
        conf_percentile = getattr(self, "confidence_percentile", None)
        if conf_percentile is not None and conf_percentile > 0 and len(results) >= 20:
            pct_val = float(np.percentile(results["confidence"], conf_percentile))
            effective_threshold = max(effective_threshold, pct_val)

        min_gap = getattr(self, "min_probability_gap", 0.04)
        if min_gap is None:
            min_gap = 0.04

        # Filter signals: require BOTH sufficient confidence AND sufficient probability gap
        filtered_signals = []
        for raw_sig, conf, gap in zip(results["raw_signal"], results["confidence"], results["prob_gap"]):
            if raw_sig != 0 and conf >= effective_threshold and gap >= min_gap:
                filtered_signals.append(raw_sig)
            else:
                filtered_signals.append(0)

        results["signal"] = filtered_signals
        results["effective_threshold"] = effective_threshold

        return results

    # ──────────────────────────────────────────────
    #  ADAPTIVE CONFIDENCE THRESHOLD
    # ──────────────────────────────────────────────

    def log_prediction_outcome(self, predicted: int, actual: int, confidence: float):
        """Log a prediction outcome for adaptive threshold tracking."""
        self._prediction_log.append({
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only recent predictions
        monitoring_cfg = self.config.get("monitoring", {})
        window = monitoring_cfg.get("rolling_accuracy_window", 50)
        if len(self._prediction_log) > window * 2:
            self._prediction_log = self._prediction_log[-window:]

    def get_adaptive_threshold(self) -> float | None:
        """
        Compute adaptive confidence threshold based on recent prediction accuracy.

        Returns higher threshold when model has been less accurate recently,
        and lower threshold when model is performing well.
        """
        monitoring_cfg = self.config.get("monitoring", {})
        if not monitoring_cfg.get("enabled", False):
            return None

        window = monitoring_cfg.get("rolling_accuracy_window", 50)

        if len(self._prediction_log) < 20:
            return None  # Not enough data to adapt

        recent = self._prediction_log[-window:]

        # Compute rolling accuracy for non-HOLD predictions
        actionable = [p for p in recent if p["predicted"] != 0]
        if len(actionable) < 10:
            return None

        accuracy = sum(
            1 for p in actionable if p["predicted"] == p["actual"]
        ) / len(actionable)

        # Adaptive threshold: lower accuracy → higher required confidence
        # Base threshold from config
        base_threshold = self.config["strategy"].get("confidence_threshold", 0.28)
        training_acc = self.training_metrics.get("train_accuracy", 0.5)

        # If recent accuracy is much worse than training, raise threshold
        accuracy_ratio = accuracy / (training_acc + 1e-10)
        if accuracy_ratio < 0.7:
            # Model is degrading significantly — raise threshold by up to 50%
            adjusted = base_threshold * (1 + (1 - accuracy_ratio) * 0.5)
        elif accuracy_ratio > 1.1:
            # Model is doing better than training — slightly lower threshold
            adjusted = base_threshold * 0.9
        else:
            adjusted = base_threshold

        self._adaptive_threshold = round(np.clip(adjusted, 0.15, 0.70), 3)
        return self._adaptive_threshold

    def get_model_health(self) -> dict:
        """Get current model health metrics."""
        monitoring_cfg = self.config.get("monitoring", {})
        window = monitoring_cfg.get("rolling_accuracy_window", 50)

        if len(self._prediction_log) < 10:
            return {"status": "insufficient_data", "predictions_logged": len(self._prediction_log)}

        recent = self._prediction_log[-window:]
        actionable = [p for p in recent if p["predicted"] != 0]

        if not actionable:
            return {"status": "no_actionable_predictions"}

        accuracy = sum(1 for p in actionable if p["predicted"] == p["actual"]) / len(actionable)
        training_acc = self.training_metrics.get("train_accuracy", 0.5)
        decay_pct = (1 - accuracy / (training_acc + 1e-10)) * 100

        decay_threshold = monitoring_cfg.get("decay_threshold_pct", 10)

        return {
            "status": "healthy" if decay_pct < decay_threshold else "degraded",
            "rolling_accuracy": round(accuracy, 4),
            "training_accuracy": round(training_acc, 4),
            "accuracy_decay_pct": round(decay_pct, 2),
            "predictions_logged": len(self._prediction_log),
            "adaptive_threshold": self._adaptive_threshold,
            "needs_retrain": decay_pct >= decay_threshold,
        }

    # ──────────────────────────────────────────────
    #  FEATURE IMPORTANCE
    # ──────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.model is None:
            return pd.DataFrame()

        try:
            # For ensemble models, try to extract from base estimators
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "estimators_"):
                # Voting/Stacking: average importances from base estimators
                all_importances = []
                for item in self.model.estimators_:
                    est = item[1] if isinstance(item, (list, tuple)) else item
                    if hasattr(est, "feature_importances_"):
                        all_importances.append(est.feature_importances_)
                if all_importances:
                    importance = np.mean(all_importances, axis=0)
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

            names = self.selected_features or self.feature_names or [f"f_{i}" for i in range(len(importance))]

            # Handle mismatched lengths
            if len(names) != len(importance):
                names = [f"f_{i}" for i in range(len(importance))]

            df = pd.DataFrame({
                "feature": names,
                "importance": importance,
            }).sort_values("importance", ascending=False)

            return df.head(top_n).reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return pd.DataFrame()

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
            "selected_features": self.selected_features,
            "model_type": self.model_type,
            "ensemble_method": self.ensemble_method,
        }, model_path)

        with open(meta_path, "w") as f:
            json.dump({
                "model_type": self.model_type,
                "ensemble_method": self.ensemble_method,
                "feature_names": self.feature_names,
                "selected_features": self.selected_features,
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
        self.selected_features = data.get("selected_features")
        self.model_type = data["model_type"]
        self.ensemble_method = data.get("ensemble_method", "single")

        # Load training metrics from meta file
        meta_path = self.model_dir / f"{name}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.training_metrics = meta.get("training_metrics", {})

        logger.info(f"Model loaded from {model_path} (ensemble={self.ensemble_method})")


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

    # Model health
    health = model.get_model_health()
    print(f"\nModel health: {health}")
