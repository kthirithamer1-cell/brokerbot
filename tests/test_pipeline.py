import pytest
import numpy as np
import pandas as pd
from ml_model import MLModel
from risk_manager import RiskManager
from model_monitor import ModelMonitor


def test_ml_model_train_predict_pipeline(tmp_path):
    np.random.seed(42)
    n_samples = 300
    n_features = 15
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3]))

    model = MLModel()
    metrics = model.train(X, y)
    assert "train_accuracy" in metrics
    assert metrics["train_accuracy"] > 0

    preds = model.predict(X, confidence_threshold=0.28)
    assert not preds.empty
    assert "signal" in preds.columns
    assert "confidence" in preds.columns
    assert "prob_buy" in preds.columns
    assert set(preds["signal"].unique()).issubset({-1, 0, 1})


def test_risk_manager_regime_and_targets():
    rm = RiskManager()
    rm.update_equity(100.0)

    # Test regime stop adjustments
    rm.update_regime("trending")
    sl_trend, tp_trend = rm.get_regime_stops()
    assert sl_trend == 1.2
    assert tp_trend == 3.5

    rm.update_regime("ranging")
    sl_range, tp_range = rm.get_regime_stops()
    assert sl_range == 2.5
    assert tp_range == 1.8

    # Test position sizing and bracket levels
    bracket = rm.compute_bracket_levels(entry_price=5.0, atr=0.2, direction=1)
    assert bracket["stop_loss"] < 5.0
    assert bracket["take_profit"] > 5.0

    sizing = rm.calculate_position_size(entry_price=5.0, stop_loss_price=4.5)
    assert sizing["shares"] >= 0
    assert "size_multiplier" in sizing


def test_model_monitor_journal_and_report():
    monitor = ModelMonitor()
    monitor.set_training_baseline(accuracy=0.55)

    monitor.log_prediction(
        symbol="TEST",
        predicted_signal=1,
        confidence=0.65,
        prob_buy=0.65,
        prob_sell=0.15,
        prob_hold=0.20,
        actual_outcome=1,
        pnl=1.50,
        regime="trending",
    )

    report = monitor.get_health_report()
    assert "overall_status" in report
    assert report["total_predictions"] >= 1
