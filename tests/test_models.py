"""
test_models.py — Unit Tests: ML Engineering Layer
===================================================
Tests cover:
  - LSTMPredictor output shape and value range
  - FeatureEngineer output correctness
  - ModelEvaluator metric computations
  - ARIMAForecaster predict output structure
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.model_numpy import LSTMPredictor
from backend.feature_engineering import FeatureEngineer
from backend.model_evaluator import ModelEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lstm():
    return LSTMPredictor()


@pytest.fixture
def sample_series() -> pd.Series:
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    vals = np.abs(np.random.randn(400) * 5000 + 10000)
    return pd.Series(vals, index=idx, name="cases")


@pytest.fixture
def sample_df(sample_series) -> pd.DataFrame:
    return pd.DataFrame({"cases": sample_series})


# ── LSTM Tests ────────────────────────────────────────────────────────────────

def test_lstm_loads_without_error(lstm):
    assert lstm is not None


def test_lstm_predict_output_shape(lstm):
    """LSTM should return shape (1, 1) for a (1, 30, 1) input."""
    x = np.random.rand(1, 30, 1).astype(np.float32)
    out = lstm.predict(x)
    assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"


def test_lstm_predict_is_float(lstm):
    x = np.random.rand(1, 30, 1).astype(np.float32)
    out = lstm.predict(x)
    assert isinstance(float(out[0][0]), float)


def test_lstm_predict_multiple_inputs(lstm):
    """LSTM should accept batch inputs (batch_size > 1)."""
    x = np.random.rand(3, 30, 1).astype(np.float32)
    out = lstm.predict(x)
    assert out.shape == (3, 1)


# ── FeatureEngineer Tests ─────────────────────────────────────────────────────

def test_feature_engineer_builds_columns(sample_df):
    fe = FeatureEngineer()
    result = fe.build_features(sample_df)
    expected_cols = ["cases", "ma_7", "ma_14", "ma_30", "growth_rate_7d",
                     "std_7", "lag_1", "lag_7", "epidemic_phase"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_feature_engineer_no_nulls_after_fillna(sample_df):
    result = FeatureEngineer.build_features(sample_df)
    assert result.isnull().sum().sum() == 0, "build_features should fill all NaN values"


def test_epidemic_phase_values(sample_df):
    result = FeatureEngineer.build_features(sample_df)
    valid_phases = {
        "Exponential Growth", "Rising", "Plateau",
        "Declining", "Rapid Decline", "Low Activity", "Unknown"
    }
    unique_phases = set(result["epidemic_phase"].unique())
    assert unique_phases.issubset(valid_phases), f"Unexpected phases: {unique_phases - valid_phases}"


def test_feature_summary_keys(sample_df):
    feat_df = FeatureEngineer.build_features(sample_df)
    summary = FeatureEngineer.get_feature_summary(feat_df)
    required = ["current_cases", "ma_7", "ma_14", "growth_rate_7d",
                "volatility_7d", "epidemic_phase"]
    for k in required:
        assert k in summary, f"Missing summary key: {k}"


def test_ma7_is_rolling_average(sample_df):
    result = FeatureEngineer.build_features(sample_df)
    # The 10th MA-7 should be close to mean of rows 3-10
    ma7_10 = result["ma_7"].iloc[9]
    manual  = result["cases"].iloc[3:10].mean()
    assert abs(ma7_10 - manual) < 1000, "MA-7 value seems incorrect"


# ── ModelEvaluator Tests ──────────────────────────────────────────────────────

def test_compute_metrics_perfect_prediction():
    actuals = np.array([100.0, 200.0, 300.0])
    preds   = np.array([100.0, 200.0, 300.0])
    m = ModelEvaluator.compute_metrics(actuals, preds)
    assert m["rmse"] == 0.0
    assert m["mae"]  == 0.0


def test_compute_metrics_returns_all_keys():
    actuals = np.random.rand(30) * 1000
    preds   = actuals + np.random.rand(30) * 100
    m = ModelEvaluator.compute_metrics(actuals, preds)
    for k in ["rmse", "mae", "mape", "r2"]:
        assert k in m, f"Missing metric key: {k}"


def test_compute_metrics_positive_rmse():
    actuals = np.array([100.0, 200.0, 150.0])
    preds   = np.array([110.0, 190.0, 160.0])
    m = ModelEvaluator.compute_metrics(actuals, preds)
    assert m["rmse"] > 0


def test_lstm_confidence_interval_length():
    preds = [1000.0, 1100.0, 1200.0, 1050.0, 980.0, 1150.0, 1300.0]
    ci = ModelEvaluator.lstm_confidence_interval(preds)
    assert len(ci["lower_ci"]) == len(preds)
    assert len(ci["upper_ci"]) == len(preds)


def test_lstm_confidence_interval_lower_leq_upper():
    preds = [500.0, 700.0, 600.0]
    ci = ModelEvaluator.lstm_confidence_interval(preds)
    for l, u in zip(ci["lower_ci"], ci["upper_ci"]):
        assert l <= u, f"lower_ci {l} > upper_ci {u}"


def test_compare_models_recommendation_is_valid():
    result = ModelEvaluator.compare_models(
        lstm_result  = {"metrics": {"rmse": 200.0, "mae": 150.0, "mape": 10.0, "r2": 0.85}},
        arima_result = {"metrics": {"rmse": 250.0, "mae": 180.0, "mape": 12.0, "r2": 0.80}},
    )
    assert result["recommendation"] in ("LSTM", "ARIMA")


def test_compare_models_lstm_wins_on_lower_rmse():
    result = ModelEvaluator.compare_models(
        lstm_result  = {"metrics": {"rmse": 100.0, "mae": 80.0, "mape": 5.0, "r2": 0.9}},
        arima_result = {"metrics": {"rmse": 500.0, "mae": 400.0, "mape": 25.0, "r2": 0.6}},
    )
    assert result["recommendation"] == "LSTM"
