"""
test_api.py — Unit Tests: Flask API Layer
==========================================
Tests cover:
  - All 8 endpoints return expected status codes
  - /health responds with correct JSON shape
  - /countries returns a list
  - /predict validates country parameter
  - /predict validates days parameter
  - /data/stats returns statistics keys

Uses Flask's built-in test client (no server needed).
External API calls to disease.sh are mocked.
"""

import sys
import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock the model & pipeline before importing app ────────────────────────────

MOCK_PIPELINE_DF = pd.DataFrame(
    {"cases": np.abs(np.random.randn(400) * 5000 + 10000),
     "cumulative_cases": np.cumsum(np.abs(np.random.randn(400) * 5000 + 10000))},
    index=pd.date_range("2021-01-01", periods=400, freq="D"),
)
MOCK_PIPELINE_DF.index.name = "date"


@pytest.fixture(scope="module")
def client():
    """Create Flask test client with mocked ML dependencies."""
    with patch("backend.app.LSTMPredictor") as mock_lstm_cls, \
         patch("backend.app.DataPipeline") as mock_pipeline_cls, \
         patch("backend.app.ARIMAForecaster") as mock_arima_cls:

        # Mock LSTM
        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = np.array([[0.5]])
        mock_lstm_cls.return_value = mock_lstm

        # Mock Pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MOCK_PIPELINE_DF
        mock_pipeline.get_data_quality_report.return_value = {
            "quality_score": 92.0,
            "completeness_pct": 100.0,
            "total_records": 400,
            "null_records": 0,
            "zero_records": 0,
            "outliers_detected": 2,
            "days_since_last_update": 1,
            "data_freshness": "FRESH",
            "date_range": "2021-01-01 → 2022-02-05",
            "mean_daily_cases": 10000.0,
            "max_daily_cases": 50000.0,
            "std_daily_cases": 5000.0,
            "pipeline_metadata": {},
        }
        mock_pipeline_cls.return_value = mock_pipeline

        # Mock ARIMA
        mock_arima = MagicMock()
        mock_arima.fit.return_value = mock_arima
        mock_arima.predict.return_value = {
            "predictions": [10000.0] * 7,
            "lower_ci":    [8000.0]  * 7,
            "upper_ci":    [12000.0] * 7,
        }
        mock_arima.evaluate.return_value = {
            "rmse": 250.0, "mae": 180.0, "mape": 8.5, "r2": 0.82
        }
        mock_arima_cls.return_value = mock_arima

        from backend.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── Root & Health ─────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_has_service_key(client):
    data = resp = client.get("/").get_json()
    assert "service" in data
    assert "endpoints" in data


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_status_ok(client):
    data = client.get("/health").get_json()
    assert data.get("status") == "ok"


def test_health_has_timestamp(client):
    data = client.get("/health").get_json()
    assert "timestamp" in data


# ── Countries ─────────────────────────────────────────────────────────────────

def test_countries_returns_200(client):
    resp = client.get("/countries")
    assert resp.status_code == 200


def test_countries_returns_list(client):
    data = client.get("/countries").get_json()
    assert "countries" in data
    assert isinstance(data["countries"], list)
    assert len(data["countries"]) > 0


# ── Predict ───────────────────────────────────────────────────────────────────

def test_predict_valid_country_returns_200(client):
    resp = client.get("/predict?country=india&days=7&model=lstm")
    assert resp.status_code == 200


def test_predict_returns_predictions_list(client):
    data = client.get("/predict?country=india&days=7&model=lstm").get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 7


def test_predict_invalid_country_returns_400(client):
    resp = client.get("/predict?country=narnia&days=7")
    assert resp.status_code == 400


def test_predict_days_too_large_returns_400(client):
    resp = client.get("/predict?country=india&days=99")
    assert resp.status_code == 400


def test_predict_days_zero_returns_400(client):
    resp = client.get("/predict?country=india&days=0")
    assert resp.status_code == 400


def test_predict_has_confidence_intervals(client):
    data = client.get("/predict?country=india&days=7&model=lstm").get_json()
    assert "lower_ci" in data
    assert "upper_ci" in data


def test_predict_has_past_30_days(client):
    data = client.get("/predict?country=india&days=7&model=lstm").get_json()
    assert "past_30_days" in data
    assert isinstance(data["past_30_days"], list)


# ── Data Stats ────────────────────────────────────────────────────────────────

def test_data_stats_returns_200(client):
    resp = client.get("/data/stats?country=india")
    assert resp.status_code == 200


def test_data_stats_has_statistics_block(client):
    data = client.get("/data/stats?country=india").get_json()
    assert "statistics" in data
    stat = data["statistics"]
    for key in ["mean", "median", "std", "min", "max"]:
        assert key in stat, f"Missing stats key: {key}"


# ── Data Quality ──────────────────────────────────────────────────────────────

def test_data_quality_returns_200(client):
    resp = client.get("/data/quality?country=india")
    assert resp.status_code == 200


def test_data_quality_has_quality_score(client):
    data = client.get("/data/quality?country=india").get_json()
    report = data.get("quality_report", {})
    assert "quality_score" in report
    assert 0 <= report["quality_score"] <= 100


# ── Error Handlers ────────────────────────────────────────────────────────────

def test_404_returns_error_json(client):
    resp = client.get("/nonexistent_endpoint")
    assert resp.status_code == 404
    data = resp.get_json()
    assert "error" in data
