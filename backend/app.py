"""
app.py — Epi-Forecast AI REST API  (v2.0)
==========================================
Demonstrates Backend / Data Engineering / ML Engineering skills:

Endpoints:
    GET /                 Root — service info + uptime
    GET /health           Health check — model & service status
    GET /countries        Available countries list
    GET /predict          LSTM or ARIMA forecast with confidence intervals
    GET /compare          Side-by-side LSTM vs ARIMA comparison + metrics
    GET /data/stats       Statistical summary of historical data
    GET /data/quality     Data quality report (ETL pipeline output)
    GET /data/features    Feature-engineered dataset (last 30 days)
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.preprocessing import MinMaxScaler

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_pipeline import DataPipeline, ALLOWED_COUNTRIES_LIST
from backend.feature_engineering import FeatureEngineer
from backend.model_numpy import LSTMPredictor
from backend.arima_model import ARIMAForecaster
from backend.model_evaluator import ModelEvaluator
from backend.preprocess import fit_scaler, descriptive_stats

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App & Extensions ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per minute"],
    storage_uri="memory://",
)

_START_TIME = time.time()
MAX_FORECAST_DAYS = int(os.environ.get("MAX_FORECAST_DAYS", 30))

# ── Global Resources (loaded once at startup) ─────────────────────────────────
try:
    lstm_model = LSTMPredictor()
    logger.info("LSTM model loaded successfully (NumPy backend)")
except Exception as exc:
    lstm_model = None
    logger.error("LSTM model load failed: %s", exc)

pipeline         = DataPipeline(cache_ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", 3600)))
feature_engineer = FeatureEngineer()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _validate(country: str, days: int = 7):
    """Validate and sanitise request parameters. Returns (country, error_msg)."""
    country = country.lower().strip()
    if country not in ALLOWED_COUNTRIES_LIST:
        return None, (
            f"Country '{country}' is not supported. "
            f"Use /countries to see the allowed list."
        )
    if not (1 <= days <= MAX_FORECAST_DAYS):
        return None, f"`days` must be between 1 and {MAX_FORECAST_DAYS}."
    return country, None


def _lstm_forecast(country: str, days: int, scaler=None):
    """
    Run LSTM inference for `days` steps.

    Returns (predictions_list, confidence_interval_dict, clean_df, scaler)
    """
    df     = pipeline.run(country)
    values = df["cases"].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(values)

    window         = 30
    last_w_scaled  = scaler.transform(values[-window:])
    current_input  = last_w_scaled.reshape(1, window, 1).astype(np.float32)

    raw_preds = []
    for _ in range(days):
        pred     = lstm_model.predict(current_input)
        pred_val = float(pred[0][0])
        raw_preds.append(pred_val)
        current_input = np.append(
            current_input[:, 1:, :],
            np.array([[[pred_val]]], dtype=np.float32),
            axis=1,
        )

    rescaled = (
        scaler.inverse_transform(np.array(raw_preds).reshape(-1, 1))
        .flatten()
    )
    predictions = [round(max(0.0, v), 2) for v in rescaled.tolist()]
    ci          = ModelEvaluator.lstm_confidence_interval(predictions)
    return predictions, ci, df, scaler


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg, "status": code}), code


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

# ── Root ──────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    """Service information and uptime."""
    return jsonify({
        "service":        "Epi-Forecast AI API",
        "version":        "2.0.0",
        "description":    "Epidemic forecasting API for COVID-19 using LSTM and ARIMA models.",
        "uptime_seconds": round(time.time() - _START_TIME),
        "model_status": {
            "lstm":  "loaded" if lstm_model else "error",
            "arima": "available (statsmodels)",
        },
        "endpoints": {
            "GET /health":          "Service & model health check",
            "GET /countries":       "Supported countries list",
            "GET /predict":         "LSTM or ARIMA forecast with confidence intervals",
            "GET /compare":         "Side-by-side LSTM vs ARIMA comparison",
            "GET /data/stats":      "Statistical summary of historical data",
            "GET /data/quality":    "ETL data quality report",
            "GET /data/features":   "Feature-engineered view (last 30 days)",
        },
    })


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Lightweight health check — suitable for Render's healthCheckPath."""
    return jsonify({
        "status":     "ok",
        "lstm":       "loaded" if lstm_model else "error",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })


# ── Countries ─────────────────────────────────────────────────────────────────

@app.route("/countries")
def countries():
    """Return the list of supported countries."""
    return jsonify({
        "countries": ALLOWED_COUNTRIES_LIST,
        "count":     len(ALLOWED_COUNTRIES_LIST),
    })


# ── Predict ───────────────────────────────────────────────────────────────────

@app.route("/predict")
@limiter.limit("120 per minute")
def predict():
    """
    Generate an epidemic forecast.

    Query params:
        country (str):  Country name (default: india)
        days    (int):  Forecast horizon 1–30 (default: 7)
        model   (str):  'lstm' or 'arima' (default: lstm)

    Returns:
        JSON with predictions, confidence intervals, and past 30 days.
    """
    country_raw = request.args.get("country", "india")
    days        = int(request.args.get("days",    7))
    model_type  = request.args.get("model",   "lstm").lower()

    country, err = _validate(country_raw, days)
    if err:
        return _error(err)

    try:
        if model_type == "arima":
            df         = pipeline.run(country)
            forecaster = ARIMAForecaster()
            forecaster.fit(df["cases"])
            result     = forecaster.predict(steps=days)
            return jsonify({
                "country":      country,
                "model":        "arima",
                "days":         days,
                "predictions":  result["predictions"],
                "lower_ci":     result["lower_ci"],
                "upper_ci":     result["upper_ci"],
                "past_30_days": df["cases"].tail(30).round(2).tolist(),
            })

        # Default: LSTM
        if lstm_model is None:
            return _error("LSTM model is not loaded. Check server logs.", 500)

        preds, ci, df, _ = _lstm_forecast(country, days)
        return jsonify({
            "country":      country,
            "model":        "lstm",
            "days":         days,
            "predictions":  preds,
            "lower_ci":     ci["lower_ci"],
            "upper_ci":     ci["upper_ci"],
            "past_30_days": df["cases"].tail(30).round(2).tolist(),
        })

    except Exception as exc:
        logger.exception("Predict error for country=%s model=%s", country, model_type)
        return _error(str(exc), 500)


# ── Compare ───────────────────────────────────────────────────────────────────

@app.route("/compare")
@limiter.limit("30 per minute")
def compare():
    """
    Run both LSTM and ARIMA, backtest each, and return a comparison report.

    Query params:
        country (str): Country name (default: india)
        days    (int): Forecast horizon (default: 7)

    Returns:
        JSON with both forecasts, backtest metrics, and winner recommendation.
    """
    country_raw = request.args.get("country", "india")
    days        = int(request.args.get("days", 7))

    country, err = _validate(country_raw, days)
    if err:
        return _error(err)

    try:
        df               = pipeline.run(country)
        scaled, scaler   = fit_scaler(df["cases"])
        test_steps       = min(30, max(7, len(df) // 10))

        # ── LSTM ─────────────────────────────────────────────────────────────
        lstm_preds, lstm_ci, _, _ = _lstm_forecast(country, days, scaler=scaler)
        lstm_backtest = (
            ModelEvaluator.backtest_lstm(lstm_model, df["cases"], scaler, test_steps)
            if lstm_model else {"metrics": {}, "predictions": [], "actuals": []}
        )

        # ── ARIMA ────────────────────────────────────────────────────────────
        forecaster   = ARIMAForecaster()
        forecaster.fit(df["cases"])
        arima_result = forecaster.predict(steps=days)
        arima_metrics = forecaster.evaluate(df["cases"], test_steps=test_steps)

        # ── Comparison ───────────────────────────────────────────────────────
        comparison = ModelEvaluator.compare_models(
            lstm_result  = {"metrics": lstm_backtest.get("metrics", {})},
            arima_result = {"metrics": arima_metrics},
        )

        return jsonify({
            "country": country,
            "days":    days,
            "lstm": {
                "predictions":       lstm_preds,
                "lower_ci":          lstm_ci["lower_ci"],
                "upper_ci":          lstm_ci["upper_ci"],
                "backtest_metrics":  lstm_backtest.get("metrics", {}),
                "backtest_actuals":  lstm_backtest.get("actuals", []),
                "backtest_preds":    lstm_backtest.get("predictions", []),
            },
            "arima": {
                "predictions":       arima_result["predictions"],
                "lower_ci":          arima_result["lower_ci"],
                "upper_ci":          arima_result["upper_ci"],
                "backtest_metrics":  arima_metrics,
            },
            "comparison":   comparison,
            "past_30_days": df["cases"].tail(30).round(2).tolist(),
        })

    except Exception as exc:
        logger.exception("Compare error for country=%s", country)
        return _error(str(exc), 500)


# ── Data: Stats ───────────────────────────────────────────────────────────────

@app.route("/data/stats")
def data_stats():
    """
    Descriptive statistics for a country's historical case series.

    Returns mean, median, std, min, max, percentiles, skewness, kurtosis.
    """
    country_raw = request.args.get("country", "india")
    country, err = _validate(country_raw)
    if err:
        return _error(err)

    try:
        df    = pipeline.run(country)
        stats = descriptive_stats(df["cases"])
        return jsonify({
            "country":    country,
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            "statistics": stats,
            "total_cumulative_cases": (
                int(df["cumulative_cases"].max())
                if "cumulative_cases" in df.columns else None
            ),
        })
    except Exception as exc:
        logger.exception("Data stats error for country=%s", country)
        return _error(str(exc), 500)


# ── Data: Quality ─────────────────────────────────────────────────────────────

@app.route("/data/quality")
def data_quality():
    """
    ETL data quality report for a country.

    Includes: completeness %, outlier count, freshness, quality score.
    """
    country_raw = request.args.get("country", "india")
    country, err = _validate(country_raw)
    if err:
        return _error(err)

    try:
        df     = pipeline.run(country)
        report = pipeline.get_data_quality_report(df)
        return jsonify({"country": country, "quality_report": report})
    except Exception as exc:
        logger.exception("Data quality error for country=%s", country)
        return _error(str(exc), 500)


# ── Data: Features ────────────────────────────────────────────────────────────

@app.route("/data/features")
def data_features():
    """
    Return the feature-engineered dataset for a country (last 30 days).

    Includes: rolling averages, growth rates, lag features, epidemic phase.
    """
    country_raw = request.args.get("country", "india")
    country, err = _validate(country_raw)
    if err:
        return _error(err)

    try:
        df      = pipeline.run(country)
        feat_df = FeatureEngineer.build_features(df)
        summary = FeatureEngineer.get_feature_summary(feat_df)

        recent = feat_df.tail(60).reset_index().copy()
        recent["date"] = recent["date"].astype(str)
        cols = ["date", "cases", "ma_7", "ma_14", "ma_30",
                "growth_rate_7d", "std_7", "doubling_time", "epidemic_phase"]
        cols = [c for c in cols if c in recent.columns]

        return jsonify({
            "country":         country,
            "feature_summary": summary,
            "recent_features": recent[cols].to_dict(orient="records"),
        })
    except Exception as exc:
        logger.exception("Data features error for country=%s", country)
        return _error(str(exc), 500)


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return _error("Endpoint not found. Visit / for the API map.", 404)


@app.errorhandler(429)
def rate_limited(e):
    return _error("Rate limit exceeded. Please slow down your requests.", 429)


@app.errorhandler(500)
def server_error(e):
    return _error("Internal server error. Check the server logs.", 500)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Epi-Forecast API on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
