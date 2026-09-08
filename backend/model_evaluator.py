"""
model_evaluator.py — Model Evaluation & Comparison
====================================================
Demonstrates ML Engineering skills:
  - Standard regression metrics: RMSE, MAE, MAPE, R²
  - LSTM backtesting without retraining
  - Confidence interval generation for LSTM (volatility-based)
  - Side-by-side model comparison with automatic winner selection
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from backend.arima_model import _compute_metrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Unified evaluation and comparison framework for epidemic forecasting models.

    Static class — all methods are class/static-level for easy import.

    Usage:
        metrics  = ModelEvaluator.compute_metrics(actuals, preds)
        backtest = ModelEvaluator.backtest_lstm(model, series, scaler)
        ci       = ModelEvaluator.lstm_confidence_interval(predictions)
        report   = ModelEvaluator.compare_models(lstm_result, arima_result)
    """

    # ── Core Metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute standard regression metrics.

        Returns:
            dict: {"rmse": float, "mae": float, "mape": float|None, "r2": float|None}
        """
        return _compute_metrics(actuals, predictions)

    # ── LSTM Backtest ─────────────────────────────────────────────────────────

    @staticmethod
    def backtest_lstm(
        model,
        series: pd.Series,
        scaler,
        test_steps: int = 30,
        window: int = 30,
    ) -> Dict[str, Any]:
        """
        Walk-forward backtest of the NumPy LSTM model on held-out data.

        Uses the window immediately preceding the test set as seed input,
        then auto-regressively generates `test_steps` predictions.

        Args:
            model:      LSTMPredictor instance.
            series:     Full historical daily-cases series.
            scaler:     Fitted MinMaxScaler (must already be fit on full data).
            test_steps: Number of end-period days to evaluate.
            window:     LSTM input window size (must match training).

        Returns:
            dict with: predictions, actuals, metrics (RMSE/MAE/MAPE/R²)
        """
        min_required = test_steps + window + 10
        if len(series) < min_required:
            logger.warning(
                "Not enough data for LSTM backtest (need %d, have %d)",
                min_required, len(series)
            )
            return {
                "predictions": [],
                "actuals": [],
                "metrics": {"rmse": None, "mae": None, "mape": None, "r2": None}
            }

        try:
            values = series.values.reshape(-1, 1)
            scaled = scaler.transform(values)

            # Seed: window immediately before the test period
            seed_end   = len(scaled) - test_steps
            seed_start = seed_end - window
            current_input = (
                scaled[seed_start:seed_end]
                .reshape(1, window, 1)
                .astype(np.float32)
            )

            predictions = []
            for _ in range(test_steps):
                pred     = model.predict(current_input)
                pred_val = float(pred[0][0])
                predictions.append(pred_val)
                current_input = np.append(
                    current_input[:, 1:, :],
                    np.array([[[pred_val]]], dtype=np.float32),
                    axis=1,
                )

            preds_rescaled = (
                scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                .flatten()
            )
            preds_rescaled = np.maximum(0, preds_rescaled)
            actuals        = series.values[-test_steps:]

            metrics = _compute_metrics(actuals, preds_rescaled)

            return {
                "predictions": [round(p, 2) for p in preds_rescaled.tolist()],
                "actuals":     [round(a, 2) for a in actuals.tolist()],
                "metrics":     metrics,
            }

        except Exception as e:
            logger.error("LSTM backtest error: %s", e)
            return {
                "predictions": [],
                "actuals": [],
                "metrics": {"rmse": None, "mae": None, "mape": None, "r2": None},
                "error": str(e),
            }

    # ── LSTM Confidence Intervals ─────────────────────────────────────────────

    @staticmethod
    def lstm_confidence_interval(
        predictions: List[float],
        base_uncertainty: float = 0.12,
    ) -> Dict[str, List[float]]:
        """
        Approximate confidence bands for LSTM point predictions.

        Uncertainty grows linearly with forecast horizon since LSTM
        has no built-in probabilistic output. Uses a volatility fraction
        that widens with each step.

        Args:
            predictions:      List of point forecast values.
            base_uncertainty: Fractional uncertainty at step 1 (default 12%).

        Returns:
            dict: {"lower_ci": [...], "upper_ci": [...]}
        """
        lower_ci, upper_ci = [], []
        for i, p in enumerate(predictions):
            # Uncertainty grows by 10% per additional step
            margin = p * base_uncertainty * (1.0 + i * 0.10)
            lower_ci.append(round(max(0.0, p - margin), 2))
            upper_ci.append(round(p + margin, 2))
        return {"lower_ci": lower_ci, "upper_ci": upper_ci}

    # ── Model Comparison ──────────────────────────────────────────────────────

    @staticmethod
    def compare_models(
        lstm_result:  Dict[str, Any],
        arima_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a structured side-by-side comparison of LSTM vs ARIMA.

        For RMSE, MAE, MAPE: lower is better.
        For R²:              higher is better.

        Returns a report including per-metric winners and an overall
        recommendation based on majority vote.
        """
        lstm_m  = lstm_result.get("metrics", {})
        arima_m = arima_result.get("metrics", {})

        def winner(key: str, lower_is_better: bool = True) -> str:
            l_val = lstm_m.get(key)
            a_val = arima_m.get(key)
            if l_val is None and a_val is None:
                return "N/A"
            if l_val is None:
                return "ARIMA"
            if a_val is None:
                return "LSTM"
            if lower_is_better:
                return "LSTM" if l_val <= a_val else "ARIMA"
            else:
                return "LSTM" if l_val >= a_val else "ARIMA"

        metric_winners = {
            "rmse": winner("rmse", lower_is_better=True),
            "mae":  winner("mae",  lower_is_better=True),
            "mape": winner("mape", lower_is_better=True),
            "r2":   winner("r2",   lower_is_better=False),
        }

        # Majority vote across RMSE, MAE, MAPE (ignore R² if None)
        votes = [v for v in metric_winners.values() if v in ("LSTM", "ARIMA")]
        recommendation = (
            "LSTM" if votes.count("LSTM") >= votes.count("ARIMA") else "ARIMA"
        )

        return {
            "lstm":           lstm_m,
            "arima":          arima_m,
            "metric_winners": metric_winners,
            "recommendation": recommendation,
            "note": (
                "Recommendation based on majority vote across RMSE, MAE, MAPE backtest metrics."
            ),
        }
