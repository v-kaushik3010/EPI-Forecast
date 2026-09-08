"""
arima_model.py — ARIMA Forecaster
===================================
Demonstrates ML Engineering skills:
  - Clean class design with fit / predict / evaluate methods
  - Native confidence intervals from statsmodels
  - Backtesting framework with standard metrics
  - Graceful fallback on fitting errors
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    ARIMA-based epidemic forecaster using statsmodels.

    Design Pattern:
        forecaster = ARIMAForecaster(order=(5, 1, 0))
        forecaster.fit(series)
        result = forecaster.predict(steps=7)
        metrics = forecaster.evaluate(series, test_steps=30)

    Attributes:
        order:       ARIMA (p, d, q) order tuple.
        _model_fit:  Fitted statsmodels ARIMAResults object.
    """

    DEFAULT_ORDER = (5, 1, 0)

    def __init__(self, order: tuple = None):
        self.order       = order or self.DEFAULT_ORDER
        self._model_fit  = None
        self._train_data: Optional[pd.Series] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, series: pd.Series, max_history: int = 365) -> "ARIMAForecaster":
        """
        Fit ARIMA model on a univariate time series.

        Args:
            series:      Pandas Series of daily case counts.
            max_history: Limit training window for speed (last N days).

        Returns:
            self (for method chaining)
        """
        from statsmodels.tsa.arima.model import ARIMA

        train = series.clip(lower=0).tail(max_history).copy()
        train = train.fillna(0)

        try:
            model = ARIMA(train, order=self.order)
            self._model_fit  = model.fit()
            self._train_data = train
            logger.info(
                "ARIMA%s fitted on %d observations (AIC=%.2f)",
                self.order, len(train), self._model_fit.aic
            )
        except Exception as e:
            logger.error("ARIMA fitting failed: %s", e)
            raise RuntimeError(f"ARIMA fit failed: {e}") from e

        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, steps: int = 7) -> Dict[str, Any]:
        """
        Generate point forecast with 90% confidence intervals.

        Args:
            steps: Number of future time steps to forecast.

        Returns:
            dict with keys: predictions, lower_ci, upper_ci
        """
        if self._model_fit is None:
            raise RuntimeError("Model not fitted — call fit() first.")

        forecast   = self._model_fit.get_forecast(steps=steps)
        point_vals = forecast.predicted_mean.values
        conf_int   = forecast.conf_int(alpha=0.10)  # 90% CI

        predictions = [round(max(0.0, v), 2) for v in point_vals.tolist()]
        lower_ci    = [round(max(0.0, v), 2) for v in conf_int.iloc[:, 0].tolist()]
        upper_ci    = [round(max(0.0, v), 2) for v in conf_int.iloc[:, 1].tolist()]

        return {
            "predictions": predictions,
            "lower_ci":    lower_ci,
            "upper_ci":    upper_ci,
            "model":       f"ARIMA{self.order}",
        }

    # ── Evaluate (Backtest) ───────────────────────────────────────────────────

    def evaluate(self, series: pd.Series, test_steps: int = 30) -> Dict[str, Any]:
        """
        Backtest the ARIMA model on held-out data.

        Splits series into train / test, fits on train, evaluates on test.

        Args:
            series:     Full historical series.
            test_steps: Number of end-period rows to use as test set.

        Returns:
            dict with RMSE, MAE, MAPE, R² metrics.
        """
        min_required = test_steps + 60
        if len(series) < min_required:
            logger.warning(
                "Not enough data for backtest (need %d, have %d)",
                min_required, len(series)
            )
            return {"rmse": None, "mae": None, "mape": None, "r2": None}

        from statsmodels.tsa.arima.model import ARIMA

        train = series.iloc[:-test_steps].clip(lower=0).tail(365)
        test  = series.iloc[-test_steps:].values

        try:
            model  = ARIMA(train, order=self.order)
            fit    = model.fit()
            preds  = np.maximum(0, fit.forecast(steps=test_steps).values)
            return _compute_metrics(test, preds)
        except Exception as e:
            logger.warning("ARIMA backtest failed: %s", e)
            return {"rmse": None, "mae": None, "mape": None, "r2": None}

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return statsmodels model summary string (for diagnostics)."""
        if self._model_fit is None:
            return "Model not fitted."
        return str(self._model_fit.summary())


# ── Shared metric helper (also used by ModelEvaluator) ───────────────────────

def _compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """Compute RMSE, MAE, MAPE, R² between actuals and predictions."""
    actuals     = np.array(actuals, dtype=float)
    predictions = np.array(predictions, dtype=float)

    mae  = float(np.mean(np.abs(actuals - predictions)))
    rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))

    # MAPE: only where actuals > 1 to avoid division by zero
    mask = actuals > 1
    mape = (
        float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)
        if mask.sum() > 0 else None
    )

    ss_res = float(np.sum((actuals - predictions) ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else None

    return {
        "rmse": round(rmse, 2),
        "mae":  round(mae,  2),
        "mape": round(mape, 2) if mape is not None else None,
        "r2":   round(r2,   4) if r2   is not None else None,
    }
