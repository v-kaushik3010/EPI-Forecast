"""
preprocess.py — Data Preprocessing with Schema Validation
===========================================================
Demonstrates Data Engineering + ML Engineering skills:
  - Pydantic schema validation for incoming API parameters
  - Train / validation / test split helper
  - Sequence generator for LSTM input
  - Summary statistics for Data Analyst consumption
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Schema for /predict endpoint request parameters."""
    country: str = Field(default="india", description="Country name (lowercase)")
    days:    int  = Field(default=7, ge=1, le=30, description="Forecast horizon (1–30)")
    model:   str  = Field(default="lstm", description="Model type: lstm | arima")

    @field_validator("country")
    @classmethod
    def country_must_be_lowercase(cls, v):
        return v.lower().strip()

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v):
        v = v.lower().strip()
        if v not in ("lstm", "arima"):
            raise ValueError("model must be 'lstm' or 'arima'")
        return v


class CompareRequest(BaseModel):
    """Schema for /compare endpoint request parameters."""
    country: str = Field(default="india")
    days:    int  = Field(default=7, ge=1, le=30)

    @field_validator("country")
    @classmethod
    def country_must_be_lowercase(cls, v):
        return v.lower().strip()


# ── Preprocessing Functions ───────────────────────────────────────────────────

def create_sequences(
    scaled_values: np.ndarray,
    window: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning sequences from a scaled 1-D array.

    Converts a time series into (X, y) pairs where:
      X[i] = values[i : i + window]   (shape: window,)
      y[i] = values[i + window]        (scalar)

    Args:
        scaled_values: 1-D numpy array of normalised values.
        window:        Look-back window size.

    Returns:
        X: shape (n_samples, window, 1)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(window, len(scaled_values)):
        X.append(scaled_values[i - window:i, 0])
        y.append(scaled_values[i, 0])
    X_arr = np.array(X).reshape(-1, window, 1)
    y_arr = np.array(y)
    return X_arr, y_arr


def train_val_test_split(
    df: pd.DataFrame,
    val_frac:  float = 0.10,
    test_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split for time-series data.

    Args:
        df:        DataFrame sorted by date.
        val_frac:  Fraction of data for validation.
        test_frac: Fraction of data for test.

    Returns:
        train_df, val_df, test_df
    """
    n       = len(df)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]
    return train, val, test


def fit_scaler(series: pd.Series) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit a MinMaxScaler on a Pandas Series and return scaled values + scaler.

    Args:
        series: 1-D Pandas Series of case counts.

    Returns:
        (scaled_array of shape (n, 1), fitted MinMaxScaler)
    """
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute descriptive statistics for a case series.

    Used by the /data/stats API endpoint and the Data Explorer tab.
    """
    return {
        "count":  int(series.count()),
        "mean":   round(float(series.mean()),   2),
        "median": round(float(series.median()), 2),
        "std":    round(float(series.std()),    2),
        "min":    round(float(series.min()),    2),
        "max":    round(float(series.max()),    2),
        "p25":    round(float(series.quantile(0.25)), 2),
        "p75":    round(float(series.quantile(0.75)), 2),
        "skewness": round(float(series.skew()),  4),
        "kurtosis": round(float(series.kurtosis()), 4),
    }
