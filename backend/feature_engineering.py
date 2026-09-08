"""
feature_engineering.py — Feature Engineering Pipeline
=======================================================
Demonstrates ML Engineering skills:
  - Multi-scale temporal features (rolling averages at 7/14/30 days)
  - Lag features for autoregressive modelling
  - Growth rate and volatility metrics
  - Epidemiological indicators (doubling time, epidemic phase)
  - Seasonality signals (week-of-year, day-of-week)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class FeatureEngineer:
    """
    Builds a rich feature set from raw epidemic time-series data.

    Usage:
        fe   = FeatureEngineer()
        df_f = fe.build_features(df)          # Full feature DataFrame
        kpis = fe.get_feature_summary(df_f)   # Latest KPI snapshot
    """

    # ── Epidemic phase thresholds ─────────────────────────────────────────────
    PHASES = {
        "Exponential Growth": lambda g, c: g > 0.20 and c > 10,
        "Rising":             lambda g, c: g > 0.05 and c > 10,
        "Plateau":            lambda g, c: -0.05 <= g <= 0.05 and c > 10,
        "Declining":          lambda g, c: g < -0.05 and c > 10,
        "Rapid Decline":      lambda g, c: g < -0.20 and c > 10,
        "Low Activity":       lambda g, c: c <= 10,
    }

    PHASE_COLORS = {
        "Exponential Growth": "#ef4444",
        "Rising":             "#f97316",
        "Plateau":            "#eab308",
        "Declining":          "#22c55e",
        "Rapid Decline":      "#10b981",
        "Low Activity":       "#6366f1",
    }

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def build_features(df: pd.DataFrame, target_col: str = "cases") -> pd.DataFrame:
        """
        Build the full feature matrix from a time-series DataFrame.

        Args:
            df:         DataFrame with DatetimeIndex and a `cases` column.
            target_col: Column name to engineer features on.

        Returns:
            DataFrame with all original columns plus engineered features.
        """
        df = df.copy()
        s  = df[target_col]

        # ── Rolling averages (multi-scale) ────────────────────────────────────
        df["ma_7"]  = s.rolling(7,  min_periods=1).mean().round(2)
        df["ma_14"] = s.rolling(14, min_periods=1).mean().round(2)
        df["ma_30"] = s.rolling(30, min_periods=1).mean().round(2)

        # ── Rolling volatility (standard deviation) ───────────────────────────
        df["std_7"]  = s.rolling(7,  min_periods=2).std().round(2)
        df["std_14"] = s.rolling(14, min_periods=2).std().round(2)

        # ── Growth rates (week-over-week and fortnight-over-fortnight) ─────────
        df["growth_rate_7d"]  = (
            s.pct_change(7)
            .replace([np.inf, -np.inf], 0)
            .round(4)
        )
        df["growth_rate_14d"] = (
            s.pct_change(14)
            .replace([np.inf, -np.inf], 0)
            .round(4)
        )

        # ── Lag features (for autoregressive and supervised ML models) ─────────
        for lag in [1, 7, 14, 21]:
            df[f"lag_{lag}"] = s.shift(lag)

        # ── Epidemiological: Doubling time ─────────────────────────────────────
        # Derived from compound growth: T_d = log(2) / log(1 + r)
        df["doubling_time"] = np.where(
            df["growth_rate_7d"] > 0,
            np.log(2) / np.log1p(df["growth_rate_7d"].clip(lower=1e-6)),
            np.nan,
        ).round(1)

        # ── Epidemic phase classification ─────────────────────────────────────
        df["epidemic_phase"] = [
            FeatureEngineer._classify_phase(row["growth_rate_7d"], row[target_col])
            for _, row in df.iterrows()
        ]

        # ── Seasonality signals ───────────────────────────────────────────────
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["day_of_week"]  = df.index.dayofweek  # 0=Monday, 6=Sunday

        # ── Fill NaN from lags/rolling at edges ───────────────────────────────
        df = df.fillna(0)
        return df

    @staticmethod
    def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a snapshot of the latest feature values for dashboard KPIs.

        Returns a flat dict suitable for JSON serialisation.
        """
        latest = df.iloc[-1]
        prev_7 = df.iloc[-8] if len(df) > 8 else df.iloc[0]

        def safe_float(val, fallback=0.0) -> float:
            try:
                return round(float(val), 2)
            except (TypeError, ValueError):
                return fallback

        return {
            "current_cases":    safe_float(latest.get("cases", 0)),
            "ma_7":             safe_float(latest.get("ma_7", 0)),
            "ma_14":            safe_float(latest.get("ma_14", 0)),
            "ma_30":            safe_float(latest.get("ma_30", 0)),
            "growth_rate_7d":   safe_float(latest.get("growth_rate_7d", 0) * 100),  # as %
            "growth_rate_14d":  safe_float(latest.get("growth_rate_14d", 0) * 100),
            "volatility_7d":    safe_float(latest.get("std_7", 0)),
            "doubling_time":    safe_float(latest.get("doubling_time", 0)) or None,
            "epidemic_phase":   str(latest.get("epidemic_phase", "Unknown")),
            "cases_7d_ago":     safe_float(prev_7.get("cases", 0)),
        }

    @staticmethod
    def _classify_phase(growth_rate: float, cases: float) -> str:
        """Classify current epidemic phase based on growth rate and case count."""
        for phase, condition in FeatureEngineer.PHASES.items():
            try:
                if condition(float(growth_rate), float(cases)):
                    return phase
            except Exception:
                pass
        return "Unknown"
