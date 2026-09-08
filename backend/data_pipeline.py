"""
data_pipeline.py — ETL Pipeline for Epidemic Data
===================================================
Demonstrates Data Engineering skills:
  - Extract: REST API ingestion with retry & timeout handling
  - Transform: Data cleaning, outlier correction, schema validation
  - Load: Disk-based JSON cache with configurable TTL
  - Quality: Automated data quality reporting
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_COUNTRIES: set = {
    "india", "usa", "brazil", "uk", "russia", "france", "germany",
    "italy", "spain", "japan", "china", "australia", "canada",
    "mexico", "indonesia", "turkey", "argentina", "iran", "colombia",
    "ukraine", "netherlands", "sweden", "switzerland", "belgium",
    "austria", "portugal", "czechia", "israel", "singapore",
}

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_SH_BASE = "https://disease.sh/v3/covid-19"


# ── DataPipeline Class ────────────────────────────────────────────────────────

class DataPipeline:
    """
    Full ETL pipeline for epidemic time-series data.

    Usage (one-liner):
        df = DataPipeline().run("india")

    Stages:
        1. extract()  — Fetch raw data from disease.sh with retry logic
        2. transform() — Convert cumulative → daily, clean outliers, smooth
        3. load()      — Persist to JSON cache with TTL metadata
    """

    def __init__(self, cache_ttl_seconds: int = 3600):
        self.cache_ttl = cache_ttl_seconds
        self._pipeline_meta: Dict[str, Any] = {}

    # ── Extract ───────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def extract(self, country: str) -> pd.DataFrame:
        """
        Extract raw COVID-19 cumulative time-series from disease.sh API.

        - Validates country against whitelist
        - Returns cached data if fresh
        - Retries up to 3 times on network errors
        """
        country = self._validate_country(country)

        # Cache hit?
        cached = self._load_cache(country)
        if cached is not None:
            logger.info("Cache HIT for country=%s", country)
            # Still populate extract meta so it reflects this country (not a stale previous run)
            self._pipeline_meta["extract"] = {
                "source": "cache",
                "country": country,
                "rows_extracted": len(cached),
                "date_range": f"{cached.index.min().date()} → {cached.index.max().date()}",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            return cached

        url = f"{DISEASE_SH_BASE}/historical/{country}?lastdays=all"
        logger.info("Fetching from disease.sh: country=%s", country)

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw = response.json()

        cases_dict = raw["timeline"]["cases"]
        df = pd.DataFrame(
            list(cases_dict.items()),
            columns=["date", "cumulative_cases"]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["country"] = country

        self._pipeline_meta["extract"] = {
            "source": "disease.sh",
            "url": url,
            "country": country,
            "rows_extracted": len(df),
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Extracted %d rows for %s", len(df), country)
        return df

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw cumulative data into cleaned daily cases.

        Steps:
          1. Cumulative → daily diff
          2. Clip negative values (reporting corrections)
          3. Detect & correct outliers (IQR method)
          4. Apply 7-day rolling smoothing
          5. Trim leading zero-rows (pre-pandemic)
        """
        df = df.copy()

        # Step 1: Cumulative → Daily
        df["daily_cases"] = df["cumulative_cases"].diff().fillna(0)

        # Step 2: Clip negatives
        neg_count = int((df["daily_cases"] < 0).sum())
        df["daily_cases"] = df["daily_cases"].clip(lower=0)

        # Step 3: IQR-based outlier detection & correction
        q1 = df["daily_cases"].quantile(0.25)
        q3 = df["daily_cases"].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        df["is_outlier"] = df["daily_cases"] > upper_fence
        outlier_count = int(df["is_outlier"].sum())
        # Replace outliers with rolling median
        df["daily_cases_raw"] = df["daily_cases"].copy()
        rolling_median = df["daily_cases"].rolling(7, center=True, min_periods=1).median()
        df.loc[df["is_outlier"], "daily_cases"] = rolling_median[df["is_outlier"]]

        # Step 4: 7-day rolling mean (smooth weekend/batch reporting effects)
        df["cases"] = (
            df["daily_cases"]
            .rolling(window=7, min_periods=1)
            .mean()
            .round(2)
        )

        # Step 5: Trim pre-pandemic leading zeros
        first_nonzero_idx = df["cases"].ne(0).idxmax()
        df = df.loc[first_nonzero_idx:]

        self._pipeline_meta["transform"] = {
            "rows_after_transform": len(df),
            "negative_values_clipped": neg_count,
            "outliers_corrected": outlier_count,
            "null_count": int(df["cases"].isnull().sum()),
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
        }
        logger.info(
            "Transformed: %d rows | negatives=%d | outliers=%d",
            len(df), neg_count, outlier_count
        )
        return df

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame, country: str) -> None:
        """
        Persist cleaned data to JSON cache with TTL metadata.

        In production, replace with Parquet on object storage (S3/GCS).
        """
        cache_path = CACHE_DIR / f"{country}.json"
        meta_path  = CACHE_DIR / f"{country}_meta.json"

        # Serialise DataFrame to JSON
        records = df.reset_index().copy()
        records["date"] = records["date"].astype(str)
        records.to_json(cache_path, orient="records")

        meta = {
            "country": country,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": self.cache_ttl,
            "rows": len(df),
            "pipeline_metadata": self._pipeline_meta,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Cached %d rows → %s", len(df), cache_path)

    # ── Run (full pipeline) ───────────────────────────────────────────────────

    def run(self, country: str) -> pd.DataFrame:
        """Execute full ETL: extract → transform → load → return DataFrame."""
        # Reset metadata at the start of every run so no cross-country bleed
        self._pipeline_meta = {}
        logger.info("[PIPELINE] Starting for country=%s", country)
        raw_df   = self.extract(country)
        clean_df = self.transform(raw_df)
        self.load(clean_df, country)
        logger.info("[PIPELINE] Done. %d rows ready.", len(clean_df))
        return clean_df

    # ── Quality Report ────────────────────────────────────────────────────────

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a structured data quality report.

        Covers: completeness, freshness, outliers, distribution stats.
        Useful for a Data Engineer or Analyst reviewing pipeline health.
        """
        total      = len(df)
        complete   = int(df["cases"].notna().sum())
        null_count = total - complete
        zero_count = int((df["cases"] == 0).sum())
        outlier_count = int(df.get("is_outlier", pd.Series(dtype=bool)).sum())

        last_date = df.index.max()
        days_stale = (datetime.now(timezone.utc).date() - last_date.date()).days

        completeness_pct = round(complete / total * 100, 2) if total else 0
        quality_score = max(
            0,
            round(
                completeness_pct
                - min(outlier_count / max(total, 1) * 100, 20)
                - min(days_stale * 2, 20),
                1,
            ),
        )

        return {
            "quality_score": quality_score,
            "completeness_pct": completeness_pct,
            "total_records": total,
            "null_records": null_count,
            "zero_records": zero_count,
            "outliers_detected": outlier_count,
            "days_since_last_update": days_stale,
            "data_freshness": "STALE (>2 days)" if days_stale > 2 else "FRESH",
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            "mean_daily_cases": round(float(df["cases"].mean()), 2),
            "max_daily_cases":  round(float(df["cases"].max()), 2),
            "std_daily_cases":  round(float(df["cases"].std()), 2),
            "pipeline_metadata": self._pipeline_meta,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate_country(self, country: str) -> str:
        country = country.lower().strip()
        if country not in ALLOWED_COUNTRIES:
            raise ValueError(
                f"Country '{country}' not supported. "
                f"Allowed: {sorted(ALLOWED_COUNTRIES)}"
            )
        return country

    def _load_cache(self, country: str) -> Optional[pd.DataFrame]:
        """Load from JSON cache if TTL has not expired."""
        cache_path = CACHE_DIR / f"{country}.json"
        meta_path  = CACHE_DIR / f"{country}_meta.json"

        if not cache_path.exists() or not meta_path.exists():
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        cached_at   = datetime.fromisoformat(meta["cached_at"])
        age_seconds = (datetime.now(timezone.utc) - cached_at).total_seconds()

        if age_seconds > meta["ttl_seconds"]:
            logger.info("Cache EXPIRED for %s (age=%.0fs)", country, age_seconds)
            return None

        df = pd.read_json(cache_path, orient="records")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df


# ── Module-level helpers ──────────────────────────────────────────────────────

ALLOWED_COUNTRIES_LIST = sorted(ALLOWED_COUNTRIES)
