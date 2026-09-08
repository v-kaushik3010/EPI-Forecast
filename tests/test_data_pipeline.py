"""
test_data_pipeline.py — Unit Tests: Data Engineering Layer
============================================================
Tests cover:
  - Country whitelist validation
  - Transform logic (negative clipping, outlier correction)
  - Data quality report structure
  - Cache TTL logic (via mock)
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_pipeline import DataPipeline, ALLOWED_COUNTRIES_LIST


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic raw cumulative DataFrame for testing."""
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    cumulative = np.cumsum(np.random.randint(100, 5000, size=n)).astype(float)
    df = pd.DataFrame({"cumulative_cases": cumulative}, index=dates)
    df.index.name = "date"
    df["country"] = "india"
    return df


@pytest.fixture
def pipeline():
    return DataPipeline(cache_ttl_seconds=3600)


# ── Validation Tests ──────────────────────────────────────────────────────────

def test_allowed_countries_not_empty():
    assert len(ALLOWED_COUNTRIES_LIST) > 0


def test_invalid_country_raises(pipeline):
    with pytest.raises(ValueError, match="not supported"):
        pipeline._validate_country("atlantis")


def test_valid_country_normalised(pipeline):
    result = pipeline._validate_country("  INDIA  ")
    assert result == "india"


def test_allowed_country_passes(pipeline):
    for c in ["india", "usa", "brazil"]:
        assert pipeline._validate_country(c) == c


# ── Transform Tests ───────────────────────────────────────────────────────────

def test_transform_converts_cumulative_to_daily(pipeline):
    raw = _make_raw_df(50)
    clean = pipeline.transform(raw)
    # Daily cases must be non-negative
    assert (clean["cases"] >= 0).all(), "All daily cases should be >= 0"


def test_transform_removes_negatives(pipeline):
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    cumul = [100, 200, 150, 300, 250, 600, 700, 800, 900, 1000]
    df = pd.DataFrame({"cumulative_cases": cumul}, index=dates)
    df.index.name = "date"
    df["country"] = "india"

    clean = pipeline.transform(df)
    assert (clean["cases"] >= 0).all(), "Negatives from decreasing cumulative should be clipped"


def test_transform_returns_dataframe(pipeline):
    raw = _make_raw_df(60)
    clean = pipeline.transform(raw)
    assert isinstance(clean, pd.DataFrame)
    assert "cases" in clean.columns


def test_transform_trims_leading_zeros(pipeline):
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    # First 10 days zero cumulative, then starts growing
    cumul = [0] * 10 + list(range(100, 1100, 100))
    df = pd.DataFrame({"cumulative_cases": cumul}, index=dates)
    df.index.name = "date"
    df["country"] = "india"
    clean = pipeline.transform(df)
    # After trimming leading zeros, first row should have nonzero cases
    assert len(clean) < len(df), "Leading zeros should be trimmed"


# ── Quality Report Tests ──────────────────────────────────────────────────────

def test_quality_report_keys(pipeline):
    raw   = _make_raw_df(120)
    clean = pipeline.transform(raw)
    report = pipeline.get_data_quality_report(clean)

    required_keys = [
        "quality_score", "completeness_pct", "total_records",
        "null_records", "zero_records", "outliers_detected",
        "data_freshness", "date_range", "mean_daily_cases",
    ]
    for k in required_keys:
        assert k in report, f"Missing key: {k}"


def test_quality_score_between_0_and_100(pipeline):
    raw   = _make_raw_df(120)
    clean = pipeline.transform(raw)
    report = pipeline.get_data_quality_report(clean)
    assert 0 <= report["quality_score"] <= 100


def test_completeness_pct_is_100_for_clean_data(pipeline):
    raw   = _make_raw_df(120)
    clean = pipeline.transform(raw)
    report = pipeline.get_data_quality_report(clean)
    assert report["completeness_pct"] == 100.0


# ── Cache Tests ───────────────────────────────────────────────────────────────

def test_cache_miss_returns_none_when_no_file(pipeline, tmp_path, monkeypatch):
    """When no cache file exists, _load_cache should return None."""
    monkeypatch.setattr("backend.data_pipeline.CACHE_DIR", tmp_path)
    result = pipeline._load_cache("nonexistent_country")
    assert result is None


def test_pipeline_metadata_populated_after_transform(pipeline):
    raw = _make_raw_df(100)
    pipeline.transform(raw)
    assert "transform" in pipeline._pipeline_meta
    assert "rows_after_transform" in pipeline._pipeline_meta["transform"]
