# EpiCast AI — Complete Project Documentation

> **Version 2.0** | Epidemic Forecast Platform for COVID-19 | LSTM + ARIMA  
> Built to showcase **Data Engineering**, **ML Engineering**, and **Data Analyst** skills  
> Deployable on [Render](https://render.com) (free tier)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Folder Structure](#3-folder-structure)
4. [Role Mapping — What Each Part Showcases](#4-role-mapping)
5. [Setup & Running Locally](#5-setup--running-locally)
6. [API Reference (8 Endpoints)](#6-api-reference)
7. [Dashboard Guide (4 Tabs)](#7-dashboard-guide)
8. [ML Models — LSTM & ARIMA](#8-ml-models)
9. [Data Engineering Pipeline](#9-data-engineering-pipeline)
10. [Feature Engineering](#10-feature-engineering)
11. [Model Evaluation & Comparison](#11-model-evaluation--comparison)
12. [Deploying to Render](#12-deploying-to-render)
13. [Running Tests](#13-running-tests)
14. [Environment Variables](#14-environment-variables)
15. [How to Extend This Project](#15-how-to-extend-this-project)

---

## 1. Project Overview

**EpiCast AI** is a full-stack epidemic forecasting platform that:

- Ingests live COVID-19 data from the [disease.sh](https://disease.sh) open API
- Runs **two forecasting models** (LSTM deep learning + ARIMA statistical)
- Provides a **REST API** (Flask, 8 endpoints) for programmatic access
- Delivers a **multi-tab interactive dashboard** (Streamlit + Plotly)
- Is fully deployable to **Render** (two services: backend + dashboard)

### Data Source

| Source | Type | Endpoint |
|--------|------|----------|
| `disease.sh` | COVID-19 historical time-series | `GET /v3/covid-19/historical/{country}?lastdays=all` |

> ⚠️ **Note**: Only COVID-19 data is available from this source. The platform focuses exclusively on COVID-19 cases. The system is designed to be extended to other diseases (flu, dengue) by plugging in additional data sources.

---

## 2. Architecture Diagram

```
                          ┌──────────────────────┐
                          │   disease.sh API      │
                          │  (External Data Src)  │
                          └──────────┬───────────┘
                                     │ HTTP (with retry)
                          ┌──────────▼───────────┐
                          │   DataPipeline        │  ◄── Data Engineer
                          │  extract → transform  │
                          │       → load (cache)  │
                          └──────────┬───────────┘
                                     │ pd.DataFrame
              ┌──────────────────────┼─────────────────────────┐
              │                      │                          │
   ┌──────────▼──────┐   ┌──────────▼──────┐   ┌─────────────▼─────┐
   │FeatureEngineer  │   │ LSTMPredictor   │   │ ARIMAForecaster    │  ◄── ML Eng
   │(rolling avgs,   │   │(NumPy, no TF)   │   │(statsmodels)       │
   │ lag features,   │   │predict(x_seq)   │   │fit → predict →     │
   │ epidemic phase) │   │                 │   │evaluate (backtest) │
   └──────────┬──────┘   └──────────┬──────┘   └─────────────┬─────┘
              │                      │                          │
              └──────────────────────▼─────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Flask REST API      │  ◄── Backend / Data Eng
                          │  (backend/app.py)     │
                          │  8 endpoints          │
                          │  Rate limiting        │
                          │  Input validation     │
                          └──────────┬───────────┘
                                     │ HTTP/JSON
                          ┌──────────▼───────────┐
                          │  Streamlit Dashboard  │  ◄── Data Analyst
                          │  (frontend/dashboard) │
                          │  4 tabs + Plotly      │
                          │  KPI cards, CI bands  │
                          │  Download CSV buttons │
                          └──────────────────────┘
```

---

## 3. Folder Structure

```
epi-forecast-ai/
│
├── backend/                      # Flask API + ML + Data Engineering
│   ├── __init__.py
│   ├── app.py                    # Flask app — 8 REST endpoints
│   ├── data_pipeline.py          # ETL pipeline (Extract → Transform → Load)
│   ├── data_source.py            # (Legacy, unused — superseded by data_pipeline)
│   ├── feature_engineering.py   # ML feature engineering (rolling avgs, lags, phases)
│   ├── model_numpy.py            # LSTM inference using pure NumPy (no TensorFlow)
│   ├── lstm_model.py             # Training script (run offline to retrain)
│   ├── arima_model.py            # ARIMAForecaster class (fit/predict/evaluate)
│   ├── model_evaluator.py        # RMSE/MAE/MAPE/R², backtest, model comparison
│   └── preprocess.py             # Pydantic schemas, sequence generation, stats
│
├── frontend/
│   └── dashboard.py              # 4-tab Streamlit dashboard
│
├── models/
│   ├── lstm_weights.npz          # Pre-trained LSTM weights (NumPy format)
│   ├── lstm_model_fixed.h5       # Original Keras model (reference only)
│   └── model_registry.json       # Model metadata (version, architecture, dates)
│
├── data/
│   ├── raw/
│   │   └── time_series_covid19_confirmed_global.csv   # JHU offline dataset
│   ├── processed/
│   │   └── india_daily_cases.csv                      # Preprocessed India data
│   └── cache/                    # Runtime JSON cache (created automatically)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py     # Tests: ETL, validation, quality report
│   ├── test_models.py            # Tests: LSTM, FeatureEngineer, ModelEvaluator
│   └── test_api.py               # Tests: All 8 Flask endpoints (mocked)
│
├── .streamlit/
│   └── config.toml               # Streamlit theme (dark mode, colors)
│
├── .env.example                  # Template for environment variables
├── requirements.txt              # All Python dependencies (pinned versions)
├── render.yaml                   # Render deployment config (2 services)
├── runtime.txt                   # Python version for Render
└── DOCUMENTATION.md              # ← You are here
```

---

## 4. Role Mapping

### 🏗️ Data Engineer

| Skill Demonstrated | File / Feature |
|---|---|
| ETL pipeline design (extract → transform → load) | `backend/data_pipeline.py` |
| External API ingestion with retry & timeout | `DataPipeline.extract()` + `tenacity` |
| Data validation & schema enforcement | `backend/preprocess.py` (Pydantic models) |
| Outlier detection & correction (IQR method) | `DataPipeline.transform()` |
| TTL-based disk caching (JSON, upgradeable to Parquet) | `DataPipeline.load()` + `_load_cache()` |
| Automated data quality reporting | `DataPipeline.get_data_quality_report()` |
| Structured logging (Python `logging` module) | All backend files |
| API input whitelist validation | `backend/app.py` `_validate()` |
| Rate limiting | `flask-limiter` in `app.py` |
| Environment-driven configuration | `.env.example`, `os.environ.get()` |
| Deployment config (`render.yaml`) | `render.yaml` |

### 🤖 ML Engineer

| Skill Demonstrated | File / Feature |
|---|---|
| Custom LSTM inference (pure NumPy, no TensorFlow) | `backend/model_numpy.py` |
| Proper ML class design (fit / predict / evaluate) | `backend/arima_model.py` |
| Feature engineering pipeline (multi-scale, lag, phase) | `backend/feature_engineering.py` |
| Rolling averages (7/14/30-day), growth rates, lag features | `FeatureEngineer.build_features()` |
| Epidemiological metrics (doubling time, phase detection) | `FeatureEngineer._classify_phase()` |
| Regression metrics (RMSE, MAE, MAPE, R²) | `backend/model_evaluator.py` |
| Walk-forward LSTM backtesting | `ModelEvaluator.backtest_lstm()` |
| Confidence interval generation | `ModelEvaluator.lstm_confidence_interval()` |
| ARIMA native confidence intervals | `ARIMAForecaster.predict()` (statsmodels CI) |
| ARIMA backtesting | `ARIMAForecaster.evaluate()` |
| Side-by-side model comparison (majority vote) | `ModelEvaluator.compare_models()` |
| Model registry (versioning, metadata) | `models/model_registry.json` |
| Train/val/test split helper | `preprocess.py` `train_val_test_split()` |
| Sequence generation for LSTM | `preprocess.py` `create_sequences()` |
| Unit tests for all ML components | `tests/test_models.py` |

### 📊 Data Analyst

| Skill Demonstrated | File / Feature |
|---|---|
| Interactive multi-tab dashboard | `frontend/dashboard.py` (4 tabs) |
| KPI cards with trend indicators | Tab 1 — Forecast KPIs |
| Interactive Plotly line charts with CI bands | All tabs |
| Severity alert system (🟢🟡🔴🟣) | Tab 1 — Severity Badge |
| Day-by-day forecast cards with delta arrows | Tab 1 |
| LSTM vs ARIMA side-by-side comparison | Tab 2 |
| Metrics table with winner highlighting | Tab 2 |
| Data quality scorecard | Tab 3 — Data Explorer |
| Descriptive statistics (mean, median, std, skew) | Tab 3 |
| Feature-engineered dataset table | Tab 3 |
| Moving average overlay chart | Tab 3 |
| Multi-country forecast comparison | Tab 4 |
| Horizontal bar chart (avg predicted cases) | Tab 4 |
| CSV download buttons on every data view | All tabs |
| Non-blocking auto-refresh | `streamlit-autorefresh` |

---

## 5. Setup & Running Locally

### Prerequisites
- Python 3.10 or 3.11
- pip

### Step 1 — Clone & Install

```bash
git clone https://github.com/samarthjain3580/epi-forecast-ai.git
cd epi-forecast-ai

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 2 — Configure Environment

```bash
# Copy the example and edit if needed
cp .env.example .env
```

Default values in `.env.example` work out of the box for local development.

### Step 3 — Start the Flask API

```bash
# From project root
python -m backend.app
```

You should see:
```
INFO  Starting Epi-Forecast API on port 5000
INFO  LSTM model loaded successfully (NumPy backend)
```

Verify: open `http://127.0.0.1:5000/health` in your browser.

### Step 4 — Start the Streamlit Dashboard

Open a **new terminal window** (keep Flask running):

```bash
# Activate venv again if needed
streamlit run frontend/dashboard.py
```

Open `http://localhost:8501` in your browser.

### Step 5 — Run Tests

```bash
pytest tests/ -v
```

---

## 6. API Reference

Base URL (local): `http://127.0.0.1:5000`  
Base URL (Render): `https://epi-forecast-api.onrender.com`

---

### `GET /`
Service information and uptime.

**Response:**
```json
{
  "service": "Epi-Forecast AI API",
  "version": "2.0.0",
  "uptime_seconds": 42,
  "model_status": { "lstm": "loaded", "arima": "available (statsmodels)" },
  "endpoints": { "GET /predict": "...", ... }
}
```

---

### `GET /health`
Lightweight health check (used by Render's `healthCheckPath`).

**Response:**
```json
{ "status": "ok", "lstm": "loaded", "timestamp": "2024-09-08T12:00:00+00:00" }
```

---

### `GET /countries`
Returns the list of supported countries.

**Response:**
```json
{ "countries": ["argentina", "australia", "austria", ...], "count": 30 }
```

---

### `GET /predict`
Generate a forecast for a given country and model.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `country` | str | `india` | Country name (must be in `/countries`) |
| `days` | int | `7` | Forecast horizon (1–30) |
| `model` | str | `lstm` | Model: `lstm` or `arima` |

**Example:** `GET /predict?country=usa&days=14&model=lstm`

**Response:**
```json
{
  "country": "usa",
  "model": "lstm",
  "days": 14,
  "predictions":  [12345.0, 11200.0, ...],
  "lower_ci":     [10800.0, 9700.0, ...],
  "upper_ci":     [13900.0, 12700.0, ...],
  "past_30_days": [15000.0, 14200.0, ...]
}
```

---

### `GET /compare`
Run both models and return a side-by-side comparison with backtest metrics.

**Query Parameters:**

| Param | Type | Default |
|-------|------|---------|
| `country` | str | `india` |
| `days` | int | `7` |

**Response:**
```json
{
  "country": "india",
  "days": 7,
  "lstm":  { "predictions": [...], "lower_ci": [...], "upper_ci": [...], "backtest_metrics": {"rmse": 1200, "mae": 900, "mape": 8.2, "r2": 0.87} },
  "arima": { "predictions": [...], "lower_ci": [...], "upper_ci": [...], "backtest_metrics": {"rmse": 1800, "mae": 1200, "mape": 12.1, "r2": 0.79} },
  "comparison": {
    "metric_winners": { "rmse": "LSTM", "mae": "LSTM", "mape": "LSTM", "r2": "LSTM" },
    "recommendation": "LSTM"
  },
  "past_30_days": [...]
}
```

---

### `GET /data/stats`
Descriptive statistics for historical case data.

**Response:**
```json
{
  "country": "india",
  "date_range": "2020-01-30 → 2023-03-09",
  "statistics": { "count": 1144, "mean": 42310.5, "median": 18500.0, "std": 55000.0, "min": 0.0, "max": 414000.0, "p25": 3200.0, "p75": 68000.0, "skewness": 2.1, "kurtosis": 5.3 },
  "total_cumulative_cases": 44690000
}
```

---

### `GET /data/quality`
ETL pipeline data quality report.

**Response:**
```json
{
  "country": "india",
  "quality_report": {
    "quality_score": 88.5,
    "completeness_pct": 100.0,
    "total_records": 1144,
    "null_records": 0,
    "zero_records": 12,
    "outliers_detected": 8,
    "data_freshness": "STALE (>2 days)",
    "date_range": "2020-01-30 → 2023-03-09",
    "mean_daily_cases": 42310.5,
    "max_daily_cases": 414000.0
  }
}
```

---

### `GET /data/features`
Feature-engineered dataset (last 60 days) with epidemic indicators.

**Response:**
```json
{
  "country": "india",
  "feature_summary": {
    "current_cases": 1200.0,
    "ma_7": 1350.0,
    "ma_14": 1500.0,
    "growth_rate_7d": -12.5,
    "epidemic_phase": "Declining",
    "doubling_time": null
  },
  "recent_features": [
    { "date": "2023-03-09", "cases": 1200.0, "ma_7": 1350.0, "ma_14": 1500.0, "growth_rate_7d": -0.125, "epidemic_phase": "Declining" },
    ...
  ]
}
```

---

## 7. Dashboard Guide

### Tab 1: 📈 Forecast

The main forecasting tab. Uses sidebar settings for country, forecast days, and model.

| Element | Description |
|---|---|
| **KPI Cards (top row)** | Current 7-day avg cases, peak predicted, forecast days, epidemic phase |
| **Severity Badge** | Color-coded alert: 🟢 Low → 🟡 Moderate → 🔴 High → 🟣 Critical |
| **Plotly Chart** | Historical (blue) + Forecast (dotted) + 90% CI band |
| **Day-by-Day Cards** | Each forecast day with value and change arrow (▲/▼) |
| **Download Button** | CSV with date, predicted_cases, lower_ci, upper_ci, model, country |

### Tab 2: 🤖 Model Comparison

Runs both LSTM and ARIMA, backtests each on the last 30 days.

| Element | Description |
|---|---|
| **Recommendation Banner** | Highlighted winner (LSTM or ARIMA) with justification |
| **Comparison Chart** | Both forecasts + CI bands on one Plotly chart |
| **Metrics Table** | RMSE, MAE, MAPE, R² for each model with winner highlighted |
| **Download Button** | CSV with date, lstm_forecast, arima_forecast |

### Tab 3: 📊 Data Explorer

Data quality, statistics, and feature engineering view.

| Element | Description |
|---|---|
| **Quality Score Card** | Score (0–100), completeness %, freshness status, outlier count |
| **Quality Progress Bar** | Visual representation of quality score |
| **Descriptive Stats** | Mean, Median, Std Dev, Peak cases |
| **Feature Table** | Last 60 days with MA-7/14/30, growth rate, epidemic phase |
| **Cases + MA Chart** | Line chart overlaying raw cases and 7-day moving average |
| **Download Button** | Full feature-engineered CSV |

### Tab 4: 🌍 Multi-Country

Compare forecasts across up to 6 countries simultaneously.

| Element | Description |
|---|---|
| **Multi-select** | Choose up to 6 countries from the supported list |
| **Forecast Horizon Radio** | 7, 14, or 30 days |
| **Line Chart** | All selected countries on one chart |
| **Bar Chart** | Average predicted daily cases ranked by country |
| **Download Button** | Summary CSV with avg_forecast per country |

---

## 8. ML Models

### LSTM (Long Short-Term Memory)

| Attribute | Value |
|---|---|
| Architecture | Single-layer LSTM (50 units) → Dense(1) |
| Input shape | `(1, 30, 1)` — 30-day window, 1 feature |
| Output | Next-day scaled case prediction |
| Inference engine | **Pure NumPy** (no TensorFlow at runtime) |
| Weights file | `models/lstm_weights.npz` |
| Normalisation | MinMaxScaler (0, 1) — fitted at request time on the specific country's data |
| Confidence interval | Volatility-based heuristic (grows ±12% + 10% per step) |
| Backtest | Walk-forward: seed with window before test period, autoregress for 30 steps |

**How NumPy inference works** (`model_numpy.py`):
```
For each timestep t in the 30-day window:
  i = σ(x_t · W_i + h_{t-1} · U_i + b_i)    ← input gate
  f = σ(x_t · W_f + h_{t-1} · U_f + b_f)    ← forget gate
  c̃ = tanh(x_t · W_c + h_{t-1} · U_c + b_c) ← cell candidate
  o = σ(x_t · W_o + h_{t-1} · U_o + b_o)    ← output gate
  c = f * c + i * c̃                           ← cell state
  h = o * tanh(c)                              ← hidden state

Output: h @ W_dense + b_dense
```

### ARIMA (AutoRegressive Integrated Moving Average)

| Attribute | Value |
|---|---|
| Order | `(5, 1, 0)` — AR=5, I=1, MA=0 |
| Library | `statsmodels.tsa.arima.model.ARIMA` |
| Training data | Last 365 days of country data (fitted at request time) |
| Confidence interval | Native 90% CI from statsmodels `get_forecast()` |
| Backtest | Re-fit on data minus last 30 days, forecast 30 steps, compare to actuals |

**Why `(5, 1, 0)`?**
- `d=1`: Daily case counts are non-stationary (trend present) → one differencing pass
- `p=5`: Includes last 5 differenced terms as autoregressive predictors
- `q=0`: No moving-average component (simple AR model sufficient for this use case)

---

## 9. Data Engineering Pipeline

The `DataPipeline` class in `backend/data_pipeline.py` implements a full ETL pattern:

```
extract(country) ──► transform(df) ──► load(df, country) ──► return df
      │                   │                    │
      ▼                   ▼                    ▼
  disease.sh API    7 transformation     JSON cache in
  (with 3 retries,  steps (diff, clip,  data/cache/{country}.json
   15s timeout,      IQR outliers,      + metadata with TTL
   whitelist check)  7-day smooth,
                      trim zeros)
```

### Transform Steps

| Step | What It Does | Why |
|---|---|---|
| 1. Diff | Cumulative → Daily | Raw data is cumulative |
| 2. Clip negatives | `daily_cases.clip(lower=0)` | Reporting corrections create negative diffs |
| 3. IQR outlier detection | `upper_fence = Q3 + 3*IQR` | Batch-reported backlogs create spikes |
| 4. Outlier correction | Replace outlier with rolling median | Preserve trend, remove artifacts |
| 5. 7-day rolling mean | Smooth weekend/batch reporting effects | Day-of-week reporting bias |
| 6. Trim leading zeros | Drop pre-pandemic rows | Avoid training/scaling on irrelevant data |

### Cache Design

```
data/cache/
├── india.json          ← DataFrame records (JSON orient)
├── india_meta.json     ← { cached_at, ttl_seconds, rows, pipeline_metadata }
├── usa.json
└── usa_meta.json
```

Cache is invalidated when `(now - cached_at) > ttl_seconds`. Default TTL = 3600s (1 hour).

**Production upgrade**: Replace JSON cache with Parquet on S3/GCS for columnar efficiency.

---

## 10. Feature Engineering

`FeatureEngineer.build_features(df)` adds the following columns:

| Feature | Formula | Purpose |
|---|---|---|
| `ma_7` | `cases.rolling(7).mean()` | Short-term trend |
| `ma_14` | `cases.rolling(14).mean()` | Mid-term trend |
| `ma_30` | `cases.rolling(30).mean()` | Long-term trend |
| `std_7` | `cases.rolling(7).std()` | Volatility signal |
| `std_14` | `cases.rolling(14).std()` | Volatility signal |
| `growth_rate_7d` | `cases.pct_change(7)` | Week-over-week growth |
| `growth_rate_14d` | `cases.pct_change(14)` | Fortnight-over-fortnight growth |
| `lag_1` | `cases.shift(1)` | Yesterday's cases |
| `lag_7` | `cases.shift(7)` | Last week's cases |
| `lag_14` | `cases.shift(14)` | Two weeks ago |
| `lag_21` | `cases.shift(21)` | Three weeks ago |
| `doubling_time` | `log(2) / log(1 + growth_rate_7d)` | Days for cases to double |
| `epidemic_phase` | Rule-based on growth_rate_7d | Categorical phase label |
| `week_of_year` | `index.isocalendar().week` | Seasonality |
| `day_of_week` | `index.dayofweek` | Reporting bias signal |

### Epidemic Phase Definitions

| Phase | Condition |
|---|---|
| **Exponential Growth** | growth_rate_7d > 20% AND cases > 10 |
| **Rising** | growth_rate_7d > 5% AND cases > 10 |
| **Plateau** | -5% ≤ growth_rate_7d ≤ 5% AND cases > 10 |
| **Declining** | growth_rate_7d < -5% AND cases > 10 |
| **Rapid Decline** | growth_rate_7d < -20% AND cases > 10 |
| **Low Activity** | cases ≤ 10 |

---

## 11. Model Evaluation & Comparison

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | `√(mean((actual - pred)²))` | Penalises large errors; scale-dependent |
| **MAE** | `mean(|actual - pred|)` | Average absolute error; robust to outliers |
| **MAPE** | `mean(|actual - pred| / actual) × 100` | % error; only computed where actual > 1 |
| **R²** | `1 - SS_res / SS_tot` | Explained variance; 1.0 = perfect |

### Backtesting

Both models are evaluated on the **last 30 days of historical data** (out-of-sample):

```
Full series:  ──────────────────────────────── [train] ───── [test: 30 days]
                                                              ↑
                                                    Predictions evaluated here
```

**LSTM backtest**: Seed with the window before test period → autoregress 30 steps  
**ARIMA backtest**: Re-fit on training data → `forecast(30)` → compare to test actuals

### Winner Selection

A **majority vote** across RMSE, MAE, MAPE determines the recommended model.
R² is included in the report but not the vote (can be undefined for some data shapes).

---

## 12. Deploying to Render

### Step 1 — Create a Render Account
Sign up at [render.com](https://render.com).

### Step 2 — Connect GitHub Repo
- New → Web Service → Connect your GitHub repo
- Render auto-detects `render.yaml` and creates **both services** automatically

### Step 3 — Environment Variables
Render will automatically wire `BACKEND_URL` from the API service to the dashboard using `fromService` in `render.yaml`. You can manually set others:

| Variable | Value | Service |
|---|---|---|
| `CACHE_TTL_SECONDS` | `3600` | Backend (API) |
| `MAX_FORECAST_DAYS` | `30` | Backend (API) |
| `LOG_LEVEL` | `INFO` | Backend (API) |

### Step 4 — Deploy
Click **Deploy**. Render will:
1. `pip install -r requirements.txt`
2. Start Flask API with `gunicorn backend.app:app --workers 1 --timeout 120`
3. Start Streamlit with `streamlit run frontend/dashboard.py --server.port=$PORT --server.headless=true`

### ⚠️ Free Tier Notes
- Services **spin down after 15 minutes of inactivity** on the free tier
- First request after spin-down takes ~30–60 seconds (cold start)
- The dashboard has a 35-second timeout for API calls to handle this
- Consider upgrading to Render Starter ($7/month) for always-on services

---

## 13. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only data pipeline tests
pytest tests/test_data_pipeline.py -v

# Run only model tests
pytest tests/test_models.py -v

# Run only API tests
pytest tests/test_api.py -v

# With coverage (install pytest-cov first)
pip install pytest-cov
pytest tests/ --cov=backend --cov-report=term-missing
```

**Note**: API tests use mocked dependencies — no real API calls are made. You can run them offline.

---

## 14. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:5000` | Backend API URL (used by dashboard) |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL in seconds (1 hour) |
| `MAX_FORECAST_DAYS` | `30` | Max forecast days allowed by API |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `PORT` | `5000` | Flask server port (overridden by Render) |

---

## 15. How to Extend This Project

### Add a New Disease (e.g., Flu)

1. Create `backend/flu_data_source.py` — fetch from WHO FluNet API
2. Add a `disease` parameter to `DataPipeline.extract(country, disease="covid")`
3. Route `disease=flu` to `flu_data_source.py`
4. Add flu to the dashboard's disease selector dropdown

### Add a New Country

Add the country name (lowercase) to `ALLOWED_COUNTRIES` in `backend/data_pipeline.py`.
Verify that `disease.sh` supports it via `https://disease.sh/v3/covid-19/historical/{country}`.

### Add a New Model (e.g., Prophet)

1. Create `backend/prophet_model.py` with `ProphetForecaster(fit, predict, evaluate)`
2. Import and instantiate in `app.py`
3. Add `model=prophet` as a valid option in `_validate()` and the `predict` route
4. Add Prophet forecasts to the comparison tab in `dashboard.py`

> ⚠️ Prophet is ~300MB — test Render build limits before adding.

### Replace JSON Cache with Database

In `DataPipeline.load()`, replace `df.to_json(cache_path)` with:
```python
# PostgreSQL example
engine = create_engine(os.environ["DATABASE_URL"])
df.to_sql(f"cases_{country}", engine, if_exists="replace")
```

### Add User Authentication

Use `flask-jwt-extended` for JWT-based API authentication:
```python
from flask_jwt_extended import jwt_required
@app.route("/predict")
@jwt_required()
def predict(): ...
```

### Add a Scheduled Retraining Job

Create a `scripts/retrain.py` that:
1. Fetches latest data via `DataPipeline`
2. Generates sequences via `create_sequences()`
3. Trains a new LSTM via TensorFlow/Keras
4. Exports weights to `models/lstm_weights.npz`
5. Updates `models/model_registry.json`

Schedule with cron or a Render Cron Job service.

---

*Documentation last updated: September 2024 | EpiCast AI v2.0*
