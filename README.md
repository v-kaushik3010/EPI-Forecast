<div align="center">

<h1>🦠 EpiCast AI</h1>
<p><strong>AI-powered epidemic forecasting platform · LSTM + ARIMA · Real-time COVID-19 data</strong></p>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

</div>

---

## What is EpiCast AI?

**EpiCast AI** is a full-stack epidemic forecasting platform that ingests live COVID-19 data, runs multiple ML models, and serves both a REST API and an interactive analytics dashboard.

It is built to demonstrate production-grade skills across three professional profiles:

| 🏗️ Data Engineer | 🤖 ML Engineer | 📊 Data Analyst |
|---|---|---|
| ETL pipeline with retry & caching | LSTM + ARIMA model classes | 4-tab interactive dashboard |
| IQR outlier detection & correction | Backtesting & RMSE/MAE/MAPE/R² | Plotly charts with confidence bands |
| Pydantic schema validation | Feature engineering pipeline | KPI cards & severity alerts |
| TTL-based disk cache | Model registry & versioning | CSV download on every view |
| Rate limiting & structured logging | Confidence interval generation | Multi-country comparison |

---

## Live Demo

| Service | URL |
|---|---|
| 🖥️ Dashboard | `https://epi-forecast-dashboard.onrender.com` |
| ⚙️ API | `https://epi-forecast-api.onrender.com` |
| ❤️ Health Check | `https://epi-forecast-api.onrender.com/health` |

> ⚠️ Free tier services spin down after inactivity — first load may take ~30s.

---

## Architecture

```
disease.sh API (COVID-19 time-series)
        │
        ▼
┌────────────────────┐
│   DataPipeline     │  ETL · retry · IQR outliers · TTL cache · quality report
└────────┬───────────┘
         │
    ┌────┴────────────────────────────┐
    ▼                                 ▼
┌──────────────────┐      ┌─────────────────────────┐
│  LSTMPredictor   │      │    ARIMAForecaster       │
│  (NumPy, no TF)  │      │  fit · predict · eval    │
└────────┬─────────┘      └────────────┬────────────┘
         └────────────────┬────────────┘
                          ▼
              ┌───────────────────────┐
              │  Flask REST API       │  8 endpoints · rate limiting
              │  (backend/app.py)     │  validation · structured logging
              └───────────┬───────────┘
                          │ HTTP/JSON
                          ▼
              ┌───────────────────────┐
              │  Streamlit Dashboard  │  4 tabs · Plotly · KPIs
              │  (frontend/dashboard) │  CI bands · CSV downloads
              └───────────────────────┘
```

---

## Project Structure

```
epi-forecast-ai/
│
├── backend/
│   ├── app.py                   # Flask API — 8 REST endpoints
│   ├── data_pipeline.py         # ETL pipeline (extract → transform → load)
│   ├── feature_engineering.py   # Rolling MA, lag features, epidemic phase
│   ├── model_numpy.py           # LSTM inference (pure NumPy — no TensorFlow)
│   ├── arima_model.py           # ARIMAForecaster class (fit/predict/evaluate)
│   ├── model_evaluator.py       # RMSE/MAE/MAPE/R², backtesting, comparison
│   └── preprocess.py            # Pydantic schemas, sequence generator, stats
│
├── frontend/
│   └── dashboard.py             # 4-tab Streamlit dashboard
│
├── models/
│   ├── lstm_weights.npz         # Pre-trained LSTM weights (NumPy format)
│   └── model_registry.json      # Model metadata — version, architecture, dates
│
├── data/
│   ├── raw/                     # JHU offline dataset
│   ├── processed/               # Preprocessed India time-series
│   └── cache/                   # Runtime JSON cache (auto-created)
│
├── tests/
│   ├── test_data_pipeline.py    # ETL, validation, quality report tests
│   ├── test_models.py           # LSTM, FeatureEngineer, ModelEvaluator tests
│   └── test_api.py              # All 8 Flask endpoints (mocked)
│
├── .env.example                 # Environment variable template
├── .streamlit/config.toml       # Dark theme configuration
├── requirements.txt             # Pinned dependencies
├── render.yaml                  # Render deployment (2 services)
├── runtime.txt                  # Python version
├── README.md                    # This file
└── DOCUMENTATION.md             # Full technical documentation
```

---

## Quick Start

### Prerequisites
- Python 3.10 or 3.11
- pip

### 1. Install dependencies

```bash
git clone https://github.com/samarthjain3580/epi-forecast-ai.git
cd epi-forecast-ai

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Start the Flask API

```bash
python -m backend.app
```

Verify at `http://127.0.0.1:5000/health`

### 3. Start the Dashboard

```bash
# New terminal
streamlit run frontend/dashboard.py
```

Open `http://localhost:8501`

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info & uptime |
| `GET` | `/health` | Model & service status |
| `GET` | `/countries` | Supported countries list (30 countries) |
| `GET` | `/predict` | LSTM or ARIMA forecast + confidence intervals |
| `GET` | `/compare` | Side-by-side model comparison + backtest metrics |
| `GET` | `/data/stats` | Descriptive statistics (mean, median, std, skew) |
| `GET` | `/data/quality` | ETL data quality report (completeness, freshness) |
| `GET` | `/data/features` | Feature-engineered dataset (last 60 days) |

**Example:**
```bash
curl "http://localhost:5000/predict?country=india&days=14&model=lstm"
```

```json
{
  "country": "india",
  "model": "lstm",
  "days": 14,
  "predictions": [12300.0, 11800.0, "..."],
  "lower_ci": [10800.0, 10200.0, "..."],
  "upper_ci": [13800.0, 13400.0, "..."],
  "past_30_days": [15000.0, 14200.0, "..."]
}
```

> Full API reference with all parameters and response schemas → [`DOCUMENTATION.md`](./DOCUMENTATION.md#6-api-reference)

---

## Dashboard Tabs

| Tab | What It Shows |
|-----|---------------|
| **📈 Forecast** | KPI cards, severity alert, interactive Plotly chart with 90% CI band, day-by-day cards, CSV download |
| **🤖 Model Comparison** | LSTM vs ARIMA forecasts overlaid, backtest metrics table (RMSE/MAE/MAPE/R²), winner recommendation |
| **📊 Data Explorer** | Data quality scorecard, descriptive stats, feature-engineered table, moving average chart |
| **🌍 Multi-Country** | Up to 6 countries compared on one chart, ranked bar chart of avg predicted cases |

---

## ML Models

### LSTM (Long Short-Term Memory)
- **Architecture**: Single-layer LSTM (50 units) → Dense(1)
- **Inference**: Pure NumPy — no TensorFlow dependency at runtime
- **Input**: 30-day sliding window of normalised daily cases
- **Output**: Next-day prediction · auto-regressed for N days
- **Confidence**: Volatility-based heuristic (grows ±12% with horizon)

### ARIMA (5, 1, 0)
- **Library**: `statsmodels`
- **Fitted at**: Request time on last 365 days of data
- **Confidence**: Native 90% intervals from `get_forecast()`
- **Backtest**: Re-fit on train set, evaluate on last 30 days

---

## Deploying to Render

1. Fork this repo on GitHub
2. Sign in to [render.com](https://render.com) → **New → Blueprint**
3. Connect your fork — Render auto-reads `render.yaml` and creates **both services**
4. Click **Deploy**

`render.yaml` provisions:
- `epi-forecast-api` — Flask + Gunicorn backend
- `epi-forecast-dashboard` — Streamlit frontend
- `BACKEND_URL` is automatically wired from API → Dashboard

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:5000` | API URL used by the dashboard |
| `CACHE_TTL_SECONDS` | `3600` | Data cache duration (seconds) |
| `MAX_FORECAST_DAYS` | `30` | Maximum forecast horizon |
| `LOG_LEVEL` | `INFO` | Python logging level |

Copy `.env.example` → `.env` for local configuration.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | Flask 3.1 + Flask-CORS + Flask-Limiter |
| Dashboard | Streamlit 1.41 + Plotly 5.24 |
| ML / Statistics | NumPy 2.0 · scikit-learn 1.6 · statsmodels 0.14 |
| Data | Pandas 2.2 · Pydantic 2.10 |
| Resilience | Tenacity (retry) · Cachetools (TTL cache) |
| Testing | Pytest 8.3 + pytest-mock |
| Deployment | Gunicorn · Render |

---

## Documentation

For deep-dives into architecture, ML model internals, data pipeline design, feature engineering, and extension guides, see:

**[📖 DOCUMENTATION.md](./DOCUMENTATION.md)**

Covers:
- Architecture diagram
- Full API reference with request/response schemas
- Dashboard tab guide
- LSTM NumPy inference walkthrough
- ARIMA order selection rationale
- ETL pipeline transform steps
- Feature engineering table
- Backtesting methodology
- Render deployment guide
- How to add new diseases, models, and countries

---

## Data Source

All data is sourced from **[disease.sh](https://disease.sh)** — a free, open-source COVID-19 API aggregating data from Johns Hopkins University, Worldometers, and government sources.

> This project uses COVID-19 data only. The architecture is designed to plug in additional disease data sources (WHO FluNet for influenza, PAHO for dengue, etc.).

---

<div align="center">
<p>Built with ❤️ to showcase Data Engineering · ML Engineering · Data Analytics skills</p>
<p>
  <a href="./DOCUMENTATION.md">📖 Full Docs</a> ·
  <a href="https://disease.sh">🌐 Data Source</a> ·
  <a href="https://render.com">🚀 Deploy on Render</a>
</p>
</div>
