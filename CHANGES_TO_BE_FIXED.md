# EpiCast AI — Changes to be Fixed & Implementation Plan
> **Status:** Work in progress  
> **Target Date:** September 2026  
> **Author:** Antigravity AI Engineering Assistant  
> **Repository:** `v-kaushik3010/EPI-Forecast`  
> **Live API:** `https://epi-forecast-api-160c.onrender.com`  
> **Live Dashboard:** `https://epi-forecast-dashboard.onrender.com`  

---

## 1. Executive Summary & Root Cause Analysis

### The Problem
When the dashboard is opened (especially after hours of inactivity or the next day), the user is greeted with red error banners:
```
API error: 502 Server Error: Bad Gateway for url: https://epi-forecast-api-160c.onrender.com/countries
API error: 502 Server Error: Bad Gateway for url: https://epi-forecast-api-160c.onrender.com/data/features?country=india
```
Additionally, visiting the API URL directly in a browser shows:
```
502 Bad Gateway
This service is currently unavailable. Please try again in a few minutes.
```

### Root Cause
1. **Render Free-Tier Cold Starts (Spin-down):**
   Render spins down free web services after 15 minutes of inactivity to save compute resources.
2. **Boot-Up Window & Cold-Start Race:**
   When the user visits `https://epi-forecast-dashboard.onrender.com`, the dashboard boots up and immediately fires multiple HTTP requests to the backend (`/countries`, `/predict`, `/data/features`).
   Because the backend container was sleeping, Render's router initiates container spin-up. Loading Python, Flask, PyTorch/NumPy, Scikit-Learn, and Statsmodels takes 15–30 seconds. During this window, Render's reverse proxy returns `502 Bad Gateway`.
3. **Frontend Fragility & Caching of Failures:**
   In `frontend/dashboard.py`:
   - `api()` immediately raised and rendered `st.error(f"⚠️ API error: {exc}")` upon seeing the 502 response instead of retrying or showing a warm-up banner.
   - Streamlit cached the failed response (`None` or default fallback) with `@st.cache_data(ttl=300)`. This meant that even after the backend woke up 20 seconds later, the user remained stuck with empty/error screens for 5 minutes unless they manually forced a rerun or cleared their cache!

---

## 2. Architecture & Service Topology

```
+-------------------------------------------------------------+
|                     User Browser                           |
+------------------------------+------------------------------+
                               |
                               v
             +---------------------------------+
             | Streamlit Frontend (Render Web) |
             | epi-forecast-dashboard          |
             +----------------+----------------+
                              |
                     HTTPS    |  (Wakes backend on sleep)
                              v
             +---------------------------------+
             |   Flask API (Render Web)        |
             |   epi-forecast-api-160c         |
             +----------------+----------------+
                              |
             +----------------+----------------+
             |   Data Pipeline / ML Inference  |
             |   disease.sh -> Cache -> NumPy  |
             +---------------------------------+
```

---

## 3. Targeted Fixes

### Fix 1: Add `/ping` Ultra-Lightweight Keep-Alive Route
- **File:** `backend/app.py`
- **Action:** Add `@app.route("/ping")` returning `{"status": "ok", "pong": True}` instantly without importing heavy data pipelines or running models.
- **Purpose:** Gives Render and the dashboard a zero-cost endpoint for health checks and wake-up pings that responds in <5ms as soon as Gunicorn binds.

### Fix 2: Optimize Gunicorn Startup & Health Check in `render.yaml`
- **File:** `render.yaml`
- **Action:**
  - Update `healthCheckPath` from `/health` to `/ping`.
  - Add `--preload` to Gunicorn command: preloads application modules before worker forking for predictable memory and rapid request handling.
  - Retain `BACKEND_URL: https://epi-forecast-api-160c.onrender.com`.

### Fix 3: Resilient Dashboard API Client with Automatic Warm-Up & Retries
- **File:** `frontend/dashboard.py`
- **Action:**
  - Enhance `api(endpoint, **params)` to detect cold-start status codes (`502`, `503`, `504`) and connection timeouts.
  - Automatically retry up to 3 times with progressive backoff when waking up a spun-down server.
  - Display a clean, informative "⏳ Waking up server from standby..." message in place of scary red 502 crashes.
  - Ensure `@st.cache_data` functions only cache valid responses, never caching failed `None` results.

---

## 4. Exact File Modifications Checklist

| File | Change Description | Status |
| :--- | :--- | :--- |
| `CHANGES_TO_BE_FIXED.md` | Create comprehensive documentation for continuity across agents | Completed |
| `backend/app.py` | Add `/ping` route | Pending |
| `render.yaml` | Switch `healthCheckPath` to `/ping`, add `--preload` to Gunicorn | Pending |
| `frontend/dashboard.py` | Implement auto-retry on 502/cold-start & prevent caching of errors | Pending |
| `tests/test_api.py` | Add test coverage for `/ping` endpoint | Pending |

---

## 5. Verification Plan

1. **Local Test Suite:** Run `python -m pytest` to verify all existing and new unit tests pass (must be 100% green).
2. **API Endpoint Verification:** Test `/ping`, `/health`, `/countries`, `/predict` locally and via `curl`.
3. **Deployment Verification:**
   - Commit and push to `origin main`.
   - Verify Render triggers deployment for `epi-forecast-api` and `epi-forecast-dashboard`.
   - Check `https://epi-forecast-api-160c.onrender.com/ping` returns 200 OK.
   - Load `https://epi-forecast-dashboard.onrender.com` in browser to confirm zero 502 errors.
