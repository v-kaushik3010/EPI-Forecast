"""
dashboard.py — EpiCast AI Interactive Dashboard  (v2.0)
=========================================================
4-tab professional dashboard demonstrating Data Analyst skills:

  Tab 1 │ 📈 Forecast       — KPI cards, Plotly forecast chart, severity alert
  Tab 2 │ 🤖 Model Compare  — LSTM vs ARIMA side-by-side, metrics table
  Tab 3 │ 📊 Data Explorer  — Feature table, epidemic phase chart, quality score
  Tab 4 │ 🌍 Multi-Country  — Compare multiple countries on one chart
"""

import os
import io
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiCast AI | Epidemic Forecast Platform",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── KPI Cards ─────────────────────────────────────────── */
  .kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
  }
  .kpi-label  { color: #94a3b8; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
  .kpi-value  { color: #f1f5f9; font-size: 28px; font-weight: 700; line-height: 1.2; }
  .kpi-delta  { font-size: 13px; font-weight: 500; margin-top: 4px; }
  .kpi-delta.up   { color: #f87171; }
  .kpi-delta.down { color: #34d399; }
  .kpi-delta.flat { color: #94a3b8; }

  /* ── Severity Badge ─────────────────────────────────────── */
  .severity-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
    margin: 4px 0;
  }
  .sev-low      { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .sev-moderate { background: #422006; color: #fbbf24; border: 1px solid #92400e; }
  .sev-high     { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
  .sev-critical { background: #2e1065; color: #c084fc; border: 1px solid #7c3aed; }

  /* ── Phase Badge ────────────────────────────────────────── */
  .phase-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }

  /* ── Section Headers ────────────────────────────────────── */
  .section-header {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
  }
  .section-sub {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 16px;
  }

  /* ── Metric Table ───────────────────────────────────────── */
  .metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    border-bottom: 1px solid #1e293b;
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-name  { color: #94a3b8; font-size: 13px; }
  .metric-value { color: #f1f5f9; font-weight: 600; }
  .winner-tag   { background: #1d4ed8; color: #bfdbfe; padding: 2px 8px;
                  border-radius: 8px; font-size: 11px; font-weight: 600; }

  /* ── Quality Score ──────────────────────────────────────── */
  .quality-bar-wrap {
    background: #1e293b; border-radius: 8px; height: 10px; overflow: hidden; margin: 8px 0;
  }
  .quality-bar-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    transition: width 0.5s ease;
  }

  /* ── Sidebar Logo ───────────────────────────────────────── */
  .sidebar-logo {
    text-align: center; padding: 20px 0 10px;
  }
  .logo-emoji { font-size: 52px; }
  .logo-title { color: #f1f5f9; font-size: 22px; font-weight: 700; margin-top: 6px; }
  .logo-sub   { color: #64748b; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_API_BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")
if _API_BASE and not _API_BASE.startswith("http"):
    _API_BASE = f"https://{_API_BASE}"

TIMEOUT = 35

PHASE_CSS = {
    "Exponential Growth": "background:#450a0a;color:#f87171;",
    "Rising":             "background:#431407;color:#fb923c;",
    "Plateau":            "background:#422006;color:#fbbf24;",
    "Declining":          "background:#052e16;color:#4ade80;",
    "Rapid Decline":      "background:#022c22;color:#34d399;",
    "Low Activity":       "background:#1e1b4b;color:#a5b4fc;",
}

COLORS = {
    "lstm":       "#6366f1",
    "arima":      "#f59e0b",
    "hist":       "#60a5fa",
    "ci":         "rgba(99,102,241,0.15)",
    "ci_arima":   "rgba(245,158,11,0.15)",
    "grid":       "#1e293b",
    "bg":         "#0f172a",
    "paper":      "#0f172a",
    "text":       "#94a3b8",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor =COLORS["bg"],
    font=dict(family="Inter", color=COLORS["text"], size=12),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    margin=dict(l=20, r=20, t=50, b=20),
    hovermode="x unified",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def api(endpoint: str, **params):
    """Call the backend API. Returns parsed JSON or None on error."""
    try:
        url = f"{_API_BASE}/{endpoint.lstrip('/')}"
        r   = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Is the Flask server running?")
        return None
    except Exception as exc:
        st.error(f"⚠️ API error: {exc}")
        return None


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def future_dates(n: int) -> list:
    today = datetime.today()
    return [(today + timedelta(days=i + 1)).strftime("%b %d") for i in range(n)]


def severity_html(growth: float, phase: str) -> str:
    if phase == "Exponential Growth":
        return '<span class="severity-badge sev-critical">🟣 CRITICAL</span>'
    elif phase in ("Rising",):
        return '<span class="severity-badge sev-high">🔴 HIGH</span>'
    elif phase in ("Plateau",):
        return '<span class="severity-badge sev-moderate">🟡 MODERATE</span>'
    else:
        return '<span class="severity-badge sev-low">🟢 LOW</span>'


def phase_html(phase: str) -> str:
    style = PHASE_CSS.get(phase, "background:#1e293b;color:#94a3b8;")
    return f'<span class="phase-badge" style="{style}">{phase}</span>'


def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "flat") -> str:
    delta_html = (
        f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    )
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>"""


def fmt(val, decimals=0):
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if decimals == 0:
            return f"{v:,.0f}"
        return f"{v:,.{decimals}f}"
    except Exception:
        return str(val)


@st.cache_data(ttl=300, show_spinner=False)
def get_countries():
    data = api("countries")
    return data.get("countries", []) if data else ["india", "usa", "brazil", "uk"]


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast(country: str, days: int, model: str):
    return api("predict", country=country, days=days, model=model)


@st.cache_data(ttl=300, show_spinner=False)
def get_features(country: str):
    return api("data/features", country=country)


@st.cache_data(ttl=300, show_spinner=False)
def get_compare(country: str, days: int):
    return api("compare", country=country, days=days)


@st.cache_data(ttl=300, show_spinner=False)
def get_quality(country: str):
    return api("data/quality", country=country)


@st.cache_data(ttl=300, show_spinner=False)
def get_stats(country: str):
    return api("data/stats", country=country)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="logo-emoji">🦠</div>
      <div class="logo-title">EpiCast AI</div>
      <div class="logo-sub">Epidemic Forecast Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    countries_list = get_countries()
    country = st.selectbox(
        "🌍 Country",
        countries_list,
        index=countries_list.index("india") if "india" in countries_list else 0,
    )

    days = st.slider("📅 Forecast Horizon (days)", min_value=7, max_value=30, value=7, step=7)

    model = st.radio("🤖 Primary Model", ["lstm", "arima"], format_func=str.upper)

    st.markdown("---")
    st.markdown("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable (every 5 min)", value=False)
    if auto_refresh:
        st_autorefresh(interval=300_000, key="auto_refresh")

    st.markdown("---")
    st.markdown(
        '<p style="color:#475569;font-size:11px;text-align:center;">'
        'Data: disease.sh API<br>Models: LSTM + ARIMA<br>v2.0.0</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="color:#f1f5f9;font-size:32px;font-weight:700;margin-bottom:4px;">'
    f'🦠 EpiCast AI — {country.title()}</h1>'
    f'<p style="color:#64748b;font-size:14px;margin-bottom:24px;">'
    f'AI-powered epidemic forecasting · LSTM + ARIMA · '
    f'{datetime.now().strftime("%d %b %Y, %H:%M")}</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast",
    "🤖 Model Comparison",
    "📊 Data Explorer",
    "🌍 Multi-Country",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner(f"Fetching {model.upper()} forecast for {country.title()}..."):
        data = get_forecast(country, days, model)
        feat = get_features(country)

    if data is None:
        st.stop()

    preds    = data.get("predicted_cases_next_7_days") or data.get("predictions", [])
    past     = data.get("past_30_days", [])
    lower_ci = data.get("lower_ci", [])
    upper_ci = data.get("upper_ci", [])
    summary  = feat.get("feature_summary", {}) if feat else {}

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📌 Key Metrics</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    growth    = summary.get("growth_rate_7d", 0)
    phase     = summary.get("epidemic_phase", "Unknown")
    cur_cases = summary.get("current_cases", past[-1] if past else 0)
    peak_pred = max(preds) if preds else 0

    delta_dir = "up" if growth > 5 else ("down" if growth < -5 else "flat")
    delta_sym = "▲" if growth > 0 else ("▼" if growth < 0 else "→")

    with k1:
        st.markdown(kpi_card(
            "Current Cases (7d avg)", fmt(cur_cases),
            f"{delta_sym} {abs(growth):.1f}% vs last week", delta_dir
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Peak Predicted", fmt(peak_pred)), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Forecast Days", str(days)), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Epidemic Phase", phase), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Severity Alert ────────────────────────────────────────────────────────
    sev_col, _ = st.columns([1, 3])
    with sev_col:
        st.markdown(
            f"**Severity Level:** {severity_html(growth, phase)}", unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Plotly Forecast Chart ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Forecast Chart</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">'
        f'Historical (last 30 days) + {days}-day {model.upper()} forecast with confidence band'
        f'</p>',
        unsafe_allow_html=True,
    )

    total_len   = len(past) + len(preds)
    x_past      = list(range(len(past)))
    x_pred      = list(range(len(past) - 1, len(past) + len(preds)))
    past_bridge = past[-1:] + preds  # connect the two lines

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=x_past, y=past,
        name="Historical",
        mode="lines",
        line=dict(color=COLORS["hist"], width=2),
    ))

    # Confidence band (area)
    if upper_ci and lower_ci:
        x_ci    = list(range(len(past) - 1, len(past) + len(preds)))
        y_upper = [past[-1]] + upper_ci
        y_lower = [past[-1]] + lower_ci
        fig.add_trace(go.Scatter(
            x=x_ci + x_ci[::-1],
            y=y_upper + y_lower[::-1],
            fill="toself",
            fillcolor=COLORS["ci"],
            line=dict(color="rgba(0,0,0,0)"),
            name="90% Confidence",
            showlegend=True,
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=past_bridge,
        name=f"{model.upper()} Forecast",
        mode="lines+markers",
        line=dict(color=COLORS["lstm"] if model == "lstm" else COLORS["arima"],
                  width=2.5, dash="dot"),
        marker=dict(size=6),
    ))

    # Divider line
    fig.add_vline(x=len(past) - 1, line_dash="dash", line_color="#475569",
                  annotation_text="Today", annotation_font_color="#94a3b8")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{country.title()} — {model.upper()} Forecast", font=dict(size=16)),
        xaxis_title="Day Index",
        yaxis_title="Daily Cases",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 7-Day Forecast Cards ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">📅 Day-by-Day Forecast</p>', unsafe_allow_html=True)
    fd = future_dates(len(preds))
    cols = st.columns(min(len(preds), 7))
    for i, (col, val) in enumerate(zip(cols, preds)):
        change = val - preds[i - 1] if i > 0 else 0
        arrow  = "▲" if change > 0 else ("▼" if change < 0 else "→")
        color  = "#f87171" if change > 0 else ("#34d399" if change < 0 else "#94a3b8")
        with col:
            st.markdown(
                f"<div style='background:#1e293b;border:1px solid #334155;border-radius:12px;"
                f"padding:14px;text-align:center;'>"
                f"<div style='color:#64748b;font-size:11px;'>{fd[i]}</div>"
                f"<div style='color:#f1f5f9;font-size:20px;font-weight:700;margin:4px 0;'>"
                f"{val:,.0f}</div>"
                f"<div style='color:{color};font-size:12px;'>{arrow} {abs(change):,.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "date": fd,
        "predicted_cases": preds,
        "lower_ci": lower_ci or [None] * len(preds),
        "upper_ci": upper_ci or [None] * len(preds),
        "model": model.upper(),
        "country": country,
    })
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=df_to_csv(forecast_df),
        file_name=f"{country}_{model}_forecast_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">🤖 LSTM vs ARIMA — Head-to-Head</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">'
        'Both models are backtested on the last 30 historical days. '
        'Metrics are computed on out-of-sample data.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Running model comparison (this may take ~10s for ARIMA fitting)..."):
        cmp_data = get_compare(country, days)

    if cmp_data is None:
        st.info("Model comparison requires the backend to be running.")
        st.stop()

    lstm_block  = cmp_data.get("lstm",  {})
    arima_block = cmp_data.get("arima", {})
    comparison  = cmp_data.get("comparison", {})
    past_cmp    = cmp_data.get("past_30_days", [])

    lstm_preds  = lstm_block.get("predictions", [])
    arima_preds = arima_block.get("predictions", [])
    lstm_ci_l   = lstm_block.get("lower_ci", [])
    lstm_ci_u   = lstm_block.get("upper_ci", [])
    arima_ci_l  = arima_block.get("lower_ci", [])
    arima_ci_u  = arima_block.get("upper_ci", [])

    recommendation = comparison.get("recommendation", "LSTM")

    # ── Recommendation Banner ─────────────────────────────────────────────────
    winner_color = "#6366f1" if recommendation == "LSTM" else "#f59e0b"
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1e293b,#0f172a);"
        f"border:1px solid {winner_color};border-radius:12px;padding:16px 24px;"
        f"margin-bottom:20px;'>"
        f"<span style='color:{winner_color};font-weight:700;font-size:18px;'>"
        f"🏆 Recommended: {recommendation}</span>"
        f"<span style='color:#64748b;font-size:13px;margin-left:16px;'>"
        f"Based on majority vote across RMSE, MAE, MAPE backtest metrics</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Side-by-side forecast chart ───────────────────────────────────────────
    x_past = list(range(len(past_cmp)))
    x_pred = list(range(len(past_cmp) - 1, len(past_cmp) + len(lstm_preds)))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_past, y=past_cmp, name="Historical",
        mode="lines", line=dict(color=COLORS["hist"], width=2)
    ))

    # LSTM CI
    if lstm_ci_u and lstm_ci_l:
        y_u = [past_cmp[-1]] + lstm_ci_u
        y_l = [past_cmp[-1]] + lstm_ci_l
        fig2.add_trace(go.Scatter(
            x=x_pred + x_pred[::-1], y=y_u + y_l[::-1],
            fill="toself", fillcolor=COLORS["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="LSTM 90% CI"
        ))

    # ARIMA CI
    if arima_ci_u and arima_ci_l:
        y_u = [past_cmp[-1]] + arima_ci_u
        y_l = [past_cmp[-1]] + arima_ci_l
        fig2.add_trace(go.Scatter(
            x=x_pred + x_pred[::-1], y=y_u + y_l[::-1],
            fill="toself", fillcolor=COLORS["ci_arima"],
            line=dict(color="rgba(0,0,0,0)"), name="ARIMA 90% CI"
        ))

    # LSTM line
    fig2.add_trace(go.Scatter(
        x=x_pred, y=[past_cmp[-1]] + lstm_preds, name="LSTM Forecast",
        mode="lines+markers",
        line=dict(color=COLORS["lstm"], width=2.5, dash="dot"),
        marker=dict(size=6),
    ))
    # ARIMA line
    fig2.add_trace(go.Scatter(
        x=x_pred, y=[past_cmp[-1]] + arima_preds, name="ARIMA Forecast",
        mode="lines+markers",
        line=dict(color=COLORS["arima"], width=2.5, dash="dash"),
        marker=dict(size=6),
    ))

    fig2.add_vline(x=len(past_cmp) - 1, line_dash="dash", line_color="#475569")
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{country.title()} — LSTM vs ARIMA Forecast", font=dict(size=16)),
        yaxis_title="Daily Cases",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Backtest Metrics</p>', unsafe_allow_html=True)

    lstm_m  = comparison.get("lstm", {})
    arima_m = comparison.get("arima", {})
    winners = comparison.get("metric_winners", {})

    metrics_rows = [
        ("RMSE (lower = better)", "rmse", True),
        ("MAE  (lower = better)", "mae",  True),
        ("MAPE (lower = better)", "mape", True),
        ("R²   (higher = better)", "r2",  False),
    ]

    metric_table_html = "<div style='background:#1e293b;border-radius:12px;overflow:hidden;'>"
    for label, key, lower_better in metrics_rows:
        lv = lstm_m.get(key)
        av = arima_m.get(key)
        w  = winners.get(key, "N/A")
        metric_table_html += (
            f"<div class='metric-row'>"
            f"<span class='metric-name'>{label}</span>"
            f"<span style='display:flex;gap:24px;align-items:center;'>"
            f"<span class='metric-value' style='color:{COLORS['lstm']};'>"
            f"LSTM: {fmt(lv, 2)}</span>"
            f"<span class='metric-value' style='color:{COLORS['arima']};'>"
            f"ARIMA: {fmt(av, 2)}</span>"
            f"{'<span class=\"winner-tag\">Winner: ' + w + '</span>' if w != 'N/A' else ''}"
            f"</span></div>"
        )
    metric_table_html += "</div>"
    st.markdown(metric_table_html, unsafe_allow_html=True)

    # ── Download comparison ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fd_cmp = future_dates(max(len(lstm_preds), len(arima_preds)))
    cmp_df = pd.DataFrame({
        "date":            fd_cmp[:min(len(lstm_preds), len(arima_preds))],
        "lstm_forecast":   lstm_preds,
        "arima_forecast":  arima_preds,
        "country":         country,
    })
    st.download_button(
        "⬇️ Download Comparison CSV",
        data=df_to_csv(cmp_df),
        file_name=f"{country}_model_comparison_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<p class="section-header">📊 Data Explorer</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Feature-engineered dataset with rolling averages, growth rates, and epidemic phase.</p>', unsafe_allow_html=True)

    with col_r:
        with st.spinner("Loading feature data..."):
            feat_data = get_features(country)
            qual_data = get_quality(country)
            stat_data = get_stats(country)

    if feat_data is None:
        st.info("Data Explorer requires the backend to be running.")
        st.stop()

    feat_summary  = feat_data.get("feature_summary", {})
    feat_records  = feat_data.get("recent_features", [])
    quality       = qual_data.get("quality_report", {}) if qual_data else {}
    statistics    = stat_data.get("statistics",    {}) if stat_data else {}

    # ── Quality Score Card ────────────────────────────────────────────────────
    qs = quality.get("quality_score", 0)
    fresh = quality.get("data_freshness", "N/A")
    comp  = quality.get("completeness_pct", 0)

    st.markdown('<p class="section-header">🔍 Data Quality Report</p>', unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        st.markdown(kpi_card("Quality Score", f"{qs}/100"), unsafe_allow_html=True)
    with qc2:
        st.markdown(kpi_card("Completeness", f"{comp}%"), unsafe_allow_html=True)
    with qc3:
        st.markdown(kpi_card("Freshness", fresh), unsafe_allow_html=True)
    with qc4:
        outliers = quality.get("outliers_detected", 0)
        st.markdown(kpi_card("Outliers Corrected", str(outliers)), unsafe_allow_html=True)

    st.markdown(
        f"<div class='quality-bar-wrap'><div class='quality-bar-fill' style='width:{qs}%;'></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Descriptive Statistics ────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Descriptive Statistics</p>', unsafe_allow_html=True)
    if statistics:
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(kpi_card("Mean Daily Cases",   fmt(statistics.get("mean"))),   unsafe_allow_html=True)
        with sc2:
            st.markdown(kpi_card("Median Daily Cases", fmt(statistics.get("median"))), unsafe_allow_html=True)
        with sc3:
            st.markdown(kpi_card("Std Deviation",      fmt(statistics.get("std"))),    unsafe_allow_html=True)
        with sc4:
            st.markdown(kpi_card("Peak Cases (max)",   fmt(statistics.get("max"))),    unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature Table ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🧬 Feature-Engineered Dataset (Last 60 days)</p>', unsafe_allow_html=True)
    if feat_records:
        feat_df = pd.DataFrame(feat_records)
        if "growth_rate_7d" in feat_df.columns:
            feat_df["growth_rate_7d"] = (feat_df["growth_rate_7d"] * 100).round(2).astype(str) + "%"
        st.dataframe(
            feat_df,
            use_container_width=True,
            height=320,
        )

        # Epidemic phase over time chart
        raw_feat = feat_data.get("recent_features", [])
        if raw_feat:
            phase_df = pd.DataFrame(raw_feat)
            phase_map = {
                "Exponential Growth": 6, "Rising": 5, "Plateau": 4,
                "Declining": 3, "Rapid Decline": 2, "Low Activity": 1, "Unknown": 0,
            }
            if "epidemic_phase" in phase_df.columns and "cases" in phase_df.columns:
                phase_df["phase_num"] = phase_df["epidemic_phase"].map(phase_map).fillna(0)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=phase_df["date"], y=phase_df["cases"],
                    name="Daily Cases", mode="lines",
                    line=dict(color=COLORS["hist"], width=2),
                ))
                fig3.add_trace(go.Scatter(
                    x=phase_df["date"], y=phase_df.get("ma_7", phase_df["cases"]),
                    name="7-day MA", mode="lines",
                    line=dict(color=COLORS["lstm"], width=1.5, dash="dot"),
                ))
                fig3.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(
                        text=f"{country.title()} — Cases & 7-day Moving Average",
                        font=dict(size=16)
                    ),
                    yaxis_title="Daily Cases",
                )
                st.plotly_chart(fig3, use_container_width=True)

        # Download feature data
        raw_df_dl = pd.DataFrame(feat_data.get("recent_features", []))
        if not raw_df_dl.empty:
            st.download_button(
                "⬇️ Download Feature Dataset CSV",
                data=df_to_csv(raw_df_dl),
                file_name=f"{country}_features_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    # ── Feature Summary KPIs ──────────────────────────────────────────────────
    if feat_summary:
        st.markdown("---")
        st.markdown('<p class="section-header">⚡ Latest Feature Snapshot</p>', unsafe_allow_html=True)
        fs1, fs2, fs3, fs4 = st.columns(4)
        with fs1:
            st.markdown(kpi_card("7-day MA",  fmt(feat_summary.get("ma_7"))),   unsafe_allow_html=True)
        with fs2:
            st.markdown(kpi_card("14-day MA", fmt(feat_summary.get("ma_14"))),  unsafe_allow_html=True)
        with fs3:
            gr = feat_summary.get("growth_rate_7d", 0)
            st.markdown(kpi_card("7d Growth Rate", f"{gr:+.1f}%",
                         "vs prior week", "up" if gr > 0 else "down"), unsafe_allow_html=True)
        with fs4:
            dt = feat_summary.get("doubling_time")
            st.markdown(kpi_card("Doubling Time", f"{dt} days" if dt else "N/A"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Multi-Country
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">🌍 Multi-Country Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">'
        'Compare 7-day LSTM forecasts across multiple countries simultaneously.'
        '</p>',
        unsafe_allow_html=True,
    )

    all_countries = get_countries()
    selected = st.multiselect(
        "Select countries to compare",
        all_countries,
        default=["india", "usa", "brazil", "uk"],
        max_selections=6,
    )

    if not selected:
        st.info("Select at least one country above.")
        st.stop()

    mc_palette = ["#6366f1", "#f59e0b", "#34d399", "#f87171", "#60a5fa", "#a78bfa"]

    compare_days = st.radio("Forecast horizon", [7, 14, 30], horizontal=True)

    fig4 = go.Figure()
    bar_data = []

    with st.spinner(f"Fetching forecasts for {len(selected)} countries..."):
        for i, c in enumerate(selected):
            mc_data = api("predict", country=c, days=compare_days, model="lstm")
            if mc_data:
                p = mc_data.get("predictions", [])
                if p:
                    avg_pred = round(np.mean(p), 2)
                    bar_data.append({"country": c.title(), "avg_forecast": avg_pred})
                    fig4.add_trace(go.Scatter(
                        x=list(range(1, len(p) + 1)),
                        y=p,
                        name=c.title(),
                        mode="lines+markers",
                        line=dict(color=mc_palette[i % len(mc_palette)], width=2),
                        marker=dict(size=5),
                    ))

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{compare_days}-Day LSTM Forecast — Multi-Country",
            font=dict(size=16)
        ),
        xaxis_title="Day",
        yaxis_title="Predicted Daily Cases",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Bar chart — average predicted cases
    if bar_data:
        st.markdown('<p class="section-header">📊 Average Predicted Daily Cases</p>', unsafe_allow_html=True)
        bar_df = pd.DataFrame(bar_data).sort_values("avg_forecast", ascending=True)
        fig5 = go.Figure(go.Bar(
            x=bar_df["avg_forecast"],
            y=bar_df["country"],
            orientation="h",
            marker=dict(
                color=mc_palette[:len(bar_df)],
                line=dict(color="rgba(0,0,0,0)"),
            ),
        ))
        fig5.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Avg Predicted Daily Cases by Country", font=dict(size=15)),
            xaxis_title="Avg Daily Cases",
            yaxis_title="",
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Download multi-country data
        mc_dl = pd.DataFrame(bar_data)
        st.download_button(
            "⬇️ Download Multi-Country Summary CSV",
            data=df_to_csv(mc_dl),
            file_name=f"multi_country_forecast_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
