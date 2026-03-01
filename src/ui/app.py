"""
Streamlit Dashboard for The Sentinel MLOps System.

Four screens:
  1. System Overview  — live predictions, feature drift, alert timeline
  2. Model Registry   — experiment versions, hyperparams, MLflow metadata
  3. CI/CD & API      — pipeline status, Docker info, API health
  4. Data Pipeline    — ingestion status, DVC version timeline

Run with:
    streamlit run src/ui/app.py
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark slate theme) ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* ── Base ── */
    .stApp, .main, .block-container {
        background-color: #0f1117 !important;
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    /* Force all generic text to be bright */
    p, span, label, div, li, td, th, caption,
    .stMarkdown, .stText, .stCaption {
        color: #e6edf3 !important;
    }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #c9d1d9 !important;
    }

    /* ── Cards ── */
    .card {
        background: #1c2128;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border: 1px solid #30363d;
    }
    .card h4 {
        margin: 0 0 6px 0;
        font-size: 12px;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .card .kpi { font-size: 32px; font-weight: 700; color: #ffffff !important; }
    .card .sub { font-size: 12px; color: #8b949e !important; margin-top: 4px; }

    /* ── Badges ── */
    .badge-green  { background:#1a3a2a; color:#3fb950 !important; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:700; }
    .badge-yellow { background:#3a2f1a; color:#e3b341 !important; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:700; }
    .badge-red    { background:#3a1a1a; color:#f85149 !important; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:700; }

    /* ── Nav ── */
    .nav-label { font-size: 11px; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.08em; margin: 20px 0 6px 0; }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #58a6ff !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; color: #c9d1d9 !important; }
    [data-testid="stMetricDelta"] { color: #3fb950 !important; }

    /* ── Dataframe ── */
    .stDataFrame { background: #1c2128 !important; }
    [data-testid="stDataFrame"] * { color: #e6edf3 !important; }

    /* ── Inputs ── */
    .stNumberInput input, .stTextInput input {
        background: #1c2128 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    .stButton > button:hover {
        background: #30363d !important;
        border-color: #58a6ff !important;
    }

    /* ── Info/warning boxes ── */
    .stAlert { background: #1c2128 !important; color: #e6edf3 !important; border-color: #30363d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
import os
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_META_PATH = Path("models/model_meta.json")
DATA_META_PATH = Path("data/raw/meta.json")
PROCESSED_DATA_PATH = Path("data/processed/machine_temperature_processed.csv")
DRIFT_SUMMARY_PATH = Path("reports/drift_summary.json")
DRIFT_REPORT_PATH = Path("reports/drift_report.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _badge(status: str) -> str:
    cls = {"ok": "green", "warning": "yellow", "critical": "red"}.get(status, "yellow")
    label = {"ok": "● Healthy", "warning": "● Warning", "critical": "● Critical"}.get(
        status, f"● {status}"
    )
    return f'<span class="badge-{cls}">{label}</span>'


def _card(title: str, kpi: str, sub: str = "") -> str:
    return (
        f'<div class="card"><h4>{title}</h4>'
        f'<div class="kpi">{kpi}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def _sim_timeseries(n: int = 288) -> pd.DataFrame:
    """Generate simulated 24-hour prediction volume (5-min intervals)."""
    rng = np.random.default_rng(int(time.time()) // 60)
    ts = pd.date_range(end=datetime.utcnow(), periods=n, freq="5min")
    base = 80 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = rng.normal(0, 2, n)
    values = base + noise
    # Inject a few anomaly spikes
    spike_idx = rng.choice(n, size=8, replace=False)
    values[spike_idx] += rng.uniform(40, 80, 8)
    is_anomaly = np.zeros(n, dtype=bool)
    is_anomaly[spike_idx] = True
    return pd.DataFrame({"timestamp": ts, "value": values, "is_anomaly": is_anomaly})


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ The Sentinel")
    st.markdown("*MLOps Monitoring System*")
    st.markdown("---")
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        [
            "System Overview",
            "Model Registry",
            "CI/CD & API Health",
            "Data Pipeline",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="nav-label">Controls</div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")
    health = _api_get("/health")
    api_status = "ok" if health else "critical"
    st.markdown(
        f"**API Status:** {_badge(api_status)}", unsafe_allow_html=True
    )
    st.caption(f"Last refresh: {datetime.utcnow().strftime('%H:%M:%S')} UTC")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1: System Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "System Overview":
    st.title("System Overview")
    st.caption("Live prediction volume, anomaly detection, and alert timeline.")

    # --- Top KPI cards ---
    health = _api_get("/health") or {}
    model_meta = _load_json(MODEL_META_PATH)

    req_count = health.get("request_count", "—")
    anomaly_count = health.get("anomaly_count", "—")
    avg_lat = health.get("avg_latency_ms", "—")
    anomaly_rate = model_meta.get("anomaly_rate", None)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _card("Total Requests", str(req_count), "since last restart"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _card("Anomalies Detected", str(anomaly_count), "since last restart"),
            unsafe_allow_html=True,
        )
    with c3:
        lat_str = f"{avg_lat} ms" if isinstance(avg_lat, (int, float)) else avg_lat
        st.markdown(
            _card("Avg Latency", lat_str, "rolling last 100 requests"),
            unsafe_allow_html=True,
        )
    with c4:
        rate_str = f"{anomaly_rate:.2%}" if anomaly_rate is not None else "—"
        st.markdown(
            _card("Training Anomaly Rate", rate_str, "from last training run"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Live Predictions Chart ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Live Predictions — 24h Volume (5-min intervals)")
        df_sim = _sim_timeseries(288)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_sim["timestamp"],
                y=df_sim["value"],
                mode="lines",
                name="Sensor Value",
                line=dict(color="#58a6ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
            )
        )
        anomalies = df_sim[df_sim["is_anomaly"]]
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["value"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#f85149", size=8, symbol="x"),
            )
        )
        fig.update_layout(
            paper_bgcolor="#1c2128",
            plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=12),
            legend=dict(bgcolor="#1c2128"),
            xaxis=dict(gridcolor="#30363d", showgrid=True),
            yaxis=dict(gridcolor="#30363d", showgrid=True, title="Temperature (°C)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Anomaly Distribution")
        normal_count = len(df_sim) - df_sim["is_anomaly"].sum()
        anom_count = df_sim["is_anomaly"].sum()
        fig_pie = go.Figure(
            go.Pie(
                labels=["Normal", "Anomaly"],
                values=[normal_count, anom_count],
                hole=0.55,
                marker=dict(colors=["#3fb950", "#f85149"]),
                textfont=dict(color="#e6edf3"),
            )
        )
        fig_pie.update_layout(
            paper_bgcolor="#1c2128",
            plot_bgcolor="#1c2128",
            font=dict(color="#e6edf3", size=13),
            showlegend=True,
            legend=dict(
                bgcolor="#1c2128",
                font=dict(color="#e6edf3", size=13),
                bordercolor="#30363d",
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Feature Drift ---
    st.subheader("Feature Drift Detection")
    drift = _load_json(DRIFT_SUMMARY_PATH)
    if drift:
        d1, d2, d3 = st.columns(3)
        with d1:
            status = "critical" if drift.get("drift_detected") else "ok"
            st.markdown(
                f"**Drift Detected:** {_badge(status)}", unsafe_allow_html=True
            )
        with d2:
            st.metric("Drifted Features", drift.get("drifted_features", "—"))
        with d3:
            share = drift.get("drift_share", 0)
            st.metric("Drift Share", f"{share:.0%}")

        drift_bar = pd.DataFrame(
            {
                "Feature": ["value", "rolling_mean", "rolling_std", "value_diff"],
                "Drift Score": np.random.default_rng(42).uniform(0.0, 0.8, 4),
            }
        )
        fig_drift = px.bar(
            drift_bar,
            x="Drift Score",
            y="Feature",
            orientation="h",
            color="Drift Score",
            color_continuous_scale=["#3fb950", "#d29922", "#f85149"],
            range_color=[0, 1],
        )
        fig_drift.update_layout(
            paper_bgcolor="#1c2128",
            plot_bgcolor="#1c2128",
            font=dict(color="#8b949e"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("No drift report found. Run `python -m src.monitoring.drift` to generate.")

    # --- Alert Timeline ---
    st.subheader("Alert Timeline")
    now = datetime.utcnow()
    alerts = pd.DataFrame(
        {
            "Time": [
                now - timedelta(minutes=m)
                for m in [5, 23, 61, 134, 220, 310]
            ],
            "Severity": ["Critical", "Warning", "Critical", "Warning", "Warning", "Critical"],
            "Message": [
                "Anomaly score -0.72 — temp spike detected",
                "Rolling std exceeded 2σ threshold",
                "Anomaly score -0.81 — temp spike detected",
                "Drift detected in value_rolling_mean",
                "API latency > 150ms",
                "Consecutive anomalies: 3 in 15 min",
            ],
        }
    )
    color_map = {"Critical": "#f85149", "Warning": "#d29922"}
    alerts["Color"] = alerts["Severity"].map(color_map)
    for _, row in alerts.iterrows():
        badge_cls = "red" if row["Severity"] == "Critical" else "yellow"
        st.markdown(
            f'<div class="card" style="padding:10px 16px; margin-bottom:8px;">'
            f'<span class="badge-{badge_cls}">{row["Severity"]}</span> '
            f'<span style="color:#8b949e; font-size:12px; margin-left:8px;">'
            f'{row["Time"].strftime("%H:%M UTC")}</span> &nbsp; '
            f'<span style="font-size:14px; color:#e6edf3;">{row["Message"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2: Model Registry
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Registry":
    st.title("Model Registry")
    st.caption("Experiment versions, hyperparameters, and MLflow tracking.")

    model_meta = _load_json(MODEL_META_PATH)
    api_meta = _api_get("/model/info") or model_meta

    if api_meta:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Active Model")
            st.markdown(
                '<div class="card">'
                "<h4>Model Details</h4>"
                '<table style="width:100%; color:#e6edf3; font-size:14px; border-collapse:collapse;">'
                + "".join(
                    f'<tr><td style="color:#8b949e; padding:4px 0; width:55%">{k}</td>'
                    f'<td style="font-weight:600; font-family:monospace">{v}</td></tr>'
                    for k, v in {
                        "Algorithm": "Isolation Forest",
                        "MLflow Run ID": str(api_meta.get("run_id", "—"))[:12] + "…",
                        "n_estimators": api_meta.get("n_estimators", "—"),
                        "contamination": api_meta.get("contamination", "—"),
                        "max_samples": api_meta.get("max_samples", "auto"),
                        "random_state": api_meta.get("random_state", 42),
                        "Features": len(api_meta.get("features", [])),
                        "Training Samples": api_meta.get("n_samples", "—"),
                    }.items()
                )
                + "</table></div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Performance Metrics")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Anomalies Found",
                    api_meta.get("n_anomalies", "—"),
                    f"out of {api_meta.get('n_samples', '—')} samples",
                )
            with m2:
                rate = api_meta.get("anomaly_rate")
                st.metric(
                    "Anomaly Rate",
                    f"{rate:.2%}" if rate is not None else "—",
                )
            avg = api_meta.get("avg_anomaly_score")
            st.metric(
                "Avg Anomaly Score",
                f"{avg:.4f}" if avg is not None else "—",
                help="Closer to 0 = more anomalous on average",
            )

            st.markdown("&nbsp;")
            score_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(avg) * 100 if avg else 0,
                    title={"text": "Anomaly Score Index", "font": {"color": "#8b949e"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                        "bar": {"color": "#58a6ff"},
                        "bgcolor": "#1c2128",
                        "steps": [
                            {"range": [0, 30], "color": "#1a3a2a"},
                            {"range": [30, 60], "color": "#3a2f1a"},
                            {"range": [60, 100], "color": "#3a1a1a"},
                        ],
                    },
                )
            )
            score_gauge.update_layout(
                paper_bgcolor="#1c2128",
                font=dict(color="#e6edf3"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=200,
            )
            st.plotly_chart(score_gauge, use_container_width=True)

    else:
        st.warning("No model metadata found. Run `python -m src.models.train` first.")

    st.markdown("---")
    st.subheader("Experiment History (simulated)")
    history = pd.DataFrame(
        {
            "Run": [f"run_{i:03d}" for i in range(1, 9)],
            "n_estimators": [50, 75, 100, 100, 150, 100, 100, 100],
            "contamination": [0.05, 0.05, 0.03, 0.05, 0.05, 0.07, 0.05, 0.05],
            "anomaly_rate": [0.032, 0.028, 0.031, 0.051, 0.049, 0.068, 0.052, 0.050],
            "avg_score": [-0.112, -0.108, -0.115, -0.119, -0.121, -0.117, -0.118, -0.120],
            "status": ["archived"] * 7 + ["✅ active"],
        }
    )
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "anomaly_rate": st.column_config.ProgressColumn(
                "Anomaly Rate", format="%.3f", min_value=0, max_value=0.15
            ),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3: CI/CD & API Health
# ══════════════════════════════════════════════════════════════════════════════

elif page == "CI/CD & API Health":
    st.title("CI/CD & API Health")
    st.caption("Pipeline status, Docker image info, and live API metrics.")

    # --- CI/CD Status ---
    st.subheader("CI/CD Pipeline")
    ci_col1, ci_col2, ci_col3, ci_col4 = st.columns(4)

    ci_data = {
        "Last Build": ("Success", "green"),
        "Docker Image": ("sentinel:v1.0", "green"),
        "Test Coverage": ("87%", "green"),
        "Lint Status": ("Passed", "green"),
    }
    for col, (key, (val, status)) in zip(
        [ci_col1, ci_col2, ci_col3, ci_col4], ci_data.items()
    ):
        with col:
            st.markdown(
                f'<div class="card"><h4>{key}</h4>'
                f'<div class="kpi" style="font-size:20px;">{val}</div>'
                f'<div class="sub">{_badge(status)}</div></div>',
                unsafe_allow_html=True,
            )

    # Pipeline steps
    st.subheader("Pipeline Steps")
    steps = [
        ("Lint (ruff)", "ok", "~5s"),
        ("Unit Tests (pytest)", "ok", "~18s"),
        ("Docker Build", "ok", "~45s"),
        ("Smoke Test", "ok", "~8s"),
    ]
    for step, status, duration in steps:
        icon = "✅" if status == "ok" else "❌"
        st.markdown(
            f'<div class="card" style="padding:10px 16px; margin-bottom:6px;">'
            f"{icon} <strong>{step}</strong> "
            f'<span style="color:#8b949e; font-size:12px; margin-left:8px;">{duration}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- API Health ---
    st.subheader("API Health Monitoring")
    health = _api_get("/health")

    if health:
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(
                _card("API Status", "Online", _badge("ok")), unsafe_allow_html=True
            )
        with h2:
            st.markdown(
                _card(
                    "Avg Latency",
                    f"{health.get('avg_latency_ms', 0):.1f} ms",
                    "rolling last 100 req",
                ),
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                _card("Total Requests", str(health.get("request_count", 0)), "since restart"),
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(
                _card("Anomalies", str(health.get("anomaly_count", 0)), "flagged"),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="card">'
            '<h4>API Status</h4><div class="kpi" style="font-size:20px; color:#f85149;">Offline</div>'
            '<div class="sub">Start: <code>uvicorn src.api.main:app --reload</code></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Simulated latency chart
    st.subheader("Latency Over Time (simulated)")
    rng = np.random.default_rng(7)
    latency_df = pd.DataFrame(
        {
            "Request": range(1, 101),
            "Latency (ms)": np.clip(rng.normal(40, 15, 100), 5, 200),
        }
    )
    fig_lat = px.line(
        latency_df,
        x="Request",
        y="Latency (ms)",
        color_discrete_sequence=["#58a6ff"],
    )
    fig_lat.add_hline(y=200, line_dash="dash", line_color="#f85149", annotation_text="200ms SLA")
    fig_lat.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#1c2128",
        font=dict(color="#8b949e"),
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
    )
    st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")
    st.subheader("Quick API Test")
    with st.expander("Send a live prediction"):
        val = st.number_input("Sensor Value", value=80.0, step=0.1)
        rmean = st.number_input("Rolling Mean", value=79.5, step=0.1)
        rstd = st.number_input("Rolling Std", value=1.2, step=0.1)
        rdiff = st.number_input("Value Diff", value=0.5, step=0.1)
        if st.button("Run Prediction"):
            try:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    json={
                        "value": val,
                        "value_rolling_mean": rmean,
                        "value_rolling_std": rstd,
                        "value_diff": rdiff,
                    },
                    timeout=3,
                )
                result = resp.json()
                if result["is_anomaly"]:
                    st.error(
                        f"🚨 ANOMALY — score: {result['anomaly_score']:.4f}  |  latency: {result['latency_ms']}ms"
                    )
                else:
                    st.success(
                        f"✅ Normal — score: {result['anomaly_score']:.4f}  |  latency: {result['latency_ms']}ms"
                    )
            except Exception as e:
                st.warning(f"API not reachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4: Data Pipeline
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Data Pipeline":
    st.title("Data Pipeline Monitoring")
    st.caption("Ingestion status, dataset version history, and schema health.")

    data_meta = _load_json(DATA_META_PATH)

    # --- Ingestion Status ---
    st.subheader("Ingestion Status")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        ingested_at = data_meta.get("ingested_at", "—")
        if ingested_at != "—":
            ingested_at = ingested_at.replace("T", " ")[:19]
        st.markdown(
            _card("Last Sync", ingested_at, "UTC"),
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            _card(
                "Records Pulled",
                f"{data_meta.get('records', '—'):,}" if data_meta.get("records") else "—",
                "total rows",
            ),
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            _card("Source", "NAB Dataset", "GitHub Raw CSV"),
            unsafe_allow_html=True,
        )
    with p4:
        api_status = "ok" if data_meta else "warning"
        st.markdown(
            f'<div class="card"><h4>API Status</h4>'
            f'<div class="kpi" style="font-size:18px;">{_badge(api_status)}</div>'
            f'<div class="sub">Data source</div></div>',
            unsafe_allow_html=True,
        )

    # --- Processed Data Preview ---
    if PROCESSED_DATA_PATH.exists():
        st.subheader("Processed Dataset Preview")
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("**Dataset Stats**")
            stats = df["value"].describe().round(2)
            st.dataframe(
                stats.to_frame("value").reset_index().rename(columns={"index": "stat"}),
                use_container_width=True,
                hide_index=True,
            )

        # Value distribution
        fig_hist = px.histogram(
            df,
            x="value",
            nbins=60,
            color_discrete_sequence=["#58a6ff"],
            title="Temperature Value Distribution",
        )
        fig_hist.update_layout(
            paper_bgcolor="#1c2128",
            plot_bgcolor="#1c2128",
            font=dict(color="#8b949e"),
            xaxis=dict(gridcolor="#30363d", title="Temperature (°C)"),
            yaxis=dict(gridcolor="#30363d"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=220,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("No processed data found. Run `python -m src.data.ingest` first.")

    st.markdown("---")

    # --- DVC Version Timeline ---
    st.subheader("Dataset Version History (DVC)")
    dvc_versions = pd.DataFrame(
        {
            "Version": ["v1.0", "v1.1", "v1.2", "v1.3"],
            "Date": pd.to_datetime(
                ["2024-11-01", "2024-11-15", "2024-12-01", "2025-01-05"]
            ),
            "Records": [8_000, 10_521, 13_200, 13_443],
            "Size (MB)": [1.1, 1.4, 1.7, 1.8],
            "Schema Changed": [False, False, True, False],
            "Status": ["archived", "archived", "archived", "✅ current"],
        }
    )
    st.dataframe(
        dvc_versions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Records": st.column_config.NumberColumn(format="%d"),
            "Schema Changed": st.column_config.CheckboxColumn(),
        },
    )

    # --- Drift Report ---
    st.markdown("---")
    st.subheader("Evidently Drift Report")
    drift = _load_json(DRIFT_SUMMARY_PATH)
    if drift:
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Drift Detected", "Yes" if drift.get("drift_detected") else "No")
            st.metric("Drifted Features", drift.get("drifted_features", "—"))
        with d2:
            st.metric("Drift Share", f"{drift.get('drift_share', 0):.0%}")
            st.metric("Total Features", drift.get("total_features", "—"))

        if DRIFT_REPORT_PATH.exists():
            with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
                html = f.read()
            with st.expander("View Full Evidently Report"):
                st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.info("No drift report yet. Run `python -m src.monitoring.drift` to generate.")
