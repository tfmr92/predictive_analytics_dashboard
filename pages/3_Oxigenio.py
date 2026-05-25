"""
Oxygen — Crew Oxygen System monitoring.
Two views: absolute PSI trend over time + aircraft below alert threshold.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Oxygen System", layout="wide")

st.title("💨 Crew Oxygen System")
st.markdown(
    "Monitors oxygen cylinder pressure on each aircraft. "
    "A **faster-than-expected pressure drop** or pressure **below minimum** requires maintenance before next flight."
)

# ── Data ──────────────────────────────────────────────────────────────────────
df = load("e2_oxy_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_oxy_report` job in Dagster.")
    st.stop()

AC_COL = next((c for c in ("aircraftSerNum-1", "ac_sn") if c in df.columns), None)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
if AC_COL:
    df[AC_COL] = df[AC_COL].astype(str)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 14, 365, 90)
    all_ac = sorted(df[AC_COL].dropna().unique().tolist()) if AC_COL else []
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=all_ac)

    # Alert threshold — absolute PSI
    PSI_MIN = st.number_input(
        "Min safe PSI threshold", min_value=500, max_value=2000, value=1800, step=50,
        help=(
            "Pressure below this level triggers an alert. "
            "Typical E2 crew oxygen minimum before required maintenance is ~1800 PSI. "
            "Adjust per your Aircraft Maintenance Manual."
        ),
    )

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

if selected_ac and AC_COL:
    df = df[df[AC_COL].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Derived alert using absolute PSI ─────────────────────────────────────────
if "psi" in df.columns:
    df["psi_alert"] = df["psi"] < PSI_MIN
elif "delta_press" in df.columns:
    # Fallback: statistical threshold on daily pressure drop
    _limiar = df["delta_press"].mean() + df["delta_press"].std()
    df["psi_alert"] = df.get("alert", df["delta_press"] > _limiar)

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_alert = 0
n_total = 0
if AC_COL and "psi_alert" in df.columns:
    latest = df.sort_values("date").groupby(AC_COL).last()
    n_alert = int(latest["psi_alert"].sum())
    n_total = len(latest)

avg_psi = df["psi"].mean() if "psi" in df.columns else None
avg_drop = df["delta_press"].mean() if "delta_press" in df.columns else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Aircraft below threshold", n_alert,
          help=f"Latest reading < {PSI_MIN} PSI")
c2.metric("✅ Aircraft above threshold", n_total - n_alert if n_total else 0)
if avg_psi is not None:
    c3.metric("Fleet avg PSI (period)", f"{avg_psi:.0f}")
if avg_drop is not None:
    c4.metric("Avg daily pressure drop (PSI)", f"{avg_drop:.1f}")

st.divider()

# ── Chart 1: Absolute PSI over time ───────────────────────────────────────────
st.subheader("1. Oxygen Pressure (PSI) Over Time")
st.caption(
    "Each line is one aircraft. "
    f"The red dashed line marks the {PSI_MIN} PSI minimum threshold. "
    "A downward trend heading toward the threshold requires scheduling."
)

if "psi" in df.columns and AC_COL:
    df_psi = df.dropna(subset=["psi", AC_COL]).copy()

    fig_psi = px.line(
        df_psi, x="date", y="psi",
        color=AC_COL,
        labels={"psi": "Pressure (PSI)", "date": "", AC_COL: "MSN"},
        title="Crew Oxygen Pressure — Absolute PSI",
    )
    fig_psi.add_hline(
        y=PSI_MIN, line_dash="dash", line_color="red",
        annotation_text=f"Min threshold ({PSI_MIN} PSI)",
        annotation_position="top right",
    )
    fig_psi.update_layout(
        height=400,
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
    )
    st.plotly_chart(fig_psi, use_container_width=True)
elif "delta_press" in df.columns and AC_COL:
    st.info(
        "Absolute PSI column (`psi`) not present — showing daily pressure drop instead. "
        "Re-run `save_oxy_report` to populate the PSI column."
    )

st.divider()

# ── Chart 2: Aircraft below alert level ───────────────────────────────────────
st.subheader("2. Aircraft Status — Latest Reading vs. Threshold")
st.caption(
    "Shows each aircraft's most recent pressure reading. "
    "Bars in red are below the minimum threshold and need attention."
)

status_col = "psi" if "psi" in df.columns else ("delta_press" if "delta_press" in df.columns else None)

if status_col and AC_COL:
    latest_status = (
        df.dropna(subset=[AC_COL, status_col])
        .sort_values("date")
        .groupby(AC_COL)
        .last()[[status_col]]
        .reset_index()
        .sort_values(status_col)
    )

    if status_col == "psi":
        latest_status["Status"] = latest_status[status_col].apply(
            lambda x: "⚠️ Below threshold" if x < PSI_MIN else "✅ Normal"
        )
        latest_status["color"] = latest_status[status_col].apply(
            lambda x: "#ef4444" if x < PSI_MIN else "#22c55e"
        )
        x_label = "Pressure (PSI)"
        threshold_val = PSI_MIN
        threshold_label = f"Threshold ({PSI_MIN} PSI)"
        threshold_type = "vline"
    else:
        _thresh = df["delta_press"].mean() + df["delta_press"].std()
        latest_status["Status"] = latest_status[status_col].apply(
            lambda x: "⚠️ High drop" if x > _thresh else "✅ Normal"
        )
        latest_status["color"] = latest_status[status_col].apply(
            lambda x: "#ef4444" if x > _thresh else "#22c55e"
        )
        x_label = "Daily Pressure Drop (PSI)"
        threshold_val = _thresh
        threshold_label = "Alert threshold"
        threshold_type = "vline"

    fig_status = go.Figure(go.Bar(
        y=latest_status[AC_COL].astype(str),
        x=latest_status[status_col],
        orientation="h",
        marker_color=latest_status["color"],
        text=latest_status["Status"],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    if threshold_type == "vline":
        fig_status.add_vline(
            x=threshold_val, line_dash="dot", line_color="red",
            annotation_text=threshold_label, annotation_position="top right",
        )
    fig_status.update_layout(
        title="Latest pressure reading per aircraft",
        xaxis_title=x_label,
        yaxis_title="MSN",
        height=max(300, len(latest_status) * 30),
        margin=dict(l=10, r=80, t=40, b=10),
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.divider()

# ── Chart 3: Daily pressure drop trend ────────────────────────────────────────
st.subheader("3. Daily Pressure Drop — Trend per Aircraft")
st.caption(
    "A rising trend in daily pressure drop indicates accelerating leakage. "
    "Aircraft with consistent high drops should be inspected."
)

if "delta_press" in df.columns and AC_COL:
    df_drop = df.dropna(subset=["delta_press", AC_COL]).copy()
    _alert_line = df_drop["delta_press"].mean() + df_drop["delta_press"].std()

    fig_drop = px.scatter(
        df_drop, x="date", y="delta_press",
        color=AC_COL,
        trendline="lowess",
        trendline_scope="overall",
        trendline_color_override="black",
        opacity=0.5,
        labels={"delta_press": "Pressure Drop (PSI/day)", "date": "", AC_COL: "MSN"},
        title="Daily Oxygen Pressure Drop",
    )
    fig_drop.add_hline(
        y=_alert_line, line_dash="dot", line_color="orange",
        annotation_text="Fleet alert level (mean + 1σ)",
        annotation_position="top right",
    )
    fig_drop.update_traces(selector=dict(mode="markers"), marker_size=5)
    fig_drop.update_layout(
        height=360,
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
    )
    st.plotly_chart(fig_drop, use_container_width=True)

# ── Top pressure-drop events table ────────────────────────────────────────────
if "delta_press" in df.columns and "date" in df.columns:
    top_cols = ["date", AC_COL, "delta_press"] + (["psi"] if "psi" in df.columns else [])
    top_cols = [c for c in top_cols if c in df.columns]
    top = (
        df.nlargest(15, "delta_press")[top_cols]
        .rename(columns={
            "date": "Date",
            AC_COL: "MSN",
            "delta_press": "Pressure Drop (PSI)",
            "psi": "Current PSI",
        })
    )
    st.caption("Top 15 highest single-day pressure drop events")
    st.dataframe(top, use_container_width=True, hide_index=True)
