"""
Oxygen — Crew Oxygen System monitoring.
Real AMM thresholds from MTM-0051-00-Vol16_E2.35.L3.PDF:
  - 1155 PSI: minimum dispatch (OBSERVER OXY LO PRESS — cyan CAS)
  - 845 PSI:  CREW OXY LO PRESS — amber CAS (do not dispatch)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Oxygen System", layout="wide")

# ── AMM limits (ATA 35, MTM-0051-00-Vol16) ────────────────────────────────────
PSI_AMBER  = 845   # CREW OXY LO PRESS — amber CAS, do not dispatch
PSI_CYAN   = 1155  # OBSERVER OXY LO PRESS — cyan CAS, reduced capability
PSI_CHARGE = 1850  # Max cylinder charge (full)

st.title("💨 Crew Oxygen System — ATA 35")
st.markdown(
    "Monitors oxygen cylinder pressure across the fleet using real AMM thresholds. "
    "Two CAS alert levels apply: **OBSERVER OXY LO PRESS** (cyan, below 1,155 PSI) "
    "and **CREW OXY LO PRESS** (amber, below 845 PSI — no dispatch)."
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

    st.divider()
    st.subheader("AMM Thresholds (ATA 35)")
    st.metric("Amber — CREW OXY LO PRESS", f"{PSI_AMBER} PSI",
              help="Below this level: no dispatch. QRH action required.")
    st.metric("Cyan — OBSERVER OXY LO PRESS", f"{PSI_CYAN} PSI",
              help="Below this level: observer may not have full oxygen supply.")
    st.metric("Max cylinder charge", f"{PSI_CHARGE} PSI")

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

if selected_ac and AC_COL:
    df = df[df[AC_COL].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Classify by alert level ────────────────────────────────────────────────────
def _alert_level(psi):
    if psi < PSI_AMBER:
        return "🔴 CREW OXY LO PRESS (Amber)"
    elif psi < PSI_CYAN:
        return "🟡 OBSERVER OXY LO PRESS (Cyan)"
    else:
        return "🟢 Normal"

def _alert_color(psi):
    if psi < PSI_AMBER:
        return "#ef4444"
    elif psi < PSI_CYAN:
        return "#f59e0b"
    else:
        return "#22c55e"

if "psi" in df.columns:
    df["alert_level"] = df["psi"].apply(_alert_level)

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_amber = n_cyan = n_normal = n_total = 0
avg_psi = df["psi"].mean() if "psi" in df.columns else None

if AC_COL and "psi" in df.columns:
    latest = df.sort_values("date").groupby(AC_COL).last()
    n_total  = len(latest)
    n_amber  = int((latest["psi"] < PSI_AMBER).sum())
    n_cyan   = int(((latest["psi"] >= PSI_AMBER) & (latest["psi"] < PSI_CYAN)).sum())
    n_normal = int((latest["psi"] >= PSI_CYAN).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 CREW OXY LO PRESS", n_amber,
          help=f"Latest PSI < {PSI_AMBER} — amber CAS, no dispatch")
c2.metric("🟡 OBSERVER OXY LO PRESS", n_cyan,
          help=f"Latest PSI {PSI_AMBER}–{PSI_CYAN - 1} — cyan CAS")
c3.metric("🟢 Normal", n_normal,
          help=f"Latest PSI ≥ {PSI_CYAN}")
if avg_psi is not None:
    c4.metric("Fleet avg PSI (period)", f"{avg_psi:.0f}")

st.divider()

# ── Chart 1: Absolute PSI over time ───────────────────────────────────────────
st.subheader("1. Oxygen Pressure (PSI) Over Time")
st.caption(
    "Each line represents one aircraft. "
    "The **amber line** (845 PSI) is the CREW OXY LO PRESS limit — no dispatch below this. "
    "The **yellow line** (1,155 PSI) triggers OBSERVER OXY LO PRESS (cyan CAS)."
)

if "psi" in df.columns and AC_COL:
    df_psi = df.dropna(subset=["psi", AC_COL]).copy()

    fig_psi = px.line(
        df_psi, x="date", y="psi",
        color=AC_COL,
        labels={"psi": "Pressure (PSI)", "date": "", AC_COL: "MSN"},
        title="Crew Oxygen Pressure — Absolute PSI",
    )

    # Amber zone: below 845
    fig_psi.add_hrect(
        y0=0, y1=PSI_AMBER, fillcolor="rgba(239,68,68,0.08)",
        line_width=0, annotation_text="No dispatch zone", annotation_position="top left",
    )
    # Cyan zone: 845–1155
    fig_psi.add_hrect(
        y0=PSI_AMBER, y1=PSI_CYAN, fillcolor="rgba(245,158,11,0.08)",
        line_width=0, annotation_text="Reduced capability", annotation_position="top left",
    )

    fig_psi.add_hline(
        y=PSI_AMBER, line_dash="dash", line_color="#ef4444",
        annotation_text=f"CREW OXY LO PRESS ({PSI_AMBER} PSI) — Amber CAS",
        annotation_position="bottom right",
    )
    fig_psi.add_hline(
        y=PSI_CYAN, line_dash="dot", line_color="#f59e0b",
        annotation_text=f"OBSERVER OXY LO PRESS ({PSI_CYAN} PSI) — Cyan CAS",
        annotation_position="top right",
    )
    fig_psi.update_layout(
        height=440,
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
        yaxis=dict(range=[0, PSI_CHARGE + 100]),
    )
    st.plotly_chart(fig_psi, use_container_width=True)

elif "delta_press" in df.columns and AC_COL:
    st.info(
        "Absolute PSI column (`psi`) not present in data — showing daily pressure drop. "
        "Re-run `save_oxy_report` to populate the PSI column with real AMM thresholds."
    )

st.divider()

# ── Chart 2: Aircraft status bar ──────────────────────────────────────────────
st.subheader("2. Aircraft Status — Latest Reading vs. AMM Limits")
st.caption(
    "Most recent pressure reading per aircraft. "
    "Red bars require immediate maintenance before next departure. "
    "Yellow bars require monitoring and may limit dispatch at some stations."
)

if "psi" in df.columns and AC_COL:
    latest_status = (
        df.dropna(subset=[AC_COL, "psi"])
        .sort_values("date")
        .groupby(AC_COL)
        .last()[["psi"]]
        .reset_index()
        .sort_values("psi")
    )
    latest_status["alert_level"] = latest_status["psi"].apply(_alert_level)
    latest_status["color"] = latest_status["psi"].apply(_alert_color)

    fig_status = go.Figure(go.Bar(
        y=latest_status[AC_COL].astype(str),
        x=latest_status["psi"],
        orientation="h",
        marker_color=latest_status["color"],
        text=latest_status["alert_level"],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f} PSI<extra></extra>",
    ))
    fig_status.add_vline(
        x=PSI_AMBER, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Amber ({PSI_AMBER} PSI)",
    )
    fig_status.add_vline(
        x=PSI_CYAN, line_dash="dot", line_color="#f59e0b",
        annotation_text=f"Cyan ({PSI_CYAN} PSI)",
    )
    fig_status.update_layout(
        title="Latest oxygen pressure per aircraft",
        xaxis_title="Pressure (PSI)",
        yaxis_title="MSN",
        height=max(320, len(latest_status) * 34),
        xaxis=dict(range=[0, PSI_CHARGE + 100]),
        margin=dict(l=10, r=160, t=40, b=10),
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.divider()

# ── Chart 3: Daily pressure drop trend ────────────────────────────────────────
st.subheader("3. Daily Pressure Drop — Leakage Rate Trend")
st.caption(
    "Rising pressure drop indicates accelerating leakage. "
    "Combined with the absolute PSI level, this helps predict **when** an aircraft "
    "will cross the dispatch threshold."
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
        title="Daily Oxygen Pressure Drop — Fleet Trend",
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

# ── Dispatch forecast ─────────────────────────────────────────────────────────
if "psi" in df.columns and "delta_press" in df.columns and AC_COL:
    st.subheader("4. Dispatch Forecast — Days Until Threshold")
    st.caption(
        "Estimated days until each aircraft crosses the 1,155 PSI (cyan) or 845 PSI (amber) "
        "threshold, based on the average daily pressure drop over the selected period."
    )

    forecast_rows = []
    for msn, grp in df.dropna(subset=["psi", "delta_press", AC_COL]).groupby(AC_COL):
        current_psi = grp.sort_values("date")["psi"].iloc[-1]
        avg_drop = grp["delta_press"].mean()
        if avg_drop > 0:
            days_to_cyan  = max(0, (current_psi - PSI_CYAN)  / avg_drop)
            days_to_amber = max(0, (current_psi - PSI_AMBER) / avg_drop)
        else:
            days_to_cyan  = float("inf")
            days_to_amber = float("inf")
        forecast_rows.append({
            "MSN": msn,
            "Current PSI": round(current_psi),
            "Avg Drop (PSI/day)": round(avg_drop, 2),
            "Days → Cyan (1155)": "—" if days_to_cyan == float("inf") else int(days_to_cyan),
            "Days → Amber (845)": "—" if days_to_amber == float("inf") else int(days_to_amber),
            "Status": _alert_level(current_psi),
        })

    if forecast_rows:
        df_fc = pd.DataFrame(forecast_rows).sort_values("Current PSI")

        def _color_rows(row):
            if row["Current PSI"] < PSI_AMBER:
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            elif row["Current PSI"] < PSI_CYAN:
                return ["background-color: rgba(245,158,11,0.12)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_fc.style.apply(_color_rows, axis=1),
            use_container_width=True,
            hide_index=True,
        )

# ── Top pressure-drop events ──────────────────────────────────────────────────
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
    with st.expander("Top 15 highest single-day pressure drop events"):
        st.dataframe(top, use_container_width=True, hide_index=True)
