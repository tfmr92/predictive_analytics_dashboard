"""
Wheels & Brakes — landing gear wheel health and removal forecasting.
Focus: accelerated degradation detection + remaining-cycle estimates.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Wheels & Brakes", layout="wide")

st.title("🛞 Wheels & Brakes")
st.markdown(
    "Identifies which wheels are in **accelerated degradation**, which should be scheduled for **removal**, "
    "and how many cycles remain before they reach the removal threshold."
)

# ── Data ──────────────────────────────────────────────────────────────────────
df = load("e2_wnb_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_wheel_brake_report` job in Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
if "ac_sn" in df.columns:
    df["ac_sn"] = df["ac_sn"].astype(str)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 30, 365, 120)
    all_ac = sorted(df["ac_sn"].dropna().unique().tolist()) if "ac_sn" in df.columns else []
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=all_ac)
    WHEEL_LIFE = st.number_input(
        "Assumed wheel life (cycles)", min_value=500, max_value=5000, value=1200, step=100,
        help="Typical E2 main-gear wheel removal threshold. Adjust per your Maintenance Manual.",
    )

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
if selected_ac and "ac_sn" in df.columns:
    df = df[df["ac_sn"].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Position mapping ──────────────────────────────────────────────────────────
_POSITIONS = {
    "mlg1":   ("MLG 1 — LH Front",  "prediction_mlg1",   "time_since_installation_1"),
    "mlg2":   ("MLG 2 — LH Rear",   "prediction_mlg2",   "time_since_installation_2"),
    "mlg3":   ("MLG 3 — RH Front",  "prediction_mlg3",   "time_since_installation_3"),
    "mlg4":   ("MLG 4 — RH Rear",   "prediction_mlg4",   "time_since_installation_4"),
    "nlg_lh": ("NLG — Left",        "prediction_nlg_lh", "time_since_installation_5"),
    "nlg_rh": ("NLG — Right",       "prediction_nlg_rh", "time_since_installation_6"),
}

_HARD_G = 1.4
_SEVERE_G = 2.0

# ── KPIs ──────────────────────────────────────────────────────────────────────
pred_cols = [v[1] for v in _POSITIONS.values() if v[1] in df.columns]
total_alerts = int(df[pred_cols].eq(1).any(axis=1).sum()) if pred_cols else 0
hard_lh = int((df["NormAccel_lh"] > _HARD_G).sum()) if "NormAccel_lh" in df.columns else 0
hard_rh = int((df["NormAccel_rh"] > _HARD_G).sum()) if "NormAccel_rh" in df.columns else 0

# Aircraft with any wheel in alert
ac_in_alert = set()
if "ac_sn" in df.columns and pred_cols:
    mask = df[pred_cols].eq(1).any(axis=1)
    ac_in_alert = set(df.loc[mask, "ac_sn"].dropna().unique())

c1, c2, c3, c4 = st.columns(4)
c1.metric("✈️ Aircraft in alert", len(ac_in_alert))
c2.metric("🔴 Flights with removal alert", total_alerts)
c3.metric("⚠️ Hard landings — LH (> 1.4 g)", hard_lh)
c4.metric("⚠️ Hard landings — RH (> 1.4 g)", hard_rh)

st.divider()

# ── Section 1: Removal Priority Table ─────────────────────────────────────────
st.subheader("1. Removal Priority — Wheels to Act On")
st.caption(
    "Shows the latest alert rate and cycles-in-service for each wheel position. "
    f"Remaining cycles = assumed life ({WHEEL_LIFE} cycles) minus time-since-installation."
)

rows = []
if "ac_sn" in df.columns:
    for pos_key, (pos_label, pred_col, tsi_col) in _POSITIONS.items():
        if pred_col not in df.columns:
            continue

        # Latest prediction per aircraft
        latest = (
            df.dropna(subset=["ac_sn"])
            .sort_values("date")
            .groupby("ac_sn")
            .last()
        )
        for ac in latest.index:
            row = latest.loc[ac]
            alert = int(row[pred_col]) if pred_col in latest.columns else 0
            tsi = float(row[tsi_col]) if tsi_col in latest.columns and pd.notna(row.get(tsi_col)) else None
            remaining = max(0, WHEEL_LIFE - tsi) if tsi is not None else None
            alert_rate = float(
                df[df["ac_sn"] == ac][pred_col].eq(1).mean() * 100
            ) if pred_col in df.columns else 0

            rows.append({
                "MSN": ac,
                "Position": pos_label,
                "Current Alert": "🔴 YES" if alert else "✅ No",
                "Alert Rate (%)": round(alert_rate, 1),
                "Cycles In Service": round(tsi) if tsi is not None else "—",
                "Est. Remaining Cycles": round(remaining) if remaining is not None else "—",
            })

if rows:
    priority_df = pd.DataFrame(rows)
    # Sort: alert first, then highest alert rate
    priority_df["_sort"] = priority_df["Current Alert"].apply(lambda x: 0 if "YES" in x else 1)
    priority_df = priority_df.sort_values(["_sort", "Alert Rate (%)"], ascending=[True, False]).drop(columns="_sort")
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
else:
    st.info("Prediction columns not found. Run the Dagster pipeline to generate predictions.")

st.divider()

# ── Section 2: Degradation Heatmap ────────────────────────────────────────────
st.subheader("2. Degradation Heatmap — Alert Rate per Aircraft × Wheel Position")
st.caption("Red = high proportion of flights with removal alert at that position.")

available_preds = [v[1] for v in _POSITIONS.values() if v[1] in df.columns]
pos_labels = {v[1]: v[0] for v in _POSITIONS.values()}

if available_preds and "ac_sn" in df.columns:
    heatmap_data = (
        df.groupby("ac_sn")[available_preds]
        .apply(lambda g: (g == 1).mean() * 100)
        .rename(columns=pos_labels)
        .reset_index()
    )
    melted = heatmap_data.melt(id_vars="ac_sn", var_name="Position", value_name="Alert Rate (%)")

    fig_heat = px.density_heatmap(
        melted, x="Position", y="ac_sn", z="Alert Rate (%)",
        color_continuous_scale=["#dcfce7", "#fef9c3", "#fca5a5", "#ef4444"],
        labels={"ac_sn": "MSN"},
        title="Alert Rate (%) — by Aircraft and Wheel Position",
    )
    fig_heat.update_layout(height=max(300, len(heatmap_data) * 30))
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Section 3: Time Since Installation Trend ──────────────────────────────────
st.subheader("3. Cycles In Service — Progress Toward Removal Threshold")
st.caption(
    f"Each line is one aircraft. The red dashed line marks the assumed {WHEEL_LIFE}-cycle removal threshold. "
    "Aircraft near or above the line should be scheduled for wheel change."
)

tsi_cols_available = [(v[0], v[2]) for v in _POSITIONS.values() if v[2] in df.columns]
if tsi_cols_available and "ac_sn" in df.columns:
    tab_names = [label for label, _ in tsi_cols_available]
    tsi_tabs = st.tabs(tab_names)
    for tab_w, (pos_label, tsi_col) in zip(tsi_tabs, tsi_cols_available):
        with tab_w:
            df_tsi = df.dropna(subset=["date", tsi_col, "ac_sn"]).copy()
            if df_tsi.empty:
                st.info("No data for this position.")
                continue

            fig_tsi = px.line(
                df_tsi, x="date", y=tsi_col,
                color="ac_sn",
                labels={tsi_col: "Cycles in Service", "date": "", "ac_sn": "MSN"},
                title=f"{pos_label} — Cycles in Service over Time",
            )
            fig_tsi.add_hline(
                y=WHEEL_LIFE, line_dash="dash", line_color="red",
                annotation_text=f"Removal threshold ({WHEEL_LIFE} cycles)",
                annotation_position="top right",
            )
            fig_tsi.update_layout(
                height=320,
                xaxis=dict(tickformat="%d-%b-%y"),
                legend_title_text="MSN",
            )
            st.plotly_chart(fig_tsi, use_container_width=True)

st.divider()

# ── Section 4: Landing Hardness ────────────────────────────────────────────────
st.subheader("4. Landing Hardness — Accelerated Wear Events")
st.caption(
    f"Hard landings (> {_HARD_G} g) accelerate wear on tyres, brakes, and structure. "
    f"Severe landings (> {_SEVERE_G} g) require mandatory inspection."
)

tab_lh_land, tab_rh_land = st.tabs(["Left Main Gear (LH)", "Right Main Gear (RH)"])

for tab_l, acol, label in [
    (tab_lh_land, "NormAccel_lh", "LH"),
    (tab_rh_land, "NormAccel_rh", "RH"),
]:
    with tab_l:
        if acol not in df.columns:
            st.info("Column not available.")
            continue

        df_land = df.dropna(subset=["date", acol]).copy()
        df_land["Severity"] = pd.cut(
            df_land[acol],
            bins=[-999, _HARD_G, _SEVERE_G, 999],
            labels=["Normal", "Hard", "Severe"],
        )
        color_map = {"Normal": "#22c55e", "Hard": "#f59e0b", "Severe": "#ef4444"}

        fig_land = px.scatter(
            df_land, x="date", y=acol,
            color="Severity",
            color_discrete_map=color_map,
            symbol="ac_sn",
            hover_data={"ac_sn": True},
            labels={acol: "Peak G-force (g)", "date": "", "ac_sn": "MSN"},
            title=f"Landing G-force — {label}",
        )
        fig_land.add_hline(y=_HARD_G, line_dash="dot", line_color="orange",
                           annotation_text="Hard (1.4 g)", annotation_position="top right")
        fig_land.add_hline(y=_SEVERE_G, line_dash="dot", line_color="red",
                           annotation_text="Severe (2.0 g)", annotation_position="top right")
        fig_land.update_layout(
            height=360,
            xaxis=dict(tickformat="%d-%b-%y"),
            legend_title_text="Severity",
        )
        st.plotly_chart(fig_land, use_container_width=True)

        # Per-MSN hard landing count bar
        if "ac_sn" in df_land.columns:
            hard_counts = (
                df_land[df_land[acol] > _HARD_G]
                .groupby("ac_sn")
                .size()
                .reset_index(name="Hard Landings")
                .sort_values("Hard Landings", ascending=False)
            )
            if not hard_counts.empty:
                fig_bar = px.bar(
                    hard_counts, x="ac_sn", y="Hard Landings",
                    color="Hard Landings",
                    color_continuous_scale=["#fef9c3", "#ef4444"],
                    labels={"ac_sn": "MSN"},
                    title=f"Hard landings per aircraft — {label}",
                )
                fig_bar.update_layout(height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_bar, use_container_width=True)
