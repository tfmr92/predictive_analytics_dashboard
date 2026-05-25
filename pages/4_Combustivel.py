"""
Fuel Consumption — per-phase fuel burn monitoring.
Detects anomalous consumption that may indicate engine degradation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Fuel Consumption", layout="wide")

st.title("⛽ Fuel Consumption")
st.markdown(
    "Tracks fuel burned in each phase of flight. "
    "An **increasing trend during cruise** can indicate engine degradation, "
    "aerodynamic issues, or an inefficient flight plan."
)

# ── Data ──────────────────────────────────────────────────────────────────────
df = load("e2_fuel_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_fuel_consumption_report` job in Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

AC_COL = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in df.columns), None)
if AC_COL:
    df[AC_COL] = df[AC_COL].astype(str)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 30, 365, 120)
    all_ac = sorted(df[AC_COL].dropna().unique().tolist()) if AC_COL else []
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=all_ac)

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
if selected_ac and AC_COL:
    df = df[df[AC_COL].astype(str).isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Discover fuel columns ──────────────────────────────────────────────────────
_PHASE_MAP = {
    "taxi_out":       "Taxi Out",
    "take_off":       "Take-off",
    "second_segment": "2nd Segment",
    "initial_climb":  "Initial Climb",
    "climb":          "Climb",
    "cruise":         "Cruise",
    "descent":        "Descent",
    "approach":       "Approach",
    "final_approach": "Final Approach",
    "landing":        "Landing",
    "taxi_in":        "Taxi In",
}

burn_cols: dict[str, tuple[str, str]] = {}
for phase_en, phase_label in _PHASE_MAP.items():
    for eng in (1, 2):
        col = f"{phase_en}fuelMeterFuelBurn{eng}Kg"
        if col in df.columns:
            burn_cols[col] = (phase_label, f"Engine {eng}")

cruise_cols = [c for c in burn_cols if "cruise" in c]

# ── KPIs ──────────────────────────────────────────────────────────────────────
avg_cruise = df[cruise_cols].sum(axis=1).mean() if cruise_cols else 0
total_burn_cols = list(burn_cols.keys())
avg_total = df[total_burn_cols].sum(axis=1).mean() if total_burn_cols else 0
n_flights = len(df)

c1, c2, c3 = st.columns(3)
c1.metric("Flights analysed", f"{n_flights:,}")
c2.metric("Avg cruise fuel (kg)", f"{avg_cruise:.0f}" if cruise_cols else "—")
c3.metric("Avg total fuel per flight (kg)", f"{avg_total:.0f}" if total_burn_cols else "—")

st.divider()

# ── Section 1: Fuel distribution by phase ─────────────────────────────────────
st.subheader("1. Where Is Fuel Burned?")
st.caption("Average consumption per flight phase across the selected period.")

if burn_cols:
    phase_totals = {}
    for col, (phase_label, motor) in burn_cols.items():
        label = f"{phase_label} ({motor})"
        phase_totals[label] = df[col].mean()

    phase_df = (
        pd.DataFrame(list(phase_totals.items()), columns=["Phase", "Avg (kg)"])
        .sort_values("Avg (kg)", ascending=False)
    )

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(
            phase_df, names="Phase", values="Avg (kg)",
            title="Proportion by Phase (avg flight)",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            phase_df, y="Phase", x="Avg (kg)",
            orientation="h",
            title="Average fuel per phase (kg)",
            color="Avg (kg)",
            color_continuous_scale=["#bbf7d0", "#fbbf24", "#ef4444"],
        )
        fig_bar.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Section 2: Cruise efficiency per aircraft ─────────────────────────────────
st.subheader("2. Cruise Efficiency per Aircraft")
st.caption(
    "Lower cruise consumption = more efficient engine. "
    "Red bars are more than 5% above the fleet average."
)

if cruise_cols and AC_COL:
    df["_cruise_total"] = df[cruise_cols].sum(axis=1)
    eff = (
        df.groupby(AC_COL)["_cruise_total"]
        .mean()
        .reset_index()
        .rename(columns={"_cruise_total": "Avg Cruise Fuel (kg)"})
        .sort_values("Avg Cruise Fuel (kg)", ascending=False)
    )
    fleet_mean = eff["Avg Cruise Fuel (kg)"].mean()
    eff["color"] = eff["Avg Cruise Fuel (kg)"].apply(
        lambda x: "#ef4444" if x > fleet_mean * 1.05 else "#22c55e"
    )

    fig_eff = go.Figure(go.Bar(
        x=eff["Avg Cruise Fuel (kg)"],
        y=eff[AC_COL].astype(str),
        orientation="h",
        marker_color=eff["color"],
        hovertemplate="%{y}: %{x:.0f} kg<extra></extra>",
    ))
    fig_eff.add_vline(
        x=fleet_mean, line_dash="dash", line_color="gray",
        annotation_text="Fleet avg", annotation_position="top right",
    )
    fig_eff.update_layout(
        title="Avg cruise fuel by aircraft",
        xaxis_title="kg", yaxis_title="MSN",
        height=max(300, len(eff) * 30),
    )
    st.plotly_chart(fig_eff, use_container_width=True)

st.divider()

# ── Section 3: Cruise fuel trend over time per aircraft ───────────────────────
st.subheader("3. Cruise Fuel Trend Over Time")
st.caption(
    "A rising line for a specific aircraft indicates increasing fuel consumption — "
    "a potential sign of engine degradation. Color = MSN."
)

if cruise_cols and AC_COL and "date" in df.columns:
    df["_cruise_total"] = df[cruise_cols].sum(axis=1)
    df_trend = df.dropna(subset=["date", "_cruise_total", AC_COL]).copy()

    # Weekly average per aircraft to smooth noise
    df_trend["week"] = df_trend["date"].dt.to_period("W").dt.start_time
    weekly = (
        df_trend.groupby([AC_COL, "week"])["_cruise_total"]
        .mean()
        .reset_index()
        .rename(columns={"week": "date", "_cruise_total": "Avg Cruise Fuel (kg)"})
    )

    fig_trend = px.line(
        weekly, x="date", y="Avg Cruise Fuel (kg)",
        color=AC_COL,
        labels={"date": "", AC_COL: "MSN"},
        title="Weekly avg cruise fuel — per aircraft",
        markers=True,
    )
    fig_trend.update_layout(
        height=380,
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Section 4: Monthly fleet trend ────────────────────────────────────────────
st.subheader("4. Fleet Monthly Fuel Trend")
st.caption("Rising fleet-wide trend may indicate deterioration across multiple aircraft.")

if cruise_cols and "date" in df.columns:
    df["_cruise_total"] = df[cruise_cols].sum(axis=1)
    monthly = (
        df.dropna(subset=["date"])
        .set_index("date")
        .resample("ME")["_cruise_total"]
        .mean()
        .reset_index()
        .rename(columns={"date": "Month", "_cruise_total": "Avg Cruise Fuel (kg)"})
    )

    if len(monthly) > 1:
        fig_monthly = px.area(
            monthly, x="Month", y="Avg Cruise Fuel (kg)",
            title="Fleet monthly avg cruise fuel",
            color_discrete_sequence=["#3b82f6"],
        )
        # Overlay trend line
        if len(monthly) > 2:
            x_num = (monthly["Month"] - monthly["Month"].min()).dt.days
            z = np.polyfit(x_num, monthly["Avg Cruise Fuel (kg)"].fillna(0), 1)
            trend_y = np.polyval(z, x_num)
            fig_monthly.add_scatter(
                x=monthly["Month"], y=trend_y,
                mode="lines", name="Trend",
                line=dict(dash="dot", color="orange", width=2),
            )
        fig_monthly.update_layout(
            height=320,
            xaxis=dict(tickformat="%b-%y"),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
