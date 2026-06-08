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

# ── Data ────────────────────────────────────────────────────────
df = load("e2_fuel_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_fuel_consumption_report` job in Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

AC_COL = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in df.columns), None)
if AC_COL:
    df[AC_COL] = df[AC_COL].astype(str)

# ── Sidebar controls ────────────────────────────────────────────
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

# ── Discover fuel columns ────────────────────────────────────────
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

# Total cruise fuel per flight, computed once and reused across KPIs and sections.
if cruise_cols:
    df["_cruise_total"] = df[cruise_cols].sum(axis=1)

# ── KPIs ───────────────────────────────────────────────────
avg_cruise = df["_cruise_total"].mean() if cruise_cols else 0
total_burn_cols = list(burn_cols.keys())
avg_total = df[total_burn_cols].sum(axis=1).mean() if total_burn_cols else 0
n_flights = len(df)

n_rising = None
if cruise_cols and AC_COL and "date" in df.columns:
    rising = 0
    counted = 0
    for _, g in df.dropna(subset=["date", "_cruise_total"]).groupby(AC_COL):
        g = g.sort_values("date")
        if len(g) < 3:
            continue
        x = (g["date"] - g["date"].min()).dt.days.to_numpy(dtype=float)
        y = g["_cruise_total"].to_numpy(dtype=float)
        if x.max() == x.min():
            continue
        baseline = y.mean()
        if baseline <= 0:
            continue
        slope = np.polyfit(x, y, 1)[0]
        projected_rise = slope * (x.max() - x.min())
        counted += 1
        if projected_rise / baseline > 0.05:
            rising += 1
    n_rising = rising if counted else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flights analysed", f"{n_flights:,}")
c2.metric("Avg cruise fuel (kg)", f"{avg_cruise:.0f}" if cruise_cols else "—")
c3.metric("Avg total fuel per flight (kg)", f"{avg_total:.0f}" if total_burn_cols else "—")
c4.metric(
    "Aircraft with rising cruise-burn trend (>5%)",
    f"{n_rising}" if n_rising is not None else "—",
    help=(
        "Count of aircraft whose own cruise fuel burn shows a rising linear trend "
        "(np.polyfit slope) projecting more than a 5% increase across the analysed "
        "window, measured against each aircraft's own baseline mean. Unlike a "
        "fleet-relative threshold, this only fires when a tail is actually trending "
        "worse than its own history — an actionable pre-failure signal."
    ),
)

st.divider()

# ── Section 1: Fuel distribution by phase ─────────────────────────────
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

# ── Section 2: Cruise efficiency per aircraft ──────────────────────────
st.subheader("2. Cruise Efficiency per Aircraft")
st.caption(
    "Lower cruise consumption = more efficient engine. "
    "Red bars are more than 5% above the fleet average."
)

if cruise_cols and AC_COL:
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

# ── Section 3: Cruise fuel trend over time per aircraft ────────────────────
st.subheader("3. Cruise Fuel Trend Over Time")
st.caption(
    "A rising line for a specific aircraft indicates increasing fuel consumption — "
    "a potential sign of engine degradation. Color = MSN."
)

if cruise_cols and AC_COL and "date" in df.columns:
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

# ── Section 4: Monthly fleet trend ───────────────────────────────────
st.subheader("4. Fleet Monthly Fuel Trend")
st.caption("Rising fleet-wide trend may indicate deterioration across multiple aircraft.")

if cruise_cols and "date" in df.columns:
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

st.divider()

# ── Section 5: Per-aircraft cruise-burn degradation vs own baseline ────────
st.subheader("5. Per-Aircraft Cruise-Burn Degradation vs Own Baseline")

# Normalize the cruise-burn signal by cruise duration so route length no longer
# confounds engine health: a longer cruise burns more fuel regardless of engine
# condition. We compare a cruise-specific burn RATE (kg/h) against each tail's own
# baseline. If the cruise-duration column is absent, fall back to absolute kg.
_USE_RATE = ("time_sec_cruise" in df.columns) and bool(cruise_cols)

if _USE_RATE:
    _dur_hr = pd.to_numeric(df["time_sec_cruise"], errors="coerce") / 3600.0
    # Guarded: only where cruise duration is present and strictly positive.
    df["_cruise_kg_per_hr"] = np.where(_dur_hr > 0, df["_cruise_total"] / _dur_hr, np.nan)
    _METRIC = "_cruise_kg_per_hr"
    _UNIT = "kg/h"
    _Y_TITLE = "Cruise fuel rate (kg/h)"
    st.caption(
        "Pick a tail to compare its per-flight cruise burn RATE against its OWN "
        "historical baseline (median of its earliest flights). The signal is "
        "normalized by cruise time (kg/h), so route length no longer confounds "
        "engine health. Amber band = 5–10% above baseline; red zone = more than "
        "10% above. This anchors the signal to each aircraft's own history instead "
        "of a fleet-relative threshold."
    )
else:
    _METRIC = "_cruise_total"
    _UNIT = "kg"
    _Y_TITLE = "Cruise fuel (kg)"
    st.caption(
        "Pick a tail to compare its per-flight cruise burn against its OWN historical "
        "baseline (median of its earliest flights). Amber band = 5–10% above baseline; "
        "red zone = more than 10% above. This anchors the signal to each aircraft's own "
        "history instead of a fleet-relative threshold. Note: cruise-duration data "
        "(time_sec_cruise) is unavailable, so this uses absolute cruise kg — longer "
        "routes may read high regardless of engine health."
    )

_MIN_BASELINE_FLIGHTS = 5
_AMBER_FACTOR = 1.05
_RED_FACTOR = 1.10

if not (cruise_cols and AC_COL and "date" in df.columns):
    st.info("Per-aircraft baseline analysis requires cruise fuel, aircraft and date columns.")
else:
    df_base = df.dropna(subset=["date", _METRIC, AC_COL]).copy()
    tails = sorted(df_base[AC_COL].unique().tolist())

    if not tails:
        st.info("No flights with valid cruise fuel data in the selected period.")
    else:
        chosen = st.selectbox("Select aircraft (MSN)", options=tails, key="deg_msn")
        g = df_base[df_base[AC_COL] == chosen].sort_values("date").reset_index(drop=True)

        if len(g) < _MIN_BASELINE_FLIGHTS:
            st.info(
                f"MSN {chosen} has only {len(g)} flight(s) in the selected window. "
                f"At least {_MIN_BASELINE_FLIGHTS} flights are required to build a "
                "reliable baseline. Widen the history range in the sidebar."
            )
        else:
            n_base = max(_MIN_BASELINE_FLIGHTS, int(np.ceil(len(g) * 0.30)))
            n_base = min(n_base, len(g))
            baseline = float(g[_METRIC].iloc[:n_base].median())

            if baseline <= 0:
                st.info(
                    f"MSN {chosen} has a non-positive baseline cruise burn — "
                    "cannot compute a degradation reference."
                )
            else:
                amber_level = baseline * _AMBER_FACTOR
                red_level = baseline * _RED_FACTOR

                def _flight_color(v: float) -> str:
                    if v > red_level:
                        return "#ef4444"
                    if v > amber_level:
                        return "#fbbf24"
                    return "#22c55e"

                g["color"] = g[_METRIC].apply(_flight_color)

                x_min = g["date"].min()
                x_max = g["date"].max()
                y_top = max(float(g[_METRIC].max()), red_level) * 1.05

                fig_deg = go.Figure()
                fig_deg.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=amber_level, y1=red_level,
                    fillcolor="rgba(251,191,36,0.18)", line_width=0, layer="below",
                )
                fig_deg.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=red_level, y1=y_top,
                    fillcolor="rgba(239,68,68,0.15)", line_width=0, layer="below",
                )
                fig_deg.add_trace(go.Scatter(
                    x=g["date"], y=g[_METRIC],
                    mode="lines+markers",
                    line=dict(color="#94a3b8", width=1),
                    marker=dict(color=g["color"], size=9),
                    name="Cruise fuel rate" if _USE_RATE else "Cruise fuel",
                    hovertemplate="%{x|%d-%b-%y}: %{y:.0f} " + _UNIT + "<extra></extra>",
                ))
                fig_deg.add_hline(
                    y=baseline, line_dash="dash", line_color="#3b82f6",
                    annotation_text=f"Own baseline ({baseline:.0f} {_UNIT})",
                    annotation_position="top left",
                )
                fig_deg.add_hline(
                    y=amber_level, line_dash="dot", line_color="#d97706",
                    annotation_text="+5%", annotation_position="bottom left",
                )
                fig_deg.add_hline(
                    y=red_level, line_dash="dot", line_color="#dc2626",
                    annotation_text="+10%", annotation_position="top left",
                )
                _metric_label = "cruise fuel rate" if _USE_RATE else "cruise fuel"
                fig_deg.update_layout(
                    title=f"MSN {chosen} — per-flight {_metric_label} vs own baseline",
                    xaxis_title="", yaxis_title=_Y_TITLE,
                    height=420,
                    xaxis=dict(tickformat="%d-%b-%y"),
                    showlegend=False,
                )
                st.plotly_chart(fig_deg, use_container_width=True)

                recent_n = min(_MIN_BASELINE_FLIGHTS, len(g))
                recent_mean = float(g[_METRIC].iloc[-recent_n:].mean())
                pct_above = (recent_mean - baseline) / baseline * 100.0
                _baseline_word = "cruise burn-rate baseline" if _USE_RATE else "baseline"

                if recent_mean > red_level:
                    st.warning(
                        f"MSN {chosen} is burning {pct_above:.1f}% above its own "
                        f"{_baseline_word} over its last {recent_n} flights — "
                        "inspect for engine degradation."
                    )
                elif recent_mean > amber_level:
                    st.warning(
                        f"MSN {chosen} is trending {pct_above:.1f}% above its own "
                        f"{_baseline_word} over its last {recent_n} flights — "
                        "monitor closely for engine degradation."
                    )
                else:
                    st.success(
                        f"MSN {chosen} is within tolerance — last {recent_n} flights are "
                        f"{pct_above:+.1f}% vs its own baseline of {baseline:.0f} {_UNIT}."
                    )
