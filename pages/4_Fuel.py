"""
Fuel Consumption — per-phase fuel burn monitoring.
Detects anomalous consumption that may indicate engine degradation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utils.drive_loader import load, clean_df, make_prefix_map, display_name, render_freshest_badge

st.set_page_config(page_title="Fuel Consumption", layout="wide")

st.title(":material/local_gas_station: Fuel Consumption")
st.markdown(
    "Tracks fuel burned in each phase of flight. "
    "An **increasing trend during cruise** can indicate engine degradation, "
    "aerodynamic issues, or an inefficient flight plan."
)

render_freshest_badge(["e2_fuel_report.parquet"], label="Fuel report")

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

# Filter future dates and invalid serials
_fuel_prefix_map = make_prefix_map()
df = clean_df(df, date_col="date", ac_col=AC_COL, prefix_map=_fuel_prefix_map)

def _dnm(msn) -> str:
    return display_name(str(msn), _fuel_prefix_map)

# ── Sidebar controls ────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    days_back = st.slider("Days of history", 30, 365, 120)
    all_ac = sorted(df[AC_COL].dropna().unique().tolist()) if AC_COL else []
    selected_ac = st.multiselect("Aircraft", options=all_ac, default=all_ac, format_func=_dnm)

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

# Map each engine number to its cruise-burn column, so Section 6 can localize a
# tail's rising cruise burn to a specific engine (ENG1 / ENG2).
_CRUISE_ENG_COLS: dict[int, str] = {}
for _c in cruise_cols:
    _motor = burn_cols[_c][1]  # e.g. "Engine 1"
    try:
        _CRUISE_ENG_COLS[int(_motor.split()[-1])] = _c
    except (ValueError, IndexError):
        pass

# Ensure all fuel columns are numeric (parquet may deserialise them as strings)
all_burn_cols = list(burn_cols.keys())
for _c in all_burn_cols:
    df[_c] = pd.to_numeric(df[_c], errors="coerce")

# Total cruise fuel per flight, computed once and reused across KPIs and sections.
if cruise_cols:
    df["_cruise_total"] = df[cruise_cols].sum(axis=1)

# ── Shared cruise-burn baseline method (KPI c4 + Section 5 + Section 6) ─────────
# One own-baseline calculation drives the headline KPI, the single-aircraft
# degradation view (Section 5) and the watchlist (Section 6), so the headline
# count and the MSNs named in the watchlist can never disagree.
# Normalize the cruise-burn signal by cruise duration when available so route
# length no longer confounds engine health (a longer cruise burns more fuel
# regardless of engine condition); otherwise fall back to absolute kg.
_USE_RATE = ("time_sec_cruise" in df.columns) and bool(cruise_cols)

if _USE_RATE:
    _dur_hr = pd.to_numeric(df["time_sec_cruise"], errors="coerce") / 3600.0
    # Guarded: only where cruise duration is present and strictly positive.
    df["_cruise_kg_per_hr"] = np.where(_dur_hr > 0, df["_cruise_total"] / _dur_hr, np.nan)
    _METRIC = "_cruise_kg_per_hr"
    _UNIT = "kg/h"
    _Y_TITLE = "Cruise fuel rate (kg/h)"
else:
    _METRIC = "_cruise_total"
    _UNIT = "kg"
    _Y_TITLE = "Cruise fuel (kg)"

_MIN_BASELINE_FLIGHTS = 5
_AMBER_FACTOR = 1.05
_RED_FACTOR = 1.10

# Per-engine cruise-burn metric, normalized identically to _cruise_kg_per_hr
# (divided by the same cruise duration when _USE_RATE, so units stay kg/h). Lets
# Section 6 re-run the own-baseline method on each engine and name the rising one.
_ENG_METRIC: dict[int, str] = {}
for _eng_n, _eng_col in _CRUISE_ENG_COLS.items():
    _mcol = f"_cruise_eng{_eng_n}_metric"
    _eng_series = pd.to_numeric(df[_eng_col], errors="coerce")
    if _USE_RATE:
        df[_mcol] = np.where(_dur_hr > 0, _eng_series / _dur_hr, np.nan)
    else:
        df[_mcol] = _eng_series
    _ENG_METRIC[_eng_n] = _mcol


@st.cache_data(ttl=300)
def _build_cruise_watchlist(df_in, metric, ac_col, min_flights, amber, red):
    df_w = df_in.dropna(subset=["date", metric, ac_col]).copy()
    rows = []
    n_excluded = 0
    for tail, g in df_w.groupby(ac_col):
        g = g.sort_values("date")
        n = len(g)
        n_base = min(n, max(min_flights, int(np.ceil(n * 0.30))))
        recent_n = min(min_flights, n)
        # Temporal separation: the baseline (earliest n_base flights) and the recent
        # window (last recent_n flights) must NOT overlap, otherwise a degrading tail
        # leaks its recent flights into its own baseline and hides the rise.
        if n < n_base + recent_n:
            n_excluded += 1
            continue
        baseline = float(g[metric].iloc[:n_base].median())
        if baseline <= 0:
            n_excluded += 1
            continue
        recent_mean = float(g[metric].iloc[-recent_n:].mean())
        pct_above = (recent_mean - baseline) / baseline * 100.0
        if recent_mean > baseline * red:
            status = ">+10%"
        elif recent_mean > baseline * amber:
            status = ">+5%"
        else:
            status = "OK"
        rows.append({
            "Aircraft": tail, "Status": status, "Recent": recent_mean,
            "Baseline": baseline, "pct_above": pct_above, "Flights": n,
        })
    return pd.DataFrame(rows), n_excluded


# Shift of the ENG1−ENG2 asymmetry must exceed this many robust flight-to-flight
# scatters of that same asymmetry before it is attributed to one engine.
_ASYM_NOISE_K = 1.0


@st.cache_data(ttl=300)
def _engine_asymmetry_map(df_in, eng_metric, ac_col, min_flights):
    """Attribute a rising cruise burn to a SINGLE engine via the ENG1-vs-ENG2 burn-rate
    ASYMMETRY (asym = ENG1_rate - ENG2_rate per flight). Because both engines fly the
    SAME weight, altitude and wind on each flight, their per-flight difference cancels
    those shared confounders — so a SHIFT of that difference away from the tail's own
    early baseline, beyond the asymmetry's own flight-to-flight noise, isolates the
    diverging engine. This is the physically honest per-engine signal: the previous
    per-engine own-baseline fired on EVERY engine whenever the aircraft simply flew
    heavier/higher (both burning more together), spuriously reading 'Both'.

    Returns {tail: {label, shift, thresh, base, recent, n}} where label is ENG1 / ENG2 /
    '—' and shift is the signed (ENG1−ENG2) move: positive → ENG1 diverging."""
    if 1 not in eng_metric or 2 not in eng_metric:
        return {}
    m1, m2 = eng_metric[1], eng_metric[2]
    d = df_in.dropna(subset=["date", m1, m2, ac_col]).copy()
    d["_asym"] = pd.to_numeric(d[m1], errors="coerce") - pd.to_numeric(d[m2], errors="coerce")
    d = d.dropna(subset=["_asym"])
    out: dict = {}
    for tail, g in d.groupby(ac_col):
        g = g.sort_values("date")
        n = len(g)
        n_base = min(n, max(min_flights, int(np.ceil(n * 0.30))))
        recent_n = min(min_flights, n)
        # Same temporal separation as the total-burn watchlist: baseline and recent
        # windows must not overlap, otherwise a diverging tail hides its own shift.
        if n < n_base + recent_n:
            continue
        asym = g["_asym"].to_numpy()
        base = float(np.median(asym[:n_base]))
        recent = float(np.mean(asym[-recent_n:]))
        shift = recent - base
        # Robust flight-to-flight scatter of the asymmetry = its own noise floor.
        mad = float(np.median(np.abs(asym - np.median(asym))))
        noise = 1.4826 * mad
        thresh = max(_ASYM_NOISE_K * noise, 1e-9)
        if shift > thresh:
            label = "ENG1"
        elif shift < -thresh:
            label = "ENG2"
        else:
            label = "—"
        out[tail] = {"label": label, "shift": shift, "thresh": thresh,
                     "base": base, "recent": recent, "n": n}
    return out


# Build the watchlist once, up front; both the headline KPI (c4) and Section 6
# read this exact result, so they can never tell different stories.
wl_all, _wl_excluded = pd.DataFrame(), 0
if cruise_cols and AC_COL and "date" in df.columns:
    wl_all, _wl_excluded = _build_cruise_watchlist(
        df[["date", _METRIC, AC_COL]], _METRIC, AC_COL,
        _MIN_BASELINE_FLIGHTS, _AMBER_FACTOR, _RED_FACTOR,
    )

# ── KPIs ───────────────────────────────────────────────────
avg_cruise = df["_cruise_total"].mean() if cruise_cols else 0
total_burn_cols = list(burn_cols.keys())
avg_total = df[total_burn_cols].sum(axis=1).mean() if total_burn_cols else 0
n_flights = len(df)

# Headline count shares Section 6's own-baseline math: the red + amber tiers of
# the same watchlist, so this number always matches the MSNs named there.
n_rising = None
if not wl_all.empty:
    n_rising = int(wl_all["Status"].isin([">+10%", ">+5%"]).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flights analysed", f"{n_flights:,}")
c2.metric("Avg cruise fuel (kg)", f"{avg_cruise:.0f}" if cruise_cols else "—")
c3.metric("Avg total fuel per flight (kg)", f"{avg_total:.0f}" if total_burn_cols else "—")
c4.metric(
    "Aircraft with rising cruise-burn trend (>5%)",
    f"{n_rising}" if n_rising is not None else "—",
    help=(
        "Count of aircraft whose recent cruise burn sits more than 5% above their "
        "OWN historical baseline — the red + amber tiers of the Section 6 Cruise-Burn "
        "Watchlist. This uses the exact same own-baseline calculation as that "
        "watchlist (not a separate trend fit), so the headline number always agrees "
        "with the specific MSNs named there."
    ),
)

st.divider()

# ── Section 1: Fuel distribution by phase ─────────────────────────────
st.subheader(":material/local_gas_station: 1. Where Is Fuel Burned?")
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
st.subheader(":material/local_gas_station: 2. Cruise Efficiency per Aircraft")
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
    _n_eff_total = len(eff)
    eff = eff.head(10)  # top 10 highest burners — the actionable end of the ranking
    if _n_eff_total > 10:
        st.caption(f"Top 10 highest cruise burners of {_n_eff_total} aircraft (fleet avg from all).")
    eff["color"] = eff["Avg Cruise Fuel (kg)"].apply(
        lambda x: "#ef4444" if x > fleet_mean * 1.05 else "#22c55e"
    )

    fig_eff = go.Figure(go.Bar(
        x=eff["Avg Cruise Fuel (kg)"],
        y=eff[AC_COL].astype(str).map(_dnm),
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
st.subheader(":material/trending_up: 3. Cruise Fuel Trend Over Time")
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
st.subheader(":material/trending_up: 4. Fleet Monthly Fuel Trend")
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
st.subheader(":material/trending_up: 5. Per-Aircraft Cruise-Burn Degradation vs Own Baseline")

# Reads the shared _USE_RATE/_METRIC/_UNIT/_Y_TITLE method selected above the KPI row.
if _USE_RATE:
    st.caption(
        "Pick a tail to compare its per-flight cruise burn RATE against its OWN "
        "historical baseline (median of its earliest flights). The signal is "
        "normalized by cruise time (kg/h), so route length no longer confounds "
        "engine health. Amber band = 5–10% above baseline; red zone = more than "
        "10% above. This anchors the signal to each aircraft's own history instead "
        "of a fleet-relative threshold."
    )
else:
    st.caption(
        "Pick a tail to compare its per-flight cruise burn against its OWN historical "
        "baseline (median of its earliest flights). Amber band = 5–10% above baseline; "
        "red zone = more than 10% above. This anchors the signal to each aircraft's own "
        "history instead of a fleet-relative threshold. Note: cruise-duration data "
        "(time_sec_cruise) is unavailable, so this uses absolute cruise kg — longer "
        "routes may read high regardless of engine health."
    )

if not (cruise_cols and AC_COL and "date" in df.columns):
    st.info("Per-aircraft baseline analysis requires cruise fuel, aircraft and date columns.")
else:
    df_base = df.dropna(subset=["date", _METRIC, AC_COL]).copy()
    tails = sorted(df_base[AC_COL].unique().tolist())

    if not tails:
        st.info("No flights with valid cruise fuel data in the selected period.")
    else:
        chosen = st.selectbox("Select aircraft", options=tails, key="deg_msn", format_func=_dnm)
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

st.divider()

# ── Section 6: Cruise-Burn Watchlist — Aircraft Above Their Own Baseline ────
st.subheader(":material/visibility: 6. Cruise-Burn Watchlist — Aircraft Above Their Own Baseline")
st.caption(
    "Ranks tails by how far their recent cruise burn sits above their OWN historical "
    "baseline (rate-normalized by cruise time when available). This is the fleet-level "
    "companion to the single-aircraft view in Section 5 and a per-tail-baseline "
    "alternative to Section 2's fleet-relative ranking — it names the specific MSNs "
    "behind the headline 'rising cruise-burn trend' count above. ADVISORY / monitor "
    "signal only: cruise fuel burn is still confounded by weight, altitude and wind, "
    "so it is NOT equivalent to a trained-model removal alert."
)

if not (cruise_cols and AC_COL and "date" in df.columns):
    st.info("Cruise-burn watchlist requires cruise fuel, aircraft and date columns.")
else:
    # Reuse the watchlist already built above the KPI row — the same wl_all/_wl_excluded
    # that backs the c4 headline count, so the two views never diverge.
    wl, n_excluded = wl_all, _wl_excluded

    if wl.empty:
        st.success("No aircraft is currently burning above its own cruise baseline.")
    else:
        n_red = int((wl["Status"] == ">+10%").sum())
        n_amber = int((wl["Status"] == ">+5%").sum())
        wc1, wc2 = st.columns(2)
        wc1.metric("Aircraft >10% above own baseline", n_red)
        wc2.metric("Aircraft 5–10% above", n_amber)

        wl_sorted = wl.sort_values("pct_above", ascending=False)
        top = wl_sorted.head(12).reset_index(drop=True)

        caption_parts = []
        if len(wl_sorted) > 12:
            caption_parts.append(f"Top 12 of {len(wl_sorted)} analysed")
        if n_excluded:
            caption_parts.append(f"{n_excluded} excluded for thin/overlapping history")
        if caption_parts:
            st.caption("; ".join(caption_parts) + ".")

        # Isolate WHICH engine to inspect via the ENG1−ENG2 cruise-burn asymmetry
        # shift — the per-flight delta cancels the weight/altitude/wind confounders
        # both engines see together, unlike the per-engine own-baseline that fired
        # 'Both' whenever the aircraft simply flew heavier/higher.
        asym_map = (
            _engine_asymmetry_map(df, _ENG_METRIC, AC_COL, _MIN_BASELINE_FLIGHTS)
            if len(_ENG_METRIC) >= 2 else {}
        )

        def _engine_diverging(tail):
            info = asym_map.get(tail)
            if not info or info["label"] == "—":
                return "—"
            sign = "+" if info["shift"] >= 0 else "−"
            return f"{info['label']}  {sign}{abs(info['shift']):.0f} {_UNIT}"

        disp = pd.DataFrame({
            "Aircraft": top["Aircraft"].map(_dnm),
            "Status": top["Status"],
            "Engine diverging (ENG1−ENG2 shift)": top["Aircraft"].map(_engine_diverging),
            f"Recent {_UNIT}": top["Recent"],
            f"Own baseline {_UNIT}": top["Baseline"],
            "% above": top["pct_above"],
            "Flights": top["Flights"],
        })

        def _row_tint(row):
            s = str(row["Status"])
            if s == ">+10%":
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            if s == ">+5%":
                return ["background-color: rgba(251,191,36,0.15)"] * len(row)
            return [""] * len(row)

        styled = (
            disp.style
            .apply(_row_tint, axis=1)
            .format({
                f"Recent {_UNIT}": "{:.0f}",
                f"Own baseline {_UNIT}": "{:.0f}",
                "% above": "{:+.1f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "'Engine diverging (ENG1−ENG2 shift)' isolates which engine to inspect first "
            "from the per-flight ENG1−ENG2 cruise-burn-rate difference. Both engines fly "
            "the SAME weight, altitude and wind on each flight, so their difference cancels "
            "those shared confounders — a shift of that difference away from the tail's own "
            "early baseline, beyond its flight-to-flight noise, points to the diverging "
            "engine (e.g. borescope ENG2). The signed value is the (ENG1−ENG2) shift "
            "magnitude: positive → ENG1 diverging, negative → ENG2. '—' means symmetric — "
            "the rise is NOT isolatable to one engine (both burning more together, as a "
            "heavier/higher flight causes) rather than a hidden confounder being attributed "
            "to a single engine."
        )

        # Which-engine evidence: per-engine cruise-burn rate over time for the single
        # worst-flagged tail, with the gap between the two engines shaded so the
        # divergence that drives the ENGx recommendation is visible, not a lone word.
        if len(_ENG_METRIC) >= 2:
            worst_tail = wl_sorted.iloc[0]["Aircraft"]
            m1, m2 = _ENG_METRIC[1], _ENG_METRIC[2]
            gw = (
                df[df[AC_COL] == worst_tail]
                .dropna(subset=["date", m1, m2])
                .sort_values("date")
            )
            if len(gw) >= 2:
                st.markdown(
                    f"**Which engine is diverging — {_dnm(worst_tail)} "
                    "(worst flagged tail)**"
                )
                info = asym_map.get(worst_tail, {})
                fig_asym = go.Figure()
                fig_asym.add_trace(go.Scatter(
                    x=gw["date"], y=gw[m1], mode="lines+markers",
                    name="ENG1 cruise-burn rate",
                    line=dict(color="#3b82f6", width=2), marker=dict(size=5),
                    hovertemplate="ENG1 %{x|%d-%b-%y}: %{y:.0f} " + _UNIT + "<extra></extra>",
                ))
                # fill='tonexty' shades the area between ENG2 and ENG1 = the divergence.
                fig_asym.add_trace(go.Scatter(
                    x=gw["date"], y=gw[m2], mode="lines+markers",
                    name="ENG2 cruise-burn rate",
                    line=dict(color="#f97316", width=2), marker=dict(size=5),
                    fill="tonexty", fillcolor="rgba(148,163,184,0.22)",
                    hovertemplate="ENG2 %{x|%d-%b-%y}: %{y:.0f} " + _UNIT + "<extra></extra>",
                ))
                fig_asym.update_layout(
                    title=f"{_dnm(worst_tail)} — per-engine cruise-burn rate "
                          "(shaded gap = ENG1−ENG2 divergence)",
                    xaxis_title="", yaxis_title=_Y_TITLE,
                    height=340,
                    xaxis=dict(tickformat="%d-%b-%y"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig_asym, use_container_width=True)
                if info and info.get("label", "—") != "—":
                    st.caption(
                        f"ENG1−ENG2 asymmetry shifted from a baseline of "
                        f"{info['base']:+.0f} {_UNIT} to {info['recent']:+.0f} {_UNIT} "
                        f"recently — a {info['shift']:+.0f} {_UNIT} move beyond its "
                        f"{info['thresh']:.1f} {_UNIT} noise band, isolating "
                        f"{info['label']} as the diverging engine → inspect "
                        f"{info['label']} first."
                    )
                else:
                    st.caption(
                        "Both engine rates track together with no baseline-beating "
                        "divergence — the rise is symmetric, so it is not isolatable to a "
                        "single engine here (consistent with a heavier/higher flight, not "
                        "one-engine degradation)."
                    )
