"""
Wheels & Brakes — ATA 32 landing gear health and removal forecasting.

Hard landing assessment uses a weight-adjusted threshold approximating the
AMM multi-variable envelope (MPP7166_05-50-03). Exact limits are weight-dependent
CGM figures in the SGML AMM; the formula here is:
    g_limit = 2.0 + max(0, (MLW_KG - gross_weight) / MLW_KG) * 0.25
yielding ~2.0 g at MLW (48,000 kg) and ~2.25 g at light weights.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utils.drive_loader import load, make_prefix_map, display_name, clean_df

st.set_page_config(page_title="Wheels & Brakes", layout="wide")

# ── AMM constants (ATA 32, MTM-0051-00-Vol18) ─────────────────────────────
MLW_KG          = 48_000   # E195-E2 max landing weight (kg)
MZFW_KG         = 40_200   # max zero fuel weight — approximate lower bound
G_LIMIT_AT_MLW  = 2.0      # inspection trigger at MLW (conservative)
G_LIMIT_DELTA   = 0.25     # additional allowance at lighter weights
WHEEL_LIFE_DEF  = 1200     # assumed carbon brake removal limit (cycles)
TIRE_SPEED_MAX  = 195.5    # max tire speed (kts) — AMM MPP7166_05-50-30


def _weight_adjusted_g_limit(gross_weight_kg: float) -> float:
    """Approximate weight-dependent hard landing threshold (AMM MPP7166_05-50-03)."""
    weight_ratio = max(0.0, (MLW_KG - gross_weight_kg) / MLW_KG)
    return G_LIMIT_AT_MLW + weight_ratio * G_LIMIT_DELTA


def _hard_landing_severity(row, acol: str) -> str:
    g = row.get(acol, 0)
    gw = row.get("gross_weight", MLW_KG)
    g_lim = _weight_adjusted_g_limit(gw)
    if g >= g_lim + 0.3:
        return "Severe — Inspect (AMM 05-50-03)"
    elif g >= g_lim:
        return "Hard — Monitor"
    else:
        return "Normal"


st.title(":material/build: Wheels & Brakes — ATA 32")
st.markdown(
    "Tracks wheel and brake health across 6 gear positions. "
    "Hard landing severity uses a **weight-adjusted threshold** per AMM MPP7166_05-50-03 "
    f"(~{G_LIMIT_AT_MLW} g at MLW / ~{G_LIMIT_AT_MLW + G_LIMIT_DELTA:.2f} g light). "
    "Positions requiring formal AMM inspection are highlighted."
)

# ── Data ──────────────────────────────────────────────────────
df = load("e2_wnb_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_wheel_brake_report` job in Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
if "ac_sn" in df.columns:
    df["ac_sn"] = df["ac_sn"].astype(str)

# Filter future dates and invalid serials; build display names
_prefix_map = make_prefix_map()
df = clean_df(df, date_col="date", ac_col="ac_sn", prefix_map=_prefix_map)
if "ac_sn" in df.columns:
    df["_display"] = df["ac_sn"].map(lambda msn: display_name(msn, _prefix_map))
_disp_col = "_display" if "_display" in df.columns else "ac_sn"

# ── Data freshness indicator ─────────────────────────────────────
if "date" in df.columns:
    _latest_event = df["date"].max()
    if pd.notna(_latest_event):
        _age_days = (pd.Timestamp.now().normalize() - _latest_event.normalize()).days
        _freshness_msg = (
            f"Latest flight event: **{_latest_event:%d-%b-%Y}** "
            f"({_age_days} day(s) ago)"
        )
        if _age_days <= 2:
            st.success(_freshness_msg)
        else:
            st.warning(f"{_freshness_msg} — wheel/brake data may be stale.")
    else:
        st.info("No valid flight dates in the loaded dataset.")

# ── Sidebar controls ──────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    days_back = st.slider("Days of history", 30, 365, 120)
    all_ac = sorted(df[_disp_col].dropna().unique().tolist()) if _disp_col in df.columns else []
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=all_ac)
    WHEEL_LIFE = st.number_input(
        "Assumed wheel life (cycles)", min_value=500, max_value=5000, value=WHEEL_LIFE_DEF,
        step=100,
        help="Carbon brake removal threshold. Adjust per your Maintenance Manual.",
    )
    st.divider()
    st.subheader(":material/straighten: AMM Reference")
    st.caption(f"Hard landing: **MPP7166_05-50-03**")
    st.caption(f"Wheel overspeed: **MPP7166_05-50-30** ({TIRE_SPEED_MAX} kts)")
    st.caption(f"LG down overspeed: **MPP7166_05-50-27**")
    st.caption(f"Threshold at MLW: **{G_LIMIT_AT_MLW:.1f} g** ({MLW_KG:,} kg)")
    st.caption(f"Threshold at {MZFW_KG:,} kg: **{_weight_adjusted_g_limit(MZFW_KG):.2f} g**")

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
if selected_ac and _disp_col in df.columns:
    df = df[df[_disp_col].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Weight-adjusted hard landing flags ───────────────────────────────
for acol, fcol in [("NormAccel_lh", "_hl_flag_lh"), ("NormAccel_rh", "_hl_flag_rh")]:
    if acol in df.columns and "gross_weight" in df.columns:
        df["_g_lim"] = df["gross_weight"].apply(_weight_adjusted_g_limit)
        df[fcol] = df[acol] >= df["_g_lim"]
    elif acol in df.columns:
        df[fcol] = df[acol] >= G_LIMIT_AT_MLW

# ── Position mapping ──────────────────────────────────────────
_POSITIONS = {
    "mlg1":   ("MLG 1 — LH Fwd",  "prediction_mlg1",   "time_since_installation_1"),
    "mlg2":   ("MLG 2 — LH Aft",  "prediction_mlg2",   "time_since_installation_2"),
    "mlg3":   ("MLG 3 — RH Fwd",  "prediction_mlg3",   "time_since_installation_3"),
    "mlg4":   ("MLG 4 — RH Aft",  "prediction_mlg4",   "time_since_installation_4"),
    "nlg_lh": ("NLG — LH",        "prediction_nlg_lh", "time_since_installation_5"),
    "nlg_rh": ("NLG — RH",        "prediction_nlg_rh", "time_since_installation_6"),
}

# ── KPIs ───────────────────────────────────────────────────────
pred_cols = [v[1] for v in _POSITIONS.values() if v[1] in df.columns]
total_alerts = int(df[pred_cols].eq(1).any(axis=1).sum()) if pred_cols else 0
ac_in_alert = set()
if "ac_sn" in df.columns and pred_cols:
    ac_in_alert = set(df.loc[df[pred_cols].eq(1).any(axis=1), "ac_sn"].dropna().unique())

hard_lh = int(df["_hl_flag_lh"].sum()) if "_hl_flag_lh" in df.columns else 0
hard_rh = int(df["_hl_flag_rh"].sum()) if "_hl_flag_rh" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aircraft with removal alert", len(ac_in_alert))
c2.metric("Flights with removal alert", total_alerts)
c3.metric("Hard landings — LH", hard_lh,
          help="Weight-adjusted threshold (AMM MPP7166_05-50-03)")
c4.metric("Hard landings — RH", hard_rh,
          help="Weight-adjusted threshold (AMM MPP7166_05-50-03)")

st.divider()

# ── Section 1: Removal Priority Table ────────────────────────────────
st.subheader(":material/build: 1. Removal Priority — Wheels to Act On")
st.caption(
    "Sorted by urgency: current alert first, then highest alert rate. "
    f"Remaining cycles = {WHEEL_LIFE}-cycle life limit minus cycles in service."
)

rows = []
if "ac_sn" in df.columns:
    for pos_key, (pos_label, pred_col, tsi_col) in _POSITIONS.items():
        if pred_col not in df.columns:
            continue
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
            alert_rate = float(df[df["ac_sn"] == ac][pred_col].eq(1).mean() * 100)
            rows.append({
                "MSN": ac,
                "Position": pos_label,
                "Current Alert": "REMOVE" if alert else "OK",
                "Alert Rate (%)": round(alert_rate, 1),
                "Cycles In Service": round(tsi) if tsi is not None else "—",
                "Est. Remaining Cycles": round(remaining) if remaining is not None else "—",
            })

if rows:
    priority_df = pd.DataFrame(rows)
    priority_df["_sort"] = priority_df["Current Alert"].apply(lambda x: 0 if "REMOVE" in x else 1)
    priority_df = priority_df.sort_values(
        ["_sort", "Alert Rate (%)"], ascending=[True, False]
    ).drop(columns="_sort")

    def _color_priority(row):
        if "REMOVE" in row["Current Alert"]:
            return ["background-color: rgba(239,68,68,0.15)"] * len(row)
        elif row["Alert Rate (%)"] > 20:
            return ["background-color: rgba(245,158,11,0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        priority_df.style.apply(_color_priority, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Prediction columns not found. Run the Dagster pipeline to generate predictions.")

st.divider()

# ── Section 2: Degradation Heatmap ──────────────────────────────────
st.subheader(":material/grid_view: 2. Alert Rate Heatmap — MSN × Wheel Position")
st.caption(
    "Color is fixed to an absolute 0–100% scale, so a 5% cell never reads as red "
    "as a 90% one. Each cell shows the exact alert rate."
)

available_preds = [v[1] for v in _POSITIONS.values() if v[1] in df.columns]
pos_labels = {v[1]: v[0] for v in _POSITIONS.values()}

if available_preds and "ac_sn" in df.columns:
    heatmap_data = (
        df.groupby("ac_sn")[available_preds]
        .apply(lambda g: (g == 1).mean() * 100)
        .rename(columns=pos_labels)
        .reset_index()
    )
    matrix = heatmap_data.set_index("ac_sn")
    if matrix.empty or matrix.to_numpy().max() <= 0:
        st.info("No positive alert rates in the current selection — all positions are clear.")
    else:
        fig_heat = px.imshow(
            matrix,
            color_continuous_scale=["#dcfce7", "#fef9c3", "#fca5a5", "#ef4444"],
            zmin=0, zmax=100,
            text_auto=".0f",
            aspect="auto",
            labels={"x": "Position", "y": "MSN", "color": "Alert Rate (%)"},
            title="Wheel Removal Alert Rate (%) — MSN × Position",
        )
        fig_heat.update_layout(height=max(300, len(heatmap_data) * 30))
        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Section 3: Cycles In Service ────────────────────────────────────
st.subheader(":material/build: 3. Cycles In Service — Progress Toward Removal Threshold")
st.caption(
    f"Red dashed line = {WHEEL_LIFE}-cycle removal threshold. "
    "Aircraft above the line must be scheduled for wheel/brake change."
)

tsi_cols_available = [(v[0], v[2]) for v in _POSITIONS.values() if v[2] in df.columns]
if tsi_cols_available and "ac_sn" in df.columns:
    tsi_tabs = st.tabs([label for label, _ in tsi_cols_available])
    for tab_w, (pos_label, tsi_col) in zip(tsi_tabs, tsi_cols_available):
        with tab_w:
            df_tsi = df.dropna(subset=["date", tsi_col, "ac_sn"]).copy()
            if df_tsi.empty:
                st.info("No data for this position.")
                continue
            fig_tsi = px.line(
                df_tsi, x="date", y=tsi_col, color="ac_sn",
                labels={tsi_col: "Cycles in Service", "date": "", "ac_sn": "MSN"},
                title=f"{pos_label} — Cycles in Service",
            )
            fig_tsi.add_hline(
                y=WHEEL_LIFE, line_dash="dash", line_color="red",
                annotation_text=f"Removal threshold ({WHEEL_LIFE} cycles)",
                annotation_position="top right",
            )
            fig_tsi.update_layout(
                height=320, xaxis=dict(tickformat="%d-%b-%y"), legend_title_text="MSN",
            )
            st.plotly_chart(fig_tsi, use_container_width=True)

st.divider()

# ── Section 4: Hard Landing Assessment ───────────────────────────────
st.subheader(":material/flight_land: 4. Hard Landing Assessment — AMM MPP7166_05-50-03")
st.caption(
    "Scatter: peak G-force vs. gross weight. The **curved threshold line** approximates the "
    "weight-dependent envelope from the AMM. Points above it trigger a formal inspection. "
    "Note: exact limits are in CGM figures in the SGML AMM; this line is an approximation."
)

tab_lh_land, tab_rh_land, tab_bounce = st.tabs(
    [":material/speed: Left Main Gear (LH)", ":material/speed: Right Main Gear (RH)", ":material/insights: Bounce Count"]
)

for tab_l, acol, impact_col, label in [
    (tab_lh_land, "NormAccel_lh", "lh_impact", "LH"),
    (tab_rh_land, "NormAccel_rh", "rh_impact", "RH"),
]:
    with tab_l:
        if acol not in df.columns:
            st.info("Column not available in current data.")
            continue

        df_land = df.dropna(subset=["date", acol]).copy()

        # Compute weight-adjusted severity
        if "gross_weight" in df_land.columns:
            df_land["_g_lim"] = df_land["gross_weight"].apply(_weight_adjusted_g_limit)
            df_land["Severity"] = df_land.apply(
                lambda r: ("Severe — Inspect" if r[acol] >= r["_g_lim"] + 0.3
                           else ("Hard — Monitor" if r[acol] >= r["_g_lim"]
                                 else "Normal")),
                axis=1,
            )
        else:
            df_land["Severity"] = pd.cut(
                df_land[acol],
                bins=[-999, G_LIMIT_AT_MLW, G_LIMIT_AT_MLW + 0.3, 999],
                labels=["Normal", "Hard — Monitor", "Severe — Inspect"],
            ).astype(str)

        color_map = {
            "Normal": "#22c55e",
            "Hard — Monitor": "#f59e0b",
            "Severe — Inspect": "#ef4444",
        }

        if "gross_weight" in df_land.columns:
            # Primary chart: G vs. weight with threshold envelope
            fig_gw = px.scatter(
                df_land, x="gross_weight", y=acol,
                color="Severity",
                color_discrete_map=color_map,
                symbol="ac_sn",
                hover_data={"ac_sn": True, "date": True},
                labels={acol: f"Peak G — {label}", "gross_weight": "Gross Weight (kg)", "ac_sn": "MSN"},
                title=f"G-force vs. Gross Weight — {label} (AMM threshold envelope)",
            )
            # Overlay weight-adjusted threshold curve
            gw_range = np.linspace(
                df_land["gross_weight"].min() * 0.95,
                df_land["gross_weight"].max() * 1.05,
                100,
            )
            g_envelope = [_weight_adjusted_g_limit(gw) for gw in gw_range]
            g_severe   = [g + 0.30 for g in g_envelope]

            fig_gw.add_trace(go.Scatter(
                x=gw_range, y=g_envelope,
                mode="lines", name="Inspect threshold (approx.)",
                line=dict(color="#f59e0b", dash="dash", width=2),
            ))
            fig_gw.add_trace(go.Scatter(
                x=gw_range, y=g_severe,
                mode="lines", name="Severe threshold (approx.)",
                line=dict(color="#ef4444", dash="dot", width=2),
            ))
            fig_gw.update_layout(height=400, legend_title_text="")
            st.plotly_chart(fig_gw, use_container_width=True)

        # Secondary chart: G over time
        fig_time = px.scatter(
            df_land, x="date", y=acol,
            color="Severity",
            color_discrete_map=color_map,
            symbol="ac_sn",
            opacity=0.7,
            labels={acol: f"Peak G — {label}", "date": "", "ac_sn": "MSN"},
            title=f"Peak G-force Over Time — {label}",
        )
        _ref_g = G_LIMIT_AT_MLW
        fig_time.add_hline(
            y=_ref_g, line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Inspect (at MLW: {_ref_g:.1f} g)",
            annotation_position="top right",
        )
        fig_time.add_hline(
            y=_ref_g + 0.3, line_dash="dot", line_color="#ef4444",
            annotation_text=f"Severe (at MLW: {_ref_g + 0.3:.1f} g)",
            annotation_position="top right",
        )
        fig_time.update_traces(selector=dict(mode="markers"), marker_size=6)
        fig_time.update_layout(
            height=360, xaxis=dict(tickformat="%d-%b-%y"), legend_title_text="",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Per-MSN inspection event count — top 10 aircraft by total events
        if "ac_sn" in df_land.columns:
            insp_counts = (
                df_land[df_land["Severity"] != "Normal"]
                .groupby(["ac_sn", "Severity"])
                .size()
                .reset_index(name="Count")
            )
            if not insp_counts.empty:
                totals = insp_counts.groupby("ac_sn")["Count"].sum().sort_values(ascending=False)
                top_msns = totals.head(10).index.tolist()
                if len(totals) > 10:
                    st.caption(f"Top 10 of {len(totals)} aircraft with hard/severe events.")
                insp_top = insp_counts[insp_counts["ac_sn"].isin(top_msns)].copy()
                insp_top["Aircraft"] = insp_top["ac_sn"].map(
                    lambda m: display_name(m, _prefix_map)
                )
                fig_bar = px.bar(
                    insp_top, x="Aircraft", y="Count",
                    color="Severity",
                    color_discrete_map=color_map,
                    category_orders={"Aircraft": [display_name(m, _prefix_map) for m in top_msns]},
                    title=f"Hard/Severe landing events per aircraft — {label} (top 10)",
                )
                fig_bar.update_layout(height=300, legend_title_text="")
                st.plotly_chart(fig_bar, use_container_width=True)

        # AMM action card for events requiring inspection
        inspect_events = df_land[df_land["Severity"].str.contains("Inspect", na=False)]
        if not inspect_events.empty:
            with st.expander(
                f"{len(inspect_events)} event(s) above inspection threshold — AMM action required"
            ):
                st.warning(
                    "**AMM MPP7166_05-50-03** — *Do An Inspection After A Hard Landing*\n\n"
                    "Required TCRF parameters to confirm: `fdrAccelNormal-1a`, "
                    "`pitchAngle-1a`, `rollAttRate-1a`.\n\n"
                    "Cross-reference gross weight against the AMM weight-dependent envelope "
                    "(CGM figures in SGML AMM SDS)."
                )
                disp_cols = ["date", "ac_sn", acol, "gross_weight"] + (
                    ["_g_lim"] if "_g_lim" in inspect_events.columns else []
                ) + (["Severity"])
                disp_cols = [c for c in disp_cols if c in inspect_events.columns]
                st.dataframe(
                    inspect_events[disp_cols].rename(columns={
                        acol: f"Peak G ({label})",
                        "gross_weight": "Gross Wt (kg)",
                        "_g_lim": "AMM Limit (approx. g)",
                        "ac_sn": "MSN",
                        "date": "Date",
                    }).sort_values("Date", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_bounce:
    bounce_cols = [c for c in ("bouncing_count_lh", "bouncing_count_rh") if c in df.columns]
    if bounce_cols and "ac_sn" in df.columns:
        st.caption(
            "Bounce count (number of partial touchdowns per landing) accelerates tire wear "
            "and indicates poor approach energy management. Persistently high values warrant "
            "crew feedback and tire inspection."
        )
        df_bounce = df.dropna(subset=["date"]).copy()

        for bcol, side in [("bouncing_count_lh", "LH"), ("bouncing_count_rh", "RH")]:
            if bcol not in df_bounce.columns:
                continue
            df_side = df_bounce.dropna(subset=[bcol])
            # lowess trendline needs a handful of points; px crashes (IndexError) on zero traces
            if len(df_side) < 5:
                st.info(f"Not enough bounce-count data for {side} in the selected window.")
                continue
            fig_b = px.scatter(
                df_side, x="date", y=bcol,
                color="ac_sn",
                trendline="lowess",
                trendline_scope="overall",
                trendline_color_override="black",
                opacity=0.55,
                labels={bcol: f"Bounce Count ({side})", "date": "", "ac_sn": "MSN"},
                title=f"Bounce Count Over Time — {side}",
            )
            _b_alert = df_side[bcol].mean() + df_side[bcol].std()
            fig_b.add_hline(
                y=_b_alert, line_dash="dot", line_color="orange",
                annotation_text="Fleet alert level (mean + 1σ)",
                annotation_position="top right",
            )
            fig_b.update_traces(selector=dict(mode="markers"), marker_size=5)
            fig_b.update_layout(
                height=320, xaxis=dict(tickformat="%d-%b-%y"), legend_title_text="MSN",
            )
            st.plotly_chart(fig_b, use_container_width=True)

        # Monthly bounce rate per aircraft
        df_bounce["month"] = df_bounce["date"].dt.to_period("M").astype(str)
        for bcol, side in [("bouncing_count_lh", "LH"), ("bouncing_count_rh", "RH")]:
            if bcol not in df_bounce.columns:
                continue
            monthly = (
                df_bounce.groupby(["month", "ac_sn"])[bcol]
                .mean()
                .reset_index()
                .rename(columns={bcol: "Avg Bounce Count"})
            )
            fig_mb = px.bar(
                monthly, x="month", y="Avg Bounce Count",
                color="ac_sn",
                barmode="group",
                labels={"month": "", "ac_sn": "MSN"},
                title=f"Monthly Avg Bounce Count — {side}",
            )
            fig_mb.update_layout(height=280, legend_title_text="MSN")
            st.plotly_chart(fig_mb, use_container_width=True)
    else:
        st.info("Bounce count columns not available in current data.")

st.divider()

# ── Section 5: Life Analysis — Weibull ────────────────────────────────────────
st.subheader(":material/build: 5. Life Analysis — Component Removal History")
st.caption(
    "Weibull analysis of historical removal cycles (TRAX data). "
    "B10 = 10% of units removed by this cycle count; B50 = median removal life. "
    "Run the Snowflake notebook to refresh maintenance data."
)

try:
    from scipy.stats import weibull_min as _weibull
    import numpy as _np

    _df_wnb_m   = load("e2_wnb_maintenance.parquet")
    _df_brake_m = load("e2_brake_maintenance.parquet")

    _tab_wheel, _tab_brake = st.tabs([":material/build: Wheels (MLG/NLG)", ":material/build: Brakes"])

    for _tab, _df_m, _comp in [
        (_tab_wheel, _df_wnb_m,   "Wheel"),
        (_tab_brake, _df_brake_m, "Brake"),
    ]:
        with _tab:
            if _df_m.empty or "CYCLES_INSTALLED" not in _df_m.columns:
                st.info(
                    f"No {_comp.lower()} maintenance data. "
                    "Run the Snowflake notebook to upload `e2_wnb_maintenance.parquet` / "
                    "`e2_brake_maintenance.parquet` to Drive."
                )
                continue

            _cycles = (
                pd.to_numeric(_df_m["CYCLES_INSTALLED"], errors="coerce")
                .dropna()
                .pipe(lambda s: s[s > 0])
                .values
            )

            if len(_cycles) < 5:
                st.info(f"Not enough data for Weibull fitting ({len(_cycles)} records).")
                continue

            _shape, _loc, _scale = _weibull.fit(_cycles, floc=0)
            _b10 = _weibull.ppf(0.10, _shape, _loc, _scale)
            _b50 = _weibull.ppf(0.50, _shape, _loc, _scale)
            _x   = _np.linspace(0, _cycles.max() * 1.15, 300)
            _pdf = _weibull.pdf(_x, _shape, _loc, _scale)

            _col_h, _col_w = st.columns(2)

            with _col_h:
                fig_ch = px.histogram(
                    x=_cycles, nbins=25,
                    labels={"x": "Cycles at Removal", "count": "Removals"},
                    title=f"{_comp} — Removal Cycles Distribution",
                    color_discrete_sequence=["#64748b"],
                )
                fig_ch.add_vline(
                    x=WHEEL_LIFE, line_dash="dash", line_color="red",
                    annotation_text=f"Current threshold ({WHEEL_LIFE}c)",
                    annotation_position="top right",
                )
                fig_ch.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_ch, use_container_width=True)

            with _col_w:
                fig_wb = go.Figure()
                fig_wb.add_histogram(
                    x=_cycles, name="Observed removals",
                    histnorm="probability density",
                    marker_color="steelblue", opacity=0.55,
                )
                fig_wb.add_scatter(
                    x=_x, y=_pdf, mode="lines",
                    name=f"Weibull (β={_shape:.2f}, η={_scale:.0f}c)",
                    line=dict(color="#f59e0b", width=2.5),
                )
                fig_wb.add_vline(x=_b10, line_dash="dot", line_color="#ef4444",
                                 annotation_text=f"B10 = {_b10:.0f}c",
                                 annotation_position="top right")
                fig_wb.add_vline(x=_b50, line_dash="dot", line_color="#64748b",
                                 annotation_text=f"B50 = {_b50:.0f}c",
                                 annotation_position="top right")
                fig_wb.add_vline(x=WHEEL_LIFE, line_dash="dash", line_color="red",
                                 annotation_text=f"Maint. threshold ({WHEEL_LIFE}c)",
                                 annotation_position="bottom right")
                fig_wb.update_layout(
                    title=f"{_comp} Removal Life — Weibull  (n={len(_cycles)} events)",
                    xaxis_title="Cycles at Removal",
                    yaxis_title="Probability Density",
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_wb, use_container_width=True)

            st.caption(
                f"**B10 = {_b10:.0f} cycles** — 10% of {_comp.lower()}s removed by this point.  "
                f"**B50 = {_b50:.0f} cycles** — median removal life.  "
                f"Current threshold set to **{WHEEL_LIFE} cycles**."
            )

            # Per-position Weibull summary table
            if "POSITION" in _df_m.columns:
                _positions = _df_m["POSITION"].dropna().unique()
                if len(_positions) > 1:
                    _pos_rows = []
                    for _pos in sorted(_positions):
                        _pc = (
                            pd.to_numeric(
                                _df_m.loc[_df_m["POSITION"] == _pos, "CYCLES_INSTALLED"],
                                errors="coerce",
                            )
                            .dropna()
                            .pipe(lambda s: s[s > 0])
                            .values
                        )
                        if len(_pc) >= 3:
                            _ps, _pl, _psc = _weibull.fit(_pc, floc=0)
                            _pos_rows.append({
                                "Position": _pos,
                                "N": len(_pc),
                                "B10 (cycles)": round(_weibull.ppf(0.10, _ps, _pl, _psc)),
                                "B50 (cycles)": round(_weibull.ppf(0.50, _ps, _pl, _psc)),
                                "Mean (cycles)": round(float(_pc.mean())),
                            })
                    if _pos_rows:
                        st.dataframe(
                            pd.DataFrame(_pos_rows),
                            use_container_width=True, hide_index=True,
                        )

except ImportError:
    st.info("scipy not installed — add `scipy>=1.12.0` to requirements.txt.")

st.divider()

# ── Section 6: Model Track Record — Predicted vs Confirmed Removals ──────────────
st.subheader(":material/analytics: 6. Model Track Record — Predicted vs Confirmed Removals")

LEAD_WINDOW_DAYS = 90

# Map raw TRAX POSITION variants directly to the safe prediction-column key
# (prediction_<key>). Faithfully replicates the producer's _POSITION_MAP variants
# -> canonical 'MLG 1'..'RH NLG' -> _SAFE key in
# repositories/azul/save_wheel_brake_report/ops/_wheel_brake_data_prep.py.
# KEEP IN SYNC WITH PRODUCER: any new variant added there must be added here too.
_POSITION_TO_KEY = {
    # Canonical
    "MLG 1": "mlg1", "MLG 2": "mlg2", "MLG 3": "mlg3", "MLG 4": "mlg4",
    "LH NLG": "nlg_lh", "RH NLG": "nlg_rh",
    # No space
    "MLG1": "mlg1", "MLG2": "mlg2", "MLG3": "mlg3", "MLG4": "mlg4",
    "LHNLG": "nlg_lh", "RHNLG": "nlg_rh",
    # Number only
    "1": "mlg1", "2": "mlg2", "3": "mlg3", "4": "mlg4",
    # LH/RH with number
    "LH MLG 1": "mlg1", "LH MLG 2": "mlg2",
    "RH MLG 1": "mlg3", "RH MLG 2": "mlg4",
    "MLG LH 1": "mlg1", "MLG LH 2": "mlg2",
    "MLG RH 1": "mlg3", "MLG RH 2": "mlg4",
    # Inboard / Outboard
    "LH INBD": "mlg1", "LH OUTBD": "mlg2",
    "RH INBD": "mlg3", "RH OUTBD": "mlg4",
    "INBD LH": "mlg1", "OUTBD LH": "mlg2",
    "INBD RH": "mlg3", "OUTBD RH": "mlg4",
    "LH INBOARD": "mlg1", "LH OUTBOARD": "mlg2",
    "RH INBOARD": "mlg3", "RH OUTBOARD": "mlg4",
    # Forward / Aft
    "LH FWD": "mlg1", "LH AFT": "mlg2",
    "RH FWD": "mlg3", "RH AFT": "mlg4",
    "FWD LH": "mlg1", "AFT LH": "mlg2",
    "FWD RH": "mlg3", "AFT RH": "mlg4",
    # Wheel / Brake numbered
    "WHEEL 1": "mlg1", "WHEEL 2": "mlg2", "WHEEL 3": "mlg3", "WHEEL 4": "mlg4",
    "BRAKE 1": "mlg1", "BRAKE 2": "mlg2", "BRAKE 3": "mlg3", "BRAKE 4": "mlg4",
    "BRK 1": "mlg1", "BRK 2": "mlg2", "BRK 3": "mlg3", "BRK 4": "mlg4",
    "BRAKE1": "mlg1", "BRAKE2": "mlg2", "BRAKE3": "mlg3", "BRAKE4": "mlg4",
    # NLG variants
    "NLG LH": "nlg_lh", "NLG RH": "nlg_rh",
    "NLG": "nlg_lh",
}


def _resolve_pos_key(raw):
    """Resolve a raw TRAX POSITION string to its safe gear-position key
    (strip + upper lookup), or None when the position is absent from the map."""
    return _POSITION_TO_KEY.get(str(raw).strip().upper())


@st.cache_data(ttl=300)
def _load_track_record():
    """Cross-check model predictions against real TRAX removals — per position.

    Reloads the FULL unfiltered report (long format: exactly one prediction_<key>
    non-null per flight row). Each TRAX removal is attributed to its OWN gear
    position via _resolve_pos_key, then matched only against that position's
    prediction column within a 90-day lead window. Returns one row per removal
    with its catch status. Removals whose TRAX position is absent from the map
    ('Position not mapped') or that have no flights of that position in the
    window ('No telemetry in window') are non-evaluable and excluded from the
    catch rate.
    """
    rep = load("e2_wnb_report.parquet")
    if rep.empty:
        return pd.DataFrame()

    rep = rep.copy()
    rep["date"] = pd.to_datetime(rep.get("date"), errors="coerce")
    pred_cols = [c for c in rep.columns if c.startswith("prediction_")]

    _key = lambda s: str(s).split(".")[0].strip()[-5:]
    rep["_key"] = rep["ac_sn"].map(_key) if "ac_sn" in rep.columns else None
    rep = rep[["_key", "date"] + pred_cols]

    req_cols = {"AC_SN", "POSITION", "TRANSACTION_DATE"}
    frames = []
    for _fname, _comp in [
        ("e2_wnb_maintenance.parquet", "Wheel"),
        ("e2_brake_maintenance.parquet", "Brake"),
    ]:
        m = load(_fname)
        if m.empty or not req_cols.issubset(m.columns):
            continue
        m = m.copy()
        m["Component"] = _comp
        frames.append(m)

    if not frames:
        return pd.DataFrame()

    maint = pd.concat(frames, ignore_index=True)
    maint["TRANSACTION_DATE"] = pd.to_datetime(maint["TRANSACTION_DATE"], errors="coerce")
    maint["_key"] = maint["AC_SN"].map(_key)

    records = []
    for _, r in maint.iterrows():
        key = r["_key"]
        td = r["TRANSACTION_DATE"]
        if pd.isna(td) or not key:
            continue

        pos_key = _resolve_pos_key(r["POSITION"])
        if pos_key is None:
            status, lead, caught, evaluable = "Position not mapped", np.nan, False, False
        else:
            col = "prediction_" + pos_key
            win_start = td - pd.Timedelta(days=LEAD_WINDOW_DAYS)
            sub = rep[(rep["_key"] == key) & (rep["date"] >= win_start) & (rep["date"] <= td)]
            pos_rows = sub[sub[col].notna()] if col in sub.columns else sub.iloc[0:0]

            if pos_rows.empty:
                status, lead, caught, evaluable = "No telemetry in window", np.nan, False, False
            else:
                evaluable = True
                flagged = pos_rows[pos_rows[col] == 1]
                caught = not flagged.empty
                if caught:
                    status = "Caught"
                    lead = (td - flagged["date"].min()).days
                else:
                    status, lead = "Missed", np.nan

        reason = "—"
        for _rc in ("REMOVAL_REASON", "DEFECT_DESCRIPTION"):
            if _rc in r.index and pd.notna(r[_rc]) and str(r[_rc]).strip():
                reason = str(r[_rc]).strip()
                break

        records.append({
            "_key": key,
            "Component": r.get("Component", "—"),
            "Position": str(r["POSITION"]),
            "RemovalDate": td,
            "Status": status,
            "LeadDays": lead,
            "Reason": reason,
            "Evaluable": evaluable,
            "Caught": caught,
        })

    return pd.DataFrame(records)


_track = _load_track_record()

if _track.empty or not _track["Evaluable"].any():
    st.info(
        "No evaluable removals yet — this needs both model predictions "
        "(`e2_wnb_report.parquet`) and TRAX removals "
        "(`e2_wnb_maintenance.parquet` / `e2_brake_maintenance.parquet`) "
        "with telemetry coverage in the 90-day window before a removal. "
        "Run the Snowflake notebook and the wheel/brake report jobs to populate them."
    )
else:
    _eval = _track[_track["Evaluable"]]
    _n = int(len(_eval))
    _k = int(_eval["Caught"].sum())
    _rate = (_k / _n * 100) if _n else 0.0
    _leads = _eval.loc[_eval["Caught"], "LeadDays"].dropna()
    _median_lead = _leads.median() if not _leads.empty else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Removals evaluated", _n, help="Telemetry-covered removals only.")
    k2.metric("Caught in advance", _k)
    k3.metric("Catch rate", f"{_rate:.0f}%")
    k4.metric(
        "Median lead time",
        f"{_median_lead:.0f} d" if _median_lead is not None else "—",
        help="Days from first alert to removal, caught removals only.",
    )

    _disp = _track.sort_values("RemovalDate", ascending=False).head(30).copy()
    _disp["MSN"] = _disp["_key"].map(lambda m: display_name(m, _prefix_map))
    _table = pd.DataFrame({
        "MSN": _disp["MSN"].values,
        "Component": _disp["Component"].values,
        "Position": _disp["Position"].values,
        "Removal date": _disp["RemovalDate"].dt.strftime("%d-%b-%Y").values,
        "Status": _disp["Status"].values,
        "Lead time (days)": _disp["LeadDays"].values,
        "Removal reason": _disp["Reason"].values,
    })
    st.dataframe(_table, use_container_width=True, hide_index=True)

    st.caption(
        "Retrospective track record over the period the prediction report covers. "
        "Per-position attribution — each removal is matched only to its own "
        f"gear-position model within a {LEAD_WINDOW_DAYS}-day lead window. This is "
        "legitimate model validation (per-position trained models cross-checked "
        "against TRAX removals), not a live alert. '' rows are non-evaluable and "
        "excluded from the catch rate: 'No telemetry in window' has no flights of "
        "that position in the window, and 'Position not mapped' has a TRAX position "
        "absent from the model's position map — unmapped positions are excluded, "
        "not counted as a catch or a miss."
    )
