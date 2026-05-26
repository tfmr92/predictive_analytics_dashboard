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

from utils.drive_loader import load

st.set_page_config(page_title="Wheels & Brakes", layout="wide")

# ── AMM constants (ATA 32, MTM-0051-00-Vol18) ─────────────────────────────────
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
        return "🔴 Severe — Inspect (AMM 05-50-03)"
    elif g >= g_lim:
        return "🟡 Hard — Monitor"
    else:
        return "🟢 Normal"


st.title("🛞 Wheels & Brakes — ATA 32")
st.markdown(
    "Tracks wheel and brake health across 6 gear positions. "
    "Hard landing severity uses a **weight-adjusted threshold** per AMM MPP7166_05-50-03 "
    f"(~{G_LIMIT_AT_MLW} g at MLW / ~{G_LIMIT_AT_MLW + G_LIMIT_DELTA:.2f} g light). "
    "Positions requiring formal AMM inspection are highlighted."
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
        "Assumed wheel life (cycles)", min_value=500, max_value=5000, value=WHEEL_LIFE_DEF,
        step=100,
        help="Carbon brake removal threshold. Adjust per your Maintenance Manual.",
    )
    st.divider()
    st.subheader("AMM Reference")
    st.caption(f"Hard landing: **MPP7166_05-50-03**")
    st.caption(f"Wheel overspeed: **MPP7166_05-50-30** ({TIRE_SPEED_MAX} kts)")
    st.caption(f"LG down overspeed: **MPP7166_05-50-27**")
    st.caption(f"Threshold at MLW: **{G_LIMIT_AT_MLW:.1f} g** ({MLW_KG:,} kg)")
    st.caption(f"Threshold at {MZFW_KG:,} kg: **{_weight_adjusted_g_limit(MZFW_KG):.2f} g**")

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
if selected_ac and "ac_sn" in df.columns:
    df = df[df["ac_sn"].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Weight-adjusted hard landing flags ────────────────────────────────────────
for acol, fcol in [("NormAccel_lh", "_hl_flag_lh"), ("NormAccel_rh", "_hl_flag_rh")]:
    if acol in df.columns and "gross_weight" in df.columns:
        df["_g_lim"] = df["gross_weight"].apply(_weight_adjusted_g_limit)
        df[fcol] = df[acol] >= df["_g_lim"]
    elif acol in df.columns:
        df[fcol] = df[acol] >= G_LIMIT_AT_MLW

# ── Position mapping ──────────────────────────────────────────────────────────
_POSITIONS = {
    "mlg1":   ("MLG 1 — LH Fwd",  "prediction_mlg1",   "time_since_installation_1"),
    "mlg2":   ("MLG 2 — LH Aft",  "prediction_mlg2",   "time_since_installation_2"),
    "mlg3":   ("MLG 3 — RH Fwd",  "prediction_mlg3",   "time_since_installation_3"),
    "mlg4":   ("MLG 4 — RH Aft",  "prediction_mlg4",   "time_since_installation_4"),
    "nlg_lh": ("NLG — LH",        "prediction_nlg_lh", "time_since_installation_5"),
    "nlg_rh": ("NLG — RH",        "prediction_nlg_rh", "time_since_installation_6"),
}

# ── KPIs ──────────────────────────────────────────────────────────────────────
pred_cols = [v[1] for v in _POSITIONS.values() if v[1] in df.columns]
total_alerts = int(df[pred_cols].eq(1).any(axis=1).sum()) if pred_cols else 0
ac_in_alert = set()
if "ac_sn" in df.columns and pred_cols:
    ac_in_alert = set(df.loc[df[pred_cols].eq(1).any(axis=1), "ac_sn"].dropna().unique())

hard_lh = int(df["_hl_flag_lh"].sum()) if "_hl_flag_lh" in df.columns else 0
hard_rh = int(df["_hl_flag_rh"].sum()) if "_hl_flag_rh" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("✈️ Aircraft with removal alert", len(ac_in_alert))
c2.metric("🔴 Flights with removal alert", total_alerts)
c3.metric("⚠️ Hard landings — LH", hard_lh,
          help="Weight-adjusted threshold (AMM MPP7166_05-50-03)")
c4.metric("⚠️ Hard landings — RH", hard_rh,
          help="Weight-adjusted threshold (AMM MPP7166_05-50-03)")

st.divider()

# ── Section 1: Removal Priority Table ─────────────────────────────────────────
st.subheader("1. Removal Priority — Wheels to Act On")
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
                "Current Alert": "🔴 REMOVE" if alert else "✅ OK",
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

# ── Section 2: Degradation Heatmap ────────────────────────────────────────────
st.subheader("2. Alert Rate Heatmap — MSN × Wheel Position")
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
        title="Wheel Removal Alert Rate (%) — MSN × Position",
    )
    fig_heat.update_layout(height=max(300, len(heatmap_data) * 30))
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Section 3: Cycles In Service ──────────────────────────────────────────────
st.subheader("3. Cycles In Service — Progress Toward Removal Threshold")
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

# ── Section 4: Hard Landing Assessment ────────────────────────────────────────
st.subheader("4. Hard Landing Assessment — AMM MPP7166_05-50-03")
st.caption(
    "Scatter: peak G-force vs. gross weight. The **curved threshold line** approximates the "
    "weight-dependent envelope from the AMM. Points above it trigger a formal inspection. "
    "Note: exact limits are in CGM figures in the SGML AMM; this line is an approximation."
)

tab_lh_land, tab_rh_land, tab_bounce = st.tabs(
    ["Left Main Gear (LH)", "Right Main Gear (RH)", "Bounce Count"]
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

        # Per-MSN inspection event count
        if "ac_sn" in df_land.columns:
            insp_counts = (
                df_land[df_land["Severity"] != "Normal"]
                .groupby(["ac_sn", "Severity"])
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
            if not insp_counts.empty:
                fig_bar = px.bar(
                    insp_counts, x="ac_sn", y="Count",
                    color="Severity",
                    color_discrete_map=color_map,
                    labels={"ac_sn": "MSN"},
                    title=f"Hard/Severe landing events per aircraft — {label}",
                )
                fig_bar.update_layout(height=280, legend_title_text="")
                st.plotly_chart(fig_bar, use_container_width=True)

        # AMM action card for events requiring inspection
        inspect_events = df_land[df_land["Severity"].str.contains("Inspect", na=False)]
        if not inspect_events.empty:
            with st.expander(
                f"🔴 {len(inspect_events)} event(s) above inspection threshold — AMM action required"
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
        df_bounce = df.dropna(subset=bounce_cols + ["date"]).copy()

        for bcol, side in [("bouncing_count_lh", "LH"), ("bouncing_count_rh", "RH")]:
            if bcol not in df_bounce.columns:
                continue
            fig_b = px.scatter(
                df_bounce, x="date", y=bcol,
                color="ac_sn",
                trendline="lowess",
                trendline_scope="overall",
                trendline_color_override="black",
                opacity=0.55,
                labels={bcol: f"Bounce Count ({side})", "date": "", "ac_sn": "MSN"},
                title=f"Bounce Count Over Time — {side}",
            )
            _b_alert = df_bounce[bcol].mean() + df_bounce[bcol].std()
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
