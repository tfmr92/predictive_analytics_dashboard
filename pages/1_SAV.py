"""
SAV — Starter Air Valve health monitoring
Displays the 5 key degradation signals per engine side (LH / RH).

Pre-failure threshold methodology (EDA, see "Threshold Analysis" tab):
for each degradation signal the population is split by the model's
pre-failure prediction and the two distributions are compared. A signal
supports the pre-failure call when the alert-class median sits beyond the
normal-class P75/P90. The same percentile bands drive the red "degradation
zone" shading on the trend charts.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, clean_df, make_prefix_map, display_name

st.set_page_config(page_title="SAV — Starter Air Valve", layout="wide")

st.title("⚙️ Starter Air Valve (SAV)")
st.markdown(
    "The starter valve opens to drive the turbine during engine start. "
    "As it degrades it opens **slower**, stays open **longer**, oscillates, "
    "and the engine reaches **lower peak N2** — all detectable before failure."
)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_parquet(filename: str) -> pd.DataFrame:
    return load(filename)

df_lh = _load_parquet("e2_sav_lh_report.parquet")
df_rh = _load_parquet("e2_sav_rh_report.parquet")

if df_lh.empty and df_rh.empty:
    st.error("No data yet. Run the `save_sav_report` job in Dagster.")
    st.stop()

for df in (df_lh, df_rh):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "ac_sn" in df.columns:
        df["ac_sn"] = df["ac_sn"].astype(str)

# Filter future dates and invalid serials
_prefix_map = make_prefix_map()
df_lh = clean_df(df_lh, date_col="date", ac_col="ac_sn", prefix_map=_prefix_map)
df_rh = clean_df(df_rh, date_col="date", ac_col="ac_sn", prefix_map=_prefix_map)

def _dnm(msn) -> str:
    return display_name(str(msn), _prefix_map)

# ── Full unfiltered fleet (safety alerts must never inherit the sidebar filter) ──
def _full_fleet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    return df.dropna(subset=["date"]).sort_values("date")

df_lh_full = _full_fleet(df_lh)
df_rh_full = _full_fleet(df_rh)

# ── Alert history: aircraft that have EVER been in predicted pre-failure ──────
def _alert_history(df: pd.DataFrame, pred_col: str) -> set:
    if df.empty or pred_col not in df.columns or "ac_sn" not in df.columns:
        return set()
    flagged = df.loc[df[pred_col].eq(1), "ac_sn"].astype(str)
    return set(flagged.unique())

ALERT_HISTORY_LH = _alert_history(df_lh_full, "pre_lh_sav_failure_prediction")
ALERT_HISTORY_RH = _alert_history(df_rh_full, "pre_rh_sav_failure_prediction")
ALERT_HISTORY_ANY = ALERT_HISTORY_LH | ALERT_HISTORY_RH

# ── Data freshness ──────────────────────────────────────────────────────────────
def _latest_flight_date(*dfs: pd.DataFrame) -> pd.Timestamp | None:
    dates = []
    for df in dfs:
        if not df.empty and "date" in df.columns:
            mx = df["date"].max()
            if pd.notna(mx):
                dates.append(mx)
    return max(dates) if dates else None

_latest_date = _latest_flight_date(df_lh, df_rh)
if _latest_date is not None:
    st.caption(f"Data through {_latest_date.strftime('%d-%b-%Y')} · auto-refreshed hourly")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 30, 365, 120)
    only_alert = st.checkbox(
        f"Only aircraft with pre-failure history ({len(ALERT_HISTORY_ANY)})",
        value=False,
        help="Restrict every chart to aircraft that have at least one flight "
             "with a predicted SAV pre-failure (LH or RH) in the loaded history.",
    )
    all_ac = sorted(
        set(df_lh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_lh.columns else [])
        | set(df_rh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_rh.columns else [])
    )
    if only_alert:
        all_ac = [m for m in all_ac if m in ALERT_HISTORY_ANY]
    selected_ac = st.multiselect(
        "Aircraft",
        options=all_ac,
        default=all_ac,
        format_func=_dnm,
    )

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

def _filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if selected_ac and "ac_sn" in df.columns:
        df = df[df["ac_sn"].isin(selected_ac)]
    if "date" in df.columns:
        df = df[df["date"] >= cutoff]
    return df.dropna(subset=["date"]).sort_values("date")

df_lh = _filter(df_lh)
df_rh = _filter(df_rh)

# ── KPIs ──────────────────────────────────────────────────────────────────────
def _alert_msns(df: pd.DataFrame, pred_col: str) -> list[str]:
    """MSNs whose latest flight predicts pre-failure — computed over the given df."""
    if df.empty or pred_col not in df.columns or "ac_sn" not in df.columns:
        return []
    latest = df.sort_values("date").groupby("ac_sn").last()
    return sorted(latest.index[latest[pred_col].eq(1)].astype(str).tolist())

def _alert_count(df: pd.DataFrame, pred_col: str) -> int:
    return len(_alert_msns(df, pred_col))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aircraft monitored (LH)", df_lh["ac_sn"].nunique() if "ac_sn" in df_lh.columns else 0)
c2.metric("🔴 In alert — LH", _alert_count(df_lh_full, "pre_lh_sav_failure_prediction"))
c3.metric("Aircraft monitored (RH)", df_rh["ac_sn"].nunique() if "ac_sn" in df_rh.columns else 0)
c4.metric("🔴 In alert — RH", _alert_count(df_rh_full, "pre_rh_sav_failure_prediction"))

# ── Fleet-wide safety triage banner (ignores the sidebar filter) ────────────────
_lh_alert_msns = _alert_msns(df_lh_full, "pre_lh_sav_failure_prediction")
_rh_alert_msns = _alert_msns(df_rh_full, "pre_rh_sav_failure_prediction")

if _lh_alert_msns or _rh_alert_msns:
    lines = []
    if _lh_alert_msns:
        lines.append(
            "**LH pre-failure predicted:** "
            + ", ".join(_dnm(m) for m in _lh_alert_msns)
            + " — inspect starter air valve per AMM"
        )
    if _rh_alert_msns:
        lines.append(
            "**RH pre-failure predicted:** "
            + ", ".join(_dnm(m) for m in _rh_alert_msns)
            + " — inspect starter air valve per AMM"
        )
    st.error("🚨 Fleet safety triage\n\n" + "\n\n".join(lines))
else:
    st.success("✅ No SAV pre-failure predicted on the latest flight across the fleet.")
st.caption("Fleet-wide alert — based on every aircraft's latest flight; ignores the sidebar filter.")

st.divider()

# ── Signal definitions — maps friendly name → (lh_col, rh_col, unit, description) ─
SIGNALS = {
    "Valve Opening Time": (
        "time_to_open_ats_vlv_1", "time_to_open_ats_vlv_2",
        "seconds", "Rising trend indicates valve is stiffening (degraded actuator or contamination).",
        "up",  # "up" = rising is bad
    ),
    "Valve Closing / Response Time": (
        "time_with_ats_vlv_closed_and_rpm_above_0-1a", "time_with_ats_vlv_closed_and_rpm_above_0-3a",
        "seconds", "Time the valve remains closed while shaft is still spinning — prolonged = slow response.",
        "up",
    ),
    "Total Valve Open Time": (
        "time_with_ats_vlv_open-1a", "time_with_ats_vlv_open-3a",
        "seconds", "Abnormally long open time suggests valve is not closing cleanly or start is laboured.",
        "up",
    ),
    "Max N2 Speed at Start": (
        "max_ats_rpm_with_n2_above_50-1a", "max_ats_rpm_with_n2_above_50-3a",
        "%RPM", "Peak N2 reached during start — falling trend means less torque delivered (wear).",
        "down",  # "down" = falling is bad
    ),
    "Valve Oscillation Count": (
        "ats_oscillation-1a", "ats_oscillation-3a",
        "count", "Number of pressure oscillations during start — rising = unstable valve behavior.",
        "up",
    ),
}

# Fallback column names for oscillation (some datasets use different naming)
_OSCILLATION_FALLBACKS = {
    "ats_oscillation-1a": ["ats_mts_oscillation_count-1a", "oscillation_count_lh", "ats_osc_count-1a"],
    "ats_oscillation-3a": ["ats_mts_oscillation_count-3a", "oscillation_count_rh", "ats_osc_count-3a"],
}

def _resolve_col(df: pd.DataFrame, col: str) -> str | None:
    if col in df.columns:
        return col
    for alt in _OSCILLATION_FALLBACKS.get(col, []):
        if alt in df.columns:
            return alt
    return None


_FLEET_GREY = "#cbd5e1"
_FLEET_LABEL = "Fleet — no alert history"


def _trend_chart(df: pd.DataFrame, col: str, title: str, unit: str, bad_dir: str,
                 alert_set: set) -> go.Figure | None:
    """Scatter trend where aircraft with pre-failure history keep an individual
    color and the rest of the fleet is collapsed into a single grey series."""
    df_plot = df.dropna(subset=[col]).copy()
    if len(df_plot) < 5:
        return None

    df_plot["_legend"] = df_plot["ac_sn"].map(
        lambda m: f"⚠ {_dnm(m)}" if m in alert_set else _FLEET_LABEL
    )
    alert_labels = sorted(l for l in df_plot["_legend"].unique() if l != _FLEET_LABEL)
    palette = px.colors.qualitative.Set1
    color_map = {_FLEET_LABEL: _FLEET_GREY}
    color_map.update({l: palette[i % len(palette)] for i, l in enumerate(alert_labels)})

    fig = px.scatter(
        df_plot,
        x="date", y=col,
        color="_legend",
        color_discrete_map=color_map,
        category_orders={"_legend": [_FLEET_LABEL] + alert_labels},
        custom_data=["ac_sn"],
        labels={col: f"{title} ({unit})", "date": "", "_legend": "Aircraft"},
        title=title,
        opacity=0.55,
        trendline="lowess",
        trendline_scope="overall",
        trendline_color_override="black",
    )
    # Add an annotation zone
    p_bad = df_plot[col].quantile(0.90 if bad_dir == "up" else 0.10)
    if bad_dir == "up":
        fig.add_hrect(y0=p_bad, y1=df_plot[col].max() * 1.05,
                      fillcolor="red", opacity=0.06,
                      annotation_text="degradation zone (P90)", annotation_position="top left")
    else:
        fig.add_hrect(y0=df_plot[col].min() * 0.95, y1=p_bad,
                      fillcolor="red", opacity=0.06,
                      annotation_text="degradation zone (P10)", annotation_position="bottom left")

    direction_text = "Rising is bad" if bad_dir == "up" else "Falling is bad"
    pct_label = "P90" if bad_dir == "up" else "P10"
    hover_template = (
        "<b>MSN %{customdata[0]}</b><br>"
        "%{x|%d-%b-%Y}<br>"
        f"{title}: %{{y:.2f}} {unit}<br>"
        f"{direction_text}<br>"
        f"Degradation threshold ({pct_label}): {p_bad:.2f} {unit}"
        "<extra></extra>"
    )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker_size=5,
        hovertemplate=hover_template,
    )
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="Aircraft (⚠ = pre-failure history)",
    )
    return fig


# ── Threshold Analysis (EDA) helpers ──────────────────────────────────────────
def _signal_eda(df_full: pd.DataFrame, col: str, pred_col: str) -> dict | None:
    """Compare a signal's distribution between normal and predicted pre-failure
    flights. Returns the stats used in the histogram + summary table."""
    if df_full.empty or col not in df_full.columns or pred_col not in df_full.columns:
        return None
    sub = df_full.dropna(subset=[col, pred_col])
    normal = sub.loc[sub[pred_col].eq(0), col].astype(float)
    alert = sub.loc[sub[pred_col].eq(1), col].astype(float)
    if len(normal) < 30 or len(alert) < 10:
        return None
    return {
        "normal": normal,
        "alert": alert,
        "median_normal": float(normal.median()),
        "median_alert": float(alert.median()),
        "p75_normal": float(normal.quantile(0.75)),
        "p90_normal": float(normal.quantile(0.90)),
        "p10_normal": float(normal.quantile(0.10)),
        "p25_normal": float(normal.quantile(0.25)),
    }


def _eda_histogram(stats: dict, title: str, unit: str, bad_dir: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=stats["normal"], name="Normal flights",
        marker_color="#22c55e", opacity=0.55, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=stats["alert"], name="Predicted pre-failure",
        marker_color="#ef4444", opacity=0.55, histnorm="probability density",
    ))
    fig.add_vline(x=stats["median_normal"], line_dash="dash", line_color="#16a34a",
                  annotation_text="median (normal)", annotation_position="top left")
    fig.add_vline(x=stats["median_alert"], line_dash="dash", line_color="#dc2626",
                  annotation_text="median (pre-failure)", annotation_position="top right")
    ref = stats["p90_normal"] if bad_dir == "up" else stats["p10_normal"]
    ref_label = "P90 normal" if bad_dir == "up" else "P10 normal"
    fig.add_vline(x=ref, line_dash="dot", line_color="#f59e0b",
                  annotation_text=ref_label, annotation_position="bottom right")
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title=unit,
        yaxis_title="density",
        height=300,
        margin=dict(t=40, b=30, l=10, r=10),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── Tabs LH / RH / Risk / EDA ─────────────────────────────────────────────────
tab_lh, tab_rh, tab_status, tab_eda = st.tabs(
    ["Engine 1 — LH", "Engine 2 — RH", "Current Risk Status", "Threshold Analysis (EDA)"]
)

for tab, df, pred_col, side_label, col_idx, alert_set in [
    (tab_lh, df_lh, "pre_lh_sav_failure_prediction", "LH", 0, ALERT_HISTORY_LH),
    (tab_rh, df_rh, "pre_rh_sav_failure_prediction", "RH", 1, ALERT_HISTORY_RH),
]:
    with tab:
        if df.empty:
            st.info(f"No data available for {side_label}.")
            continue

        if alert_set:
            st.caption(
                "⚠ Aircraft with pre-failure history are individually colored; "
                "the rest of the fleet is shown in grey: "
                + ", ".join(_dnm(m) for m in sorted(alert_set))
            )

        rendered = 0
        for signal_name, (lh_col, rh_col, unit, caption, bad_dir) in SIGNALS.items():
            raw_col = lh_col if col_idx == 0 else rh_col
            col = _resolve_col(df, raw_col)
            if col is None:
                continue

            fig = _trend_chart(df, col, f"{signal_name} — {side_label}", unit, bad_dir, alert_set)
            if fig is None:
                continue

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"_{caption}_")
            rendered += 1

        if rendered == 0:
            st.warning(
                "The key degradation columns are not present in this dataset yet. "
                "Run the SAV feature-engineering pipeline to populate them."
            )
            # Show alert timeline as fallback
            if pred_col in df.columns and "ac_sn" in df.columns:
                df_alert = df.copy()
                df_alert["Status"] = df_alert[pred_col].map({0: "Normal", 1: "⚠️ Alert"})
                fig_fallback = px.scatter(
                    df_alert, x="date", y="ac_sn",
                    color="Status",
                    color_discrete_map={"Normal": "#86efac", "⚠️ Alert": "#ef4444"},
                    labels={"date": "", "ac_sn": "MSN"},
                    title=f"Alert timeline — {side_label}",
                )
                fig_fallback.update_traces(marker_size=6)
                fig_fallback.update_layout(
                    height=max(250, df_alert["ac_sn"].nunique() * 30),
                    xaxis=dict(tickformat="%d-%b-%y"),
                )
                st.plotly_chart(fig_fallback, use_container_width=True)

# ── Current Risk Status ────────────────────────────────────────────────────────
with tab_status:
    st.caption("Fleet-wide — every monitored aircraft; ignores the sidebar filter.")
    col_left, col_right = st.columns(2)

    for col_widget, df, pred_col, title in [
        (col_left,  df_lh_full, "pre_lh_sav_failure_prediction",  "LH — Latest flight risk per aircraft"),
        (col_right, df_rh_full, "pre_rh_sav_failure_prediction", "RH — Latest flight risk per aircraft"),
    ]:
        with col_widget:
            if df.empty or pred_col not in df.columns or "ac_sn" not in df.columns:
                st.info("No data.")
                continue

            latest = (
                df.sort_values("date")
                .groupby("ac_sn")
                .last()[[pred_col]]
                .reset_index()
                .sort_values(pred_col, ascending=False)
            )
            n_total = len(latest)
            n_alert = int(latest[pred_col].eq(1).sum())
            # Top 10: every alert aircraft first, fill remaining slots with normals
            latest = latest.head(max(10, n_alert))
            if n_total > len(latest):
                st.caption(
                    f"Top {len(latest)} of {n_total} aircraft (all {n_alert} in alert shown; "
                    f"{n_total - len(latest)} normal aircraft hidden)."
                )
            latest["Status"] = latest[pred_col].map({0: "Normal", 1: "Alert"})
            latest["color"] = latest[pred_col].map({0: "#22c55e", 1: "#ef4444"})
            latest["Display"] = latest["ac_sn"].map(_dnm)

            fig_risk = go.Figure(go.Bar(
                y=latest["Display"],
                x=latest[pred_col],
                orientation="h",
                marker_color=latest["color"],
                text=latest["Status"],
                textposition="inside",
            ))
            fig_risk.update_layout(
                title=title,
                xaxis=dict(tickvals=[0, 1], ticktext=["Normal", "Alert"], range=[0, 1.2]),
                yaxis_title="Aircraft",
                height=max(300, len(latest) * 28),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

# ── Threshold Analysis (EDA) ──────────────────────────────────────────────────
with tab_eda:
    st.subheader("What does a pre-failure flight look like?")
    st.markdown(
        "For each degradation signal, the full flight history is split by the "
        "model's prediction: **normal** (green) vs **predicted pre-failure** (red). "
        "When the red distribution's median sits beyond the normal P75/P90, the "
        "signal physically confirms what the model flags — the valve in pre-failure "
        "opens slower, stays open longer and delivers less N2. "
        "Use the **separation** column to see which signals discriminate best."
    )
    st.caption(
        "Computed over the full loaded history (ignores the sidebar filter) so the "
        "distributions are stable. Signals need ≥30 normal and ≥10 pre-failure "
        "flights to be analysed."
    )

    side_pick = st.radio("Engine side", ["LH", "RH"], horizontal=True, key="eda_side")
    df_eda = df_lh_full if side_pick == "LH" else df_rh_full
    pred_eda = "pre_lh_sav_failure_prediction" if side_pick == "LH" else "pre_rh_sav_failure_prediction"
    col_pick = 0 if side_pick == "LH" else 1

    if df_eda.empty or pred_eda not in df_eda.columns:
        st.info("Prediction column not available — cannot run the threshold analysis.")
    else:
        summary_rows = []
        n_plotted = 0
        plot_cols = st.columns(2)

        for signal_name, (lh_col, rh_col, unit, _caption, bad_dir) in SIGNALS.items():
            raw_col = lh_col if col_pick == 0 else rh_col
            col = _resolve_col(df_eda, raw_col)
            if col is None:
                continue
            stats = _signal_eda(df_eda, col, pred_eda)
            if stats is None:
                continue

            # Separation: share of pre-failure flights beyond the normal P75
            if bad_dir == "up":
                sep = float((stats["alert"] > stats["p75_normal"]).mean())
                confirms = stats["median_alert"] > stats["p75_normal"]
            else:
                sep = float((stats["alert"] < stats["p25_normal"]).mean())
                confirms = stats["median_alert"] < stats["p25_normal"]

            summary_rows.append({
                "Signal": signal_name,
                "Direction": "↑ rising is bad" if bad_dir == "up" else "↓ falling is bad",
                "Median — normal": round(stats["median_normal"], 2),
                "Median — pre-failure": round(stats["median_alert"], 2),
                "Normal P90" if bad_dir == "up" else "Normal P10":
                    round(stats["p90_normal"] if bad_dir == "up" else stats["p10_normal"], 2),
                "Separation (% pre-failure beyond normal P75/P25)": f"{sep:.0%}",
                "Confirms pre-failure?": "✅" if confirms else "⚠️ weak",
            })

            with plot_cols[n_plotted % 2]:
                st.plotly_chart(
                    _eda_histogram(stats, f"{signal_name} — {side_pick}", unit, bad_dir),
                    use_container_width=True,
                )
            n_plotted += 1

        if summary_rows:
            st.subheader("Signal separation summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.markdown(
                "**Reading this table** — a signal *confirms* pre-failure when the "
                "median of the predicted pre-failure population lies beyond the "
                "normal P75 (P25 for falling signals). Those signals can be used as "
                "physical sanity checks of the model: an aircraft flagged by the "
                "model **and** beyond the normal P90 on a confirming signal deserves "
                "priority inspection."
            )
        else:
            st.info(
                "Not enough flights in both classes to run the analysis "
                "(need ≥30 normal and ≥10 pre-failure flights per signal)."
            )
