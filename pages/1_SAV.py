"""
SAV — Starter Air Valve health monitoring
Displays the 5 key degradation signals per engine side (LH / RH).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

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

# ── Full unfiltered fleet (safety alerts must never inherit the sidebar filter) ──
def _full_fleet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    return df.dropna(subset=["date"]).sort_values("date")

df_lh_full = _full_fleet(df_lh)
df_rh_full = _full_fleet(df_rh)

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
    all_ac = sorted(
        set(df_lh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_lh.columns else [])
        | set(df_rh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_rh.columns else [])
    )
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=all_ac)

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
            + ", ".join(f"MSN {m}" for m in _lh_alert_msns)
            + " — inspect starter air valve per AMM"
        )
    if _rh_alert_msns:
        lines.append(
            "**RH pre-failure predicted:** "
            + ", ".join(f"MSN {m}" for m in _rh_alert_msns)
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


def _trend_chart(df: pd.DataFrame, col: str, title: str, unit: str, bad_dir: str) -> go.Figure:
    df_plot = df.dropna(subset=[col]).copy()
    if df_plot.empty:
        return None

    fig = px.scatter(
        df_plot,
        x="date", y=col,
        color="ac_sn",
        custom_data=["ac_sn"],
        labels={col: f"{title} ({unit})", "date": "", "ac_sn": "MSN"},
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
        height=300,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
    )
    return fig


# ── Tabs LH / RH ──────────────────────────────────────────────────────────────
tab_lh, tab_rh, tab_status = st.tabs(["Engine 1 — LH", "Engine 2 — RH", "Current Risk Status"])

for tab, df, pred_col, side_label, col_idx in [
    (tab_lh, df_lh, "pre_lh_sav_failure_prediction", "LH", 0),
    (tab_rh, df_rh, "pre_rh_sav_failure_prediction", "RH", 1),
]:
    with tab:
        if df.empty:
            st.info(f"No data available for {side_label}.")
            continue

        rendered = 0
        for signal_name, (lh_col, rh_col, unit, caption, bad_dir) in SIGNALS.items():
            raw_col = lh_col if col_idx == 0 else rh_col
            col = _resolve_col(df, raw_col)
            if col is None:
                continue

            fig = _trend_chart(df, col, f"{signal_name} — {side_label}", unit, bad_dir)
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
            latest["Status"] = latest[pred_col].map({0: "Normal", 1: "Alert"})
            latest["color"] = latest[pred_col].map({0: "#22c55e", 1: "#ef4444"})

            fig_risk = go.Figure(go.Bar(
                y=latest["ac_sn"].astype(str),
                x=latest[pred_col],
                orientation="h",
                marker_color=latest["color"],
                text=latest["Status"],
                textposition="inside",
            ))
            fig_risk.update_layout(
                title=title,
                xaxis=dict(tickvals=[0, 1], ticktext=["Normal", "Alert"], range=[0, 1.2]),
                yaxis_title="MSN",
                height=max(300, len(latest) * 28),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_risk, use_container_width=True)
