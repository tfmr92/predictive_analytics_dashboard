"""
SAV A320 — Starter Air Valve health monitoring (A320/A321, LEAP-1A).
Per-engine degradation signals from QAR engine-start snapshots plus the
ML pre-failure prediction (see airbus_sav_model_training_job).

Threshold methodology mirrors the E2 SAV page: distributions are split by
the model's prediction and compared against the normal-population P75/P90
(see "Threshold Analysis" tab).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, render_freshest_badge

st.set_page_config(page_title="SAV A320 — Starter Air Valve", layout="wide")

st.title(":material/settings: Starter Air Valve — A320/A321")
st.markdown(
    "Engine-start health from QAR snapshots (LEAP-1A). A degrading starter air "
    "valve opens **slower**, holds **less supply pressure**, and the core "
    "accelerates **slower** — all visible before an inop start."
)

render_freshest_badge(
    ["airbus_sav_eng1_report.parquet", "airbus_sav_eng2_report.parquet"],
    label="A320 SAV report",
)

PRED_COL = "sav_failure_pred"
PROB_COL = "sav_failure_prob"
LABEL_COL = "pre_failure"
AC_COL = "aircraft_id"

# Days without a recorded engine start after which a pre-failure prediction is
# treated as stale: an in-service A320 starts engines several times daily, so a
# 30-day gap reliably means the aircraft is out of normal service.
RECENCY_DAYS = 30

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load(filename: str) -> pd.DataFrame:
    df = load(filename)
    if df.empty:
        return df
    if "flight_datetime" in df.columns:
        df["date"] = pd.to_datetime(df["flight_datetime"], errors="coerce")
    if AC_COL in df.columns:
        df[AC_COL] = df[AC_COL].astype(str).str.strip()
        df = df[df[AC_COL] != ""]
    return df.dropna(subset=["date"]).sort_values("date")

df_e1 = _load("airbus_sav_eng1_report.parquet")
df_e2 = _load("airbus_sav_eng2_report.parquet")

if df_e1.empty and df_e2.empty:
    st.error("No data yet. Run the `save_airbus_sav_report` job in Dagster.")
    st.stop()

# ── Alert history: aircraft that have EVER been in predicted pre-failure ──────
def _alert_history(df: pd.DataFrame) -> set:
    if df.empty or PRED_COL not in df.columns:
        return set()
    return set(df.loc[df[PRED_COL].eq(1), AC_COL].unique())

ALERT_HISTORY_E1 = _alert_history(df_e1)
ALERT_HISTORY_E2 = _alert_history(df_e2)
ALERT_HISTORY_ANY = ALERT_HISTORY_E1 | ALERT_HISTORY_E2

_latest_date = max(
    [d["date"].max() for d in (df_e1, df_e2) if not d.empty],
    default=None,
)
if _latest_date is not None and pd.notna(_latest_date):
    st.caption(f"Data through {_latest_date.strftime('%d-%b-%Y')} · auto-refreshed hourly")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    days_back = st.slider("Days of history", 30, 540, 180)
    only_alert = st.checkbox(
        f"Only aircraft with pre-failure history ({len(ALERT_HISTORY_ANY)})",
        value=False,
    )
    all_ac = sorted(
        set(df_e1[AC_COL].unique() if not df_e1.empty else [])
        | set(df_e2[AC_COL].unique() if not df_e2.empty else [])
    )
    if only_alert:
        all_ac = [t for t in all_ac if t in ALERT_HISTORY_ANY]
    selected_ac = st.multiselect("Aircraft", options=all_ac, default=all_ac)

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

def _filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if selected_ac:
        df = df[df[AC_COL].isin(selected_ac)]
    return df[df["date"] >= cutoff]

df_e1_full, df_e2_full = df_e1, df_e2
df_e1, df_e2 = _filter(df_e1), _filter(df_e2)

# ── KPIs ──────────────────────────────────────────────────────────────────────
def _alert_tails(df: pd.DataFrame) -> list[str]:
    if df.empty or PRED_COL not in df.columns:
        return []
    latest = df.sort_values("date").groupby(AC_COL).last()
    return sorted(latest.index[latest[PRED_COL].eq(1)].tolist())

def _alert_tails_aged(df: pd.DataFrame) -> dict[str, int]:
    """{tail: age_days} for tails whose latest start predicts pre-failure."""
    if df.empty or PRED_COL not in df.columns:
        return {}
    latest = df.sort_values("date").groupby(AC_COL).last()
    flagged = latest[latest[PRED_COL].eq(1)]
    today = pd.Timestamp.now().normalize()
    return {
        str(tail): int((today - row["date"].normalize()).days)
        for tail, row in flagged.iterrows()
    }

_e1_alerts = _alert_tails(df_e1_full)
_e2_alerts = _alert_tails(df_e2_full)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aircraft monitored (Eng 1)", df_e1[AC_COL].nunique() if not df_e1.empty else 0)
c2.metric("In alert — Eng 1", len(_e1_alerts))
c3.metric("Aircraft monitored (Eng 2)", df_e2[AC_COL].nunique() if not df_e2.empty else 0)
c4.metric("In alert — Eng 2", len(_e2_alerts))

_aged_e1 = _alert_tails_aged(df_e1_full)
_aged_e2 = _alert_tails_aged(df_e2_full)
_active_e1 = {t: a for t, a in _aged_e1.items() if a <= RECENCY_DAYS}
_stale_e1 = {t: a for t, a in _aged_e1.items() if a > RECENCY_DAYS}
_active_e2 = {t: a for t, a in _aged_e2.items() if a <= RECENCY_DAYS}
_stale_e2 = {t: a for t, a in _aged_e2.items() if a > RECENCY_DAYS}

def _fmt_active(tails: dict[str, int]) -> str:
    return "\n- ".join(
        f"{t} (last start {a}d ago) — inspect starter air valve (ATA 80)"
        for t, a in sorted(tails.items())
    )

def _fmt_stale(tails: dict[str, int]) -> str:
    return ", ".join(f"{t} (last start {a}d ago)" for t, a in sorted(tails.items()))

_any_active = bool(_active_e1 or _active_e2)
_any_stale = bool(_stale_e1 or _stale_e2)

if _any_active:
    lines = []
    if _active_e1:
        lines.append("**Engine 1 pre-failure predicted:**\n- " + _fmt_active(_active_e1))
    if _active_e2:
        lines.append("**Engine 2 pre-failure predicted:**\n- " + _fmt_active(_active_e2))
    st.error("Fleet safety triage\n\n" + "\n\n".join(lines))

if _any_stale:
    warn = []
    if _stale_e1:
        warn.append("**Engine 1:** " + _fmt_stale(_stale_e1))
    if _stale_e2:
        warn.append("**Engine 2:** " + _fmt_stale(_stale_e2))
    st.warning(
        "Flagged but no start in 30+ days — likely grounded/in maintenance, "
        "verify aircraft status:\n\n" + "\n\n".join(warn)
    )

if not _any_active and not _any_stale:
    st.success("No SAV pre-failure predicted on the latest start across the A320 fleet.")

st.caption(
    "Fleet-wide alert — based on every aircraft's latest start; ignores the sidebar "
    "filter. Red = predicted pre-failure on a recent start; amber = stale prediction "
    "(no start in 30+ days) pending aircraft-status verification."
)

st.divider()

# ── Signal definitions ────────────────────────────────────────────────────────
SIGNALS = {
    "Valve Open Time": (
        "valve_open_time_s", "seconds",
        "Rising trend = valve slow to cycle — degraded actuator or low supply muscle pressure.",
        "up",
    ),
    "Total Start Time": (
        "total_start_time_s", "seconds",
        "Time from starter engagement to start complete — laboured starts take longer.",
        "up",
    ),
    "Starter Air Pressure (mean, active)": (
        "sap_mean_active", "psi",
        "Mean duct pressure while the starter is engaged — a leaking/slow valve delivers less.",
        "down",
    ),
    "N2 Acceleration 12→30%": (
        "n2_accel_12_30_pct_per_s", "%/s",
        "Core acceleration during cranking — falling trend means less starter torque delivered.",
        "down",
    ),
    "EGT at Light-off": (
        "egt_lightoff_c", "°C",
        "Hotter light-offs accompany weak cranking (lower airflow at fuel-on).",
        "up",
    ),
}

_FLEET_GREY = "#cbd5e1"
_FLEET_LABEL = "Fleet — no alert history"


def _trend_chart(df: pd.DataFrame, col: str, title: str, unit: str, bad_dir: str,
                 alert_set: set) -> go.Figure | None:
    df_plot = df.dropna(subset=[col]).copy()
    if len(df_plot) < 5:
        return None
    df_plot["_legend"] = df_plot[AC_COL].map(
        lambda t: f"{t}" if t in alert_set else _FLEET_LABEL
    )
    alert_labels = sorted(l for l in df_plot["_legend"].unique() if l != _FLEET_LABEL)
    palette = px.colors.qualitative.Set1
    color_map = {_FLEET_LABEL: _FLEET_GREY}
    color_map.update({l: palette[i % len(palette)] for i, l in enumerate(alert_labels)})

    fig = px.scatter(
        df_plot, x="date", y=col,
        color="_legend",
        color_discrete_map=color_map,
        category_orders={"_legend": [_FLEET_LABEL] + alert_labels},
        custom_data=[AC_COL],
        labels={col: f"{title} ({unit})", "date": "", "_legend": "Aircraft"},
        title=title, opacity=0.55,
        trendline="lowess", trendline_scope="overall", trendline_color_override="black",
    )
    p_bad = df_plot[col].quantile(0.90 if bad_dir == "up" else 0.10)
    if bad_dir == "up":
        fig.add_hrect(y0=p_bad, y1=df_plot[col].max() * 1.05, fillcolor="red", opacity=0.06,
                      annotation_text="degradation zone (P90)", annotation_position="top left")
    else:
        fig.add_hrect(y0=df_plot[col].min() * 0.95, y1=p_bad, fillcolor="red", opacity=0.06,
                      annotation_text="degradation zone (P10)", annotation_position="bottom left")
    fig.update_traces(
        selector=dict(mode="markers"), marker_size=5,
        hovertemplate=("<b>%{customdata[0]}</b><br>%{x|%d-%b-%Y}<br>"
                       f"{title}: %{{y:.2f}} {unit}<extra></extra>"),
    )
    fig.update_layout(
        height=320, margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="Aircraft (= pre-failure history)",
    )
    return fig


def _signal_eda(df_full: pd.DataFrame, col: str, pred_col: str) -> dict | None:
    if df_full.empty or col not in df_full.columns or pred_col not in df_full.columns:
        return None
    sub = df_full.dropna(subset=[col, pred_col])
    normal = sub.loc[sub[pred_col].eq(0), col].astype(float)
    alert = sub.loc[sub[pred_col].eq(1), col].astype(float)
    if len(normal) < 30 or len(alert) < 10:
        return None
    return {
        "normal": normal, "alert": alert,
        "median_normal": float(normal.median()),
        "median_alert": float(alert.median()),
        "p75_normal": float(normal.quantile(0.75)),
        "p90_normal": float(normal.quantile(0.90)),
        "p25_normal": float(normal.quantile(0.25)),
        "p10_normal": float(normal.quantile(0.10)),
    }


def _eda_histogram(stats: dict, title: str, unit: str, bad_dir: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=stats["normal"], name="Normal starts",
                               marker_color="#22c55e", opacity=0.55,
                               histnorm="probability density"))
    fig.add_trace(go.Histogram(x=stats["alert"], name="Predicted pre-failure",
                               marker_color="#ef4444", opacity=0.55,
                               histnorm="probability density"))
    fig.add_vline(x=stats["median_normal"], line_dash="dash", line_color="#16a34a",
                  annotation_text="median (normal)", annotation_position="top left")
    fig.add_vline(x=stats["median_alert"], line_dash="dash", line_color="#dc2626",
                  annotation_text="median (pre-failure)", annotation_position="top right")
    ref = stats["p90_normal"] if bad_dir == "up" else stats["p10_normal"]
    fig.add_vline(x=ref, line_dash="dot", line_color="#f59e0b",
                  annotation_text="P90 normal" if bad_dir == "up" else "P10 normal",
                  annotation_position="bottom right")
    fig.update_layout(barmode="overlay", title=title, xaxis_title=unit,
                      yaxis_title="density", height=300,
                      margin=dict(t=40, b=30, l=10, r=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_e1, tab_e2, tab_status, tab_eda, tab_rank = st.tabs(
    [":material/settings: Engine 1", ":material/settings: Engine 2", ":material/health_and_safety: Current Risk Status", ":material/straighten: Threshold Analysis (EDA)",
     ":material/leaderboard: Fleet Degradation Ranking"]
)

for tab, df, side_label, alert_set in [
    (tab_e1, df_e1, "Engine 1", ALERT_HISTORY_E1),
    (tab_e2, df_e2, "Engine 2", ALERT_HISTORY_E2),
]:
    with tab:
        if df.empty:
            st.info(f"No data available for {side_label}.")
            continue
        if alert_set:
            st.caption(
                "Aircraft with pre-failure history are individually colored; "
                "the rest of the fleet is shown in grey: " + ", ".join(sorted(alert_set))
            )
        rendered = 0
        for signal_name, (col, unit, caption, bad_dir) in SIGNALS.items():
            if col not in df.columns:
                continue
            fig = _trend_chart(df, col, f"{signal_name} — {side_label}", unit, bad_dir, alert_set)
            if fig is None:
                continue
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"_{caption}_")
            rendered += 1
        if rendered == 0:
            st.warning("The degradation columns are not present in this dataset yet.")

with tab_status:
    st.caption("Fleet-wide — every monitored aircraft; ignores the sidebar filter. "
               "Bar length = latest model probability of pre-failure.")
    col_left, col_right = st.columns(2)
    for col_widget, df, title in [
        (col_left, df_e1_full, "Engine 1 — latest start risk per aircraft"),
        (col_right, df_e2_full, "Engine 2 — latest start risk per aircraft"),
    ]:
        with col_widget:
            if df.empty or PROB_COL not in df.columns:
                st.info("No data.")
                continue
            latest = (
                df.sort_values("date").groupby(AC_COL).last()[[PROB_COL, PRED_COL]]
                .reset_index().sort_values(PROB_COL, ascending=False)
            )
            n_total = len(latest)
            n_alert = int(latest[PRED_COL].eq(1).sum())
            latest = latest.head(max(10, n_alert))
            if n_total > len(latest):
                st.caption(f"Top {len(latest)} of {n_total} aircraft by latest risk "
                           f"(all {n_alert} in alert shown).")
            latest["color"] = latest[PRED_COL].map({0: "#22c55e", 1: "#ef4444"})
            fig_risk = go.Figure(go.Bar(
                y=latest[AC_COL], x=latest[PROB_COL],
                orientation="h", marker_color=latest["color"],
                text=latest[PROB_COL].map(lambda p: f"{p:.0%}"),
                textposition="outside",
            ))
            fig_risk.update_layout(
                title=title, xaxis=dict(range=[0, 1.15], tickformat=".0%"),
                yaxis_title="Aircraft", height=max(300, len(latest) * 28),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

with tab_eda:
    st.subheader(":material/insights: What does a pre-failure start look like?")
    st.markdown(
        "Each start in the history is split by the model's prediction: **normal** "
        "(green) vs **predicted pre-failure** (red). A signal physically confirms "
        "the pre-failure call when the red median sits beyond the normal P75/P90."
    )
    side_pick = st.radio("Engine", ["Engine 1", "Engine 2"], horizontal=True, key="ab_eda")
    df_eda = df_e1_full if side_pick == "Engine 1" else df_e2_full

    if df_eda.empty or PRED_COL not in df_eda.columns:
        st.info("Prediction column not available — cannot run the threshold analysis.")
    else:
        summary_rows = []
        n_plotted = 0
        plot_cols = st.columns(2)
        for signal_name, (col, unit, _caption, bad_dir) in SIGNALS.items():
            stats = _signal_eda(df_eda, col, PRED_COL)
            if stats is None:
                continue
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
                "Separation (% pre-failure beyond normal P75/P25)": f"{sep:.0%}",
                "Confirms pre-failure?": "" if confirms else "weak",
            })
            with plot_cols[n_plotted % 2]:
                st.plotly_chart(_eda_histogram(stats, f"{signal_name} — {side_pick}", unit, bad_dir),
                                use_container_width=True)
            n_plotted += 1

        if summary_rows:
            st.subheader(":material/analytics: Signal separation summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough starts in both classes to run the analysis "
                    "(need ≥30 normal and ≥10 pre-failure starts per signal).")

with tab_rank:
    st.subheader(":material/leaderboard: Fleet degradation ranking")
    st.markdown(
        "Each aircraft's recent starts are scored against the fleet's own "
        "**normal** start distribution, per confirming signal, and combined into a "
        "composite degradation score. This early-warning ranking **complements** — "
        "it does not replace — the binary Current Risk Status."
    )
    side_rank = st.radio("Engine", ["Engine 1", "Engine 2"], horizontal=True, key="ab_rank")
    df_rank = df_e1_full if side_rank == "Engine 1" else df_e2_full
    alert_hist_rank = ALERT_HISTORY_E1 if side_rank == "Engine 1" else ALERT_HISTORY_E2

    MIN_RANK_SIGNALS = 2

    if df_rank.empty or PRED_COL not in df_rank.columns or AC_COL not in df_rank.columns:
        st.info("Prediction column not available — cannot build the degradation ranking.")
    else:
        # (1) keep only signals whose predicted-pre-failure median confirms degradation
        confirming = {}  # signal_name -> (col, bad_dir, stats)
        for signal_name, (col, _unit, _caption, bad_dir) in SIGNALS.items():
            stats = _signal_eda(df_rank, col, PRED_COL)
            if stats is None:
                continue
            if bad_dir == "up" and stats["median_alert"] > stats["p75_normal"]:
                confirming[signal_name] = (col, bad_dir, stats)
            elif bad_dir == "down" and stats["median_alert"] < stats["p25_normal"]:
                confirming[signal_name] = (col, bad_dir, stats)

        if not confirming:
            st.info(
                "No signal currently confirms pre-failure (no signal's predicted "
                "median sits beyond the normal P75/P25), so a fleet degradation "
                "ranking would not be meaningful yet."
            )
        else:
            # (2)+(3) per-aircraft directional degradation percentile per signal
            signal_cols = list(confirming.keys())
            rows = []
            for tail, g in df_rank.sort_values("date").groupby(AC_COL):
                cell = {"Aircraft": str(tail)}
                for signal_name, (col, bad_dir, stats) in confirming.items():
                    recent = g[col].dropna().tail(5)
                    if len(recent) < 3:
                        cell[signal_name] = float("nan")
                        continue
                    val = float(recent.median())
                    normal = stats["normal"]
                    if bad_dir == "up":
                        cell[signal_name] = float((normal < val).mean()) * 100
                    else:
                        cell[signal_name] = float((normal > val).mean()) * 100
                rows.append(cell)

            rank_df = pd.DataFrame(rows).set_index("Aircraft")

            # (4) THE FIX: require >= MIN_RANK_SIGNALS non-null cells before the
            # composite, so a single high signal can never top the ranking.
            coverage = rank_df[signal_cols].notna().sum(axis=1)
            eligible = rank_df[coverage >= MIN_RANK_SIGNALS].copy()
            n_dropped = int((coverage < MIN_RANK_SIGNALS).sum())

            if eligible.empty:
                st.info(
                    f"No aircraft has at least {MIN_RANK_SIGNALS} confirming signals "
                    "with enough recent starts to compute a comparable composite score."
                )
            else:
                eligible["_composite"] = eligible[signal_cols].mean(axis=1, skipna=True)
                eligible = eligible.sort_values("_composite", ascending=False)

                if n_dropped:
                    st.caption(
                        f"{n_dropped} aircraft excluded for thin coverage (fewer than "
                        f"{MIN_RANK_SIGNALS} confirming signals with enough recent starts)."
                    )

                # (5) heatmap — rows = aircraft, cols = confirming signals, top 12
                heat = eligible.head(12)
                z = heat[signal_cols].values.tolist()
                text = [
                    ["" if pd.isna(v) else f"{round(v)}" for v in row]
                    for row in z
                ]
                fig_heat = go.Figure(go.Heatmap(
                    z=z,
                    x=signal_cols,
                    y=list(heat.index),
                    zmin=0, zmax=100,
                    colorscale="RdYlGn_r",
                    text=text,
                    texttemplate="%{text}",
                    colorbar=dict(title="Degradation %"),
                ))
                fig_heat.update_layout(
                    title=f"Degradation percentile vs normal starts — {side_rank} (top 12)",
                    height=max(320, len(heat) * 34 + 120),
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # (6) ranked table tying the continuous score back to the binary model
                table_rows = []
                for tail, row in eligible.iterrows():
                    vals = row[signal_cols]
                    table_rows.append({
                        "Aircraft": tail,
                        "Composite degradation score": round(float(row["_composite"]), 1),
                        "Signals with data": int(vals.notna().sum()),
                        "Signals in red zone (≥75%)": int((vals >= 75).sum()),
                        "Model alert": "yes" if tail in alert_hist_rank else "—",
                    })
                st.dataframe(
                    pd.DataFrame(table_rows), use_container_width=True, hide_index=True
                )

                st.caption(
                    "Continuous early-warning degradation comparison against the fleet's "
                    "own normal start distribution. It **complements** (does not replace) "
                    "the binary Current Risk Status, and is computed fleet-wide, ignoring "
                    f"the sidebar filter. Aircraft with fewer than {MIN_RANK_SIGNALS} "
                    "confirming signals with data are excluded to keep the composite "
                    "comparable. This is a relative heuristic, not a model validated "
                    "against confirmed SAV removals."
                )
