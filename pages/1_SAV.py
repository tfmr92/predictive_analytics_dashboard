"""
SAV — Starter Air Valve health monitoring (transient-model edition)

Source of truth: the **start-transient** model (`sav_transient_model`, honest
GroupKFold AUC 0.74) scored per aircraft x engine by the `save_sav_transient_report`
job. It reads the raw start transient (valve-opening shape, oscillation, spool
rate, timing) plus cumulative ageing signals and emits a single **calibrated
probability** per aircraft x engine — a far stronger predictor than the aggregated
per-flight telemetry it replaces (honest AUC ~0.55-0.60).

The report parquet is a fleet **snapshot** (one row per aircraft x engine, median
of the last N starts), so this page ranks risk and explains it by driver rather
than plotting per-flight trends.

Operating point (binary inspect / monitor):
the model is sigmoid-calibrated, so the probability is meaningful (P=0.6 ≈ 60%).
If the producing job stamps a model-derived F2 threshold into the parquet
(`alert_threshold` / `sav_transient_alert`) the page uses it; otherwise it falls
back to documented calibrated bands — High ≥ 0.60, Watch ≥ 0.45 — and always shows
the raw probability so no decision is hostage to the cut.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import (
    load,
    clean_df,
    make_prefix_map,
    display_name,
    render_freshest_badge,
)

st.set_page_config(page_title="SAV — Starter Air Valve", layout="wide")

st.title(":material/settings: Starter Air Valve (SAV)")
st.markdown(
    "The starter valve opens to drive the turbine during engine start. As it "
    "degrades it opens **slower**, **oscillates**, the engine reaches **lower "
    "peak N2** and the start **labours**. The **start-transient model** reads the "
    "raw shape of each start (not just per-flight summaries) and returns a "
    "calibrated pre-failure probability per aircraft and engine — honest "
    "GroupKFold **AUC 0.74**, validated against ACARS-confirmed removals."
)

# ── Operating point ───────────────────────────────────────────────────────────
# Documented calibrated-probability bands, used as a fallback when the producing
# job has not (yet) stamped a model-derived F2 threshold into the parquet.
_THRESHOLD_HIGH = 0.60   # "High" — recommend SAV inspection per AMM
_THRESHOLD_WATCH = 0.45  # "Watch" — monitor; trend toward the high band
_TIER_COLOR = {"High": "#ef4444", "Watch": "#f59e0b", "Normal": "#22c55e"}

# ── Transient driver catalogue — (column, unit, bad_dir, caption) ──────────────
# Curated shape + ageing signals the model leans on, each with the physical
# direction in which degradation moves it. Used for the risk-driver heatmap and
# the "why this recommendation" explainability.
SIGNALS = {
    "Start oscillation (σ)": (
        "ss_osc_std", "AU", "up",
        "Steady-state pressure-oscillation amplitude during start — the strongest "
        "shape signal of a sticking/hunting valve.",
    ),
    "Peak N2 at start": (
        "n2_peak", "%N2", "down",
        "Peak N2 reached during the start — a degrading valve delivers less torque, "
        "so peak N2 falls.",
    ),
    "Pressure reversals": (
        "ss_n_reversals", "count", "up",
        "Reversals in the start pressure trace — unstable, hunting valve behaviour.",
    ),
    "Valve opening delay": (
        "t_cmd_to_air", "s", "up",
        "Delay from the start command to bleed-air pressure rise — a slow-opening valve.",
    ),
    "Time since installation": (
        "time_since_installation", "FC", "up",
        "Flight cycles since the valve was last installed — the dominant ageing signal.",
    ),
    "Cumulative N2 vibration": (
        "vib_cum_sum", "AU·FC", "up",
        "N2 vibration accumulated over the installation life.",
    ),
    "Start attempts": (
        "start_attempts", "count", "up",
        "Average start attempts in the window — repeated attempts flag a struggling start.",
    ),
}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_parquet(filename: str) -> pd.DataFrame:
    return load(filename)


df_lh = _load_parquet("e2_sav_transient_lh_report.parquet")
df_rh = _load_parquet("e2_sav_transient_rh_report.parquet")

if df_lh.empty and df_rh.empty:
    st.error(
        "No transient-model data yet. Run the `save_sav_transient_report` job in "
        "Dagster (it scores the fleet from `e2_sav_transient_live_features`)."
    )
    st.stop()

_prefix_map = make_prefix_map()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "last_flight_dt" in df.columns:
        df["last_flight_dt"] = pd.to_datetime(df["last_flight_dt"], errors="coerce")
    if "ac_sn" in df.columns:
        df["ac_sn"] = (
            df["ac_sn"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        )
    if "sav_transient_prob" in df.columns:
        df["sav_transient_prob"] = pd.to_numeric(df["sav_transient_prob"], errors="coerce")
    return clean_df(df, date_col="date", ac_col="ac_sn", prefix_map=_prefix_map)


df_lh = _normalize(df_lh)
df_rh = _normalize(df_rh)


def _dnm(msn) -> str:
    return display_name(str(msn), _prefix_map)


def _operating_threshold(*frames: pd.DataFrame) -> float:
    """Prefer a model-derived F2 threshold if the job stamped one into the
    parquet; otherwise use the documented calibrated High band."""
    for df in frames:
        if (
            not df.empty
            and "alert_threshold" in df.columns
            and df["alert_threshold"].notna().any()
        ):
            try:
                return float(df["alert_threshold"].dropna().iloc[0])
            except (TypeError, ValueError):
                pass
    return _THRESHOLD_HIGH


_HIGH = _operating_threshold(df_lh, df_rh)
_MODEL_THRESHOLD = _HIGH != _THRESHOLD_HIGH  # True when the job supplied the threshold


def _tier(p: float) -> str:
    if pd.isna(p):
        return "Normal"
    if p >= _HIGH:
        return "High"
    if p >= _THRESHOLD_WATCH:
        return "Watch"
    return "Normal"


def _is_alert(df: pd.DataFrame) -> pd.Series:
    """Per-row High-risk flag — prefers an explicit model alert column."""
    if "sav_transient_alert" in df.columns:
        return df["sav_transient_alert"].fillna(0).astype(int).eq(1)
    return df["sav_transient_prob"].ge(_HIGH)


for df in (df_lh, df_rh):
    if not df.empty and "sav_transient_prob" in df.columns:
        df["tier"] = df["sav_transient_prob"].map(_tier)


# ── Action fusion: probability tier + physical confirmation ───────────────────
# The probability ranks/severity; the binary action triggers maintenance. We make
# the binary a *confirmed* decision rather than a naive threshold by gating it on
# physical evidence — fewer false escalations, fully auditable.
_ACTION_COLOR = {
    "Inspect": "#ef4444", "Investigate": "#f59e0b",
    "Monitor": "#eab308", "—": "#22c55e",
}
_ACTION_ORDER = ["Inspect", "Investigate", "Monitor", "—"]


def _present_signals(df: pd.DataFrame) -> list:
    """(name, col, unit, bad_dir, caption) for signals present in df."""
    out = []
    for name, (col, unit, bad_dir, caption) in SIGNALS.items():
        if col in df.columns and df[col].notna().any():
            out.append((name, col, unit, bad_dir, caption))
    return out


def _action_assessment(df: pd.DataFrame) -> dict:
    """Fuse the probability tier with physical confirmation.

    A signal *confirms at fleet level* when the high-risk median sits beyond the
    rest-of-fleet P75 (P25 for falling signals). An aircraft is *confirmed* when
    it sits beyond that same rest P75/P25 on at least one fleet-confirming signal.

      High prob + confirmed                  → Inspect      (model + physical evidence)
      High prob + not confirmed              → Investigate  (model only; check sensor/context)
      High prob + no fleet-confirming signal → Inspect      (model only; confirmation n/a)
      Watch                                  → Monitor
      Normal                                 → —
    """
    out = {"conf_sigs": [], "rows": {}, "available": False}
    if df.empty or "sav_transient_prob" not in df.columns:
        return out
    alert = _is_alert(df)
    high, rest = df[alert], df[~alert]
    conf = []  # (name, col, bad_dir, ref)
    if len(high) >= 4 and len(rest) >= 6:
        for name, col, _unit, bad_dir, _cap in _present_signals(df):
            h = pd.to_numeric(high[col], errors="coerce").dropna()
            r = pd.to_numeric(rest[col], errors="coerce").dropna()
            if len(h) < 3 or len(r) < 4:
                continue
            if bad_dir == "up":
                ref = float(r.quantile(0.75))
                if float(h.median()) > ref:
                    conf.append((name, col, bad_dir, ref))
            else:
                ref = float(r.quantile(0.25))
                if float(h.median()) < ref:
                    conf.append((name, col, bad_dir, ref))
    out["conf_sigs"] = [c[0] for c in conf]
    out["available"] = len(conf) > 0

    rows = {}
    for _, row in df.iterrows():
        ac = row["ac_sn"]
        p = row["sav_transient_prob"]
        tier = _tier(p)
        drivers = []
        for name, col, bad_dir, ref in conf:
            v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.isna(v):
                continue
            if (bad_dir == "up" and v > ref) or (bad_dir == "down" and v < ref):
                drivers.append(name)
        confirmed = len(drivers) > 0
        if tier == "High":
            action = "Inspect" if (confirmed or not out["available"]) else "Investigate"
        elif tier == "Watch":
            action = "Monitor"
        else:
            action = "—"
        rows[ac] = {
            "prob": float(p) if pd.notna(p) else float("nan"),
            "tier": tier, "confirmed": confirmed,
            "drivers": drivers, "action": action,
        }
    out["rows"] = rows
    return out


def _acs_by_action(assess: dict, action: str) -> list:
    return sorted(ac for ac, r in assess["rows"].items() if r["action"] == action)


def _action_of(assess: dict, ac_sn: str) -> str:
    r = assess["rows"].get(ac_sn)
    return r["action"] if r else "—"


assess_lh = _action_assessment(df_lh)
assess_rh = _action_assessment(df_rh)

# ── Data freshness ────────────────────────────────────────────────────────────
render_freshest_badge(
    ["e2_sav_transient_lh_report.parquet", "e2_sav_transient_rh_report.parquet"],
    label="SAV transient report",
    stale_hours=48,
)

if _MODEL_THRESHOLD:
    st.caption(
        f"Operating point: model-derived F2 threshold = **{_HIGH:.2f}** "
        "(stamped by the scoring job)."
    )
else:
    st.caption(
        f"Operating point: calibrated bands — High ≥ **{_THRESHOLD_HIGH:.2f}**, "
        f"Watch ≥ **{_THRESHOLD_WATCH:.2f}** (raw probability shown throughout). "
        "Honest GroupKFold AUC 0.74."
    )

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    _alert_any = (
        set(df_lh.loc[_is_alert(df_lh), "ac_sn"]) if not df_lh.empty else set()
    ) | (set(df_rh.loc[_is_alert(df_rh), "ac_sn"]) if not df_rh.empty else set())
    only_alert = st.checkbox(
        f"Only high-risk aircraft ({len(_alert_any)})",
        value=False,
        help="Restrict the per-engine views to aircraft at or above the High "
             "operating point on either engine.",
    )
    all_ac = sorted(
        set(df_lh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_lh.columns else [])
        | set(df_rh["ac_sn"].dropna().unique().tolist() if "ac_sn" in df_rh.columns else [])
    )
    if only_alert:
        all_ac = [m for m in all_ac if m in _alert_any]
    selected_ac = st.multiselect(
        "Aircraft", options=all_ac, default=all_ac, format_func=_dnm,
    )


def _filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not selected_ac or "ac_sn" not in df.columns:
        return df
    return df[df["ac_sn"].isin(selected_ac)].reset_index(drop=True)


df_lh_f = _filter(df_lh)
df_rh_f = _filter(df_rh)

# ── KPIs (fleet-wide, ignores the sidebar filter) ─────────────────────────────
_n_monitored = len(
    set(df_lh["ac_sn"].dropna() if "ac_sn" in df_lh.columns else [])
    | set(df_rh["ac_sn"].dropna() if "ac_sn" in df_rh.columns else [])
)
_lh_insp, _rh_insp = _acs_by_action(assess_lh, "Inspect"), _acs_by_action(assess_rh, "Inspect")
_lh_invs, _rh_invs = _acs_by_action(assess_lh, "Investigate"), _acs_by_action(assess_rh, "Investigate")
_n_mon = len(_acs_by_action(assess_lh, "Monitor")) + len(_acs_by_action(assess_rh, "Monitor"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aircraft monitored", _n_monitored)
c2.metric("Inspect now", len(_lh_insp) + len(_rh_insp),
          help="High probability AND a fleet-discriminating driver degraded — "
               "model and physical evidence agree.")
c3.metric("Investigate", len(_lh_invs) + len(_rh_invs),
          help="High probability but no confirming driver — verify sensor/context "
               "before scheduling an inspection.")
c4.metric("Monitor", _n_mon,
          help="Watch band — trending toward the High operating point.")

# ── Fleet action triage banner (ignores the sidebar filter) ───────────────────
if _lh_insp or _rh_insp:
    lines = []
    if _lh_insp:
        lines.append("**LH:** " + ", ".join(_dnm(m) for m in _lh_insp))
    if _rh_insp:
        lines.append("**RH:** " + ", ".join(_dnm(m) for m in _rh_insp))
    st.error(
        "Inspect now — high pre-failure probability confirmed by a degraded driver "
        "(inspect starter air valve per AMM)\n\n" + "\n\n".join(lines)
    )

if _lh_invs or _rh_invs:
    lines = []
    if _lh_invs:
        lines.append("**LH:** " + ", ".join(_dnm(m) for m in _lh_invs))
    if _rh_invs:
        lines.append("**RH:** " + ", ".join(_dnm(m) for m in _rh_invs))
    st.warning(
        "Investigate — the model flags high risk but no fleet-discriminating driver "
        "is degraded; check sensor/context before scheduling\n\n" + "\n\n".join(lines)
    )

if not (_lh_insp or _rh_insp or _lh_invs or _rh_invs):
    st.success("No aircraft above the High operating point on either engine.")

_all_prob = pd.concat(
    [df_lh.get("sav_transient_prob", pd.Series(dtype=float)),
     df_rh.get("sav_transient_prob", pd.Series(dtype=float))]
)
st.caption(
    "Fleet-wide triage — fuses the transient probability with physical confirmation; "
    "ignores the sidebar filter."
    + (f" Fleet mean probability {_all_prob.mean():.0%}." if not _all_prob.empty else "")
)

st.divider()


# ── Shared helpers ────────────────────────────────────────────────────────────
def _risk_bar(df: pd.DataFrame, title: str, assess: dict) -> go.Figure | None:
    """Horizontal probability ranking, colored by the fused recommended action,
    with the operating-point line."""
    if df.empty or "sav_transient_prob" not in df.columns:
        return None
    d = df.dropna(subset=["sav_transient_prob"]).copy()
    if d.empty:
        return None
    d = d.sort_values("sav_transient_prob", ascending=True)
    d["Display"] = d["ac_sn"].map(_dnm)
    d["action"] = d["ac_sn"].map(lambda a: _action_of(assess, a))
    d["drivers_str"] = d["ac_sn"].map(
        lambda a: ", ".join(assess["rows"].get(a, {}).get("drivers", [])) or "—"
    )
    n_flights = d["n_flights"] if "n_flights" in d.columns else pd.Series([np.nan] * len(d))
    fig = go.Figure(go.Bar(
        y=d["Display"],
        x=d["sav_transient_prob"],
        orientation="h",
        marker_color=d["action"].map(_ACTION_COLOR),
        text=[f"{p:.0%}" for p in d["sav_transient_prob"]],
        textposition="outside",
        customdata=np.stack([d["action"], d["drivers_str"], n_flights], axis=-1),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.1%}<br>"
                      "Action: %{customdata[0]}<br>"
                      "Confirming drivers: %{customdata[1]}<br>"
                      "Starts in window: %{customdata[2]}"
                      "<extra></extra>",
    ))
    fig.add_vline(x=_HIGH, line_dash="dash", line_color="#dc2626",
                  annotation_text="High", annotation_position="top")
    if not _MODEL_THRESHOLD:
        fig.add_vline(x=_THRESHOLD_WATCH, line_dash="dot", line_color="#d97706",
                      annotation_text="Watch", annotation_position="top")
    fig.update_layout(
        title=title,
        xaxis=dict(title="Pre-failure probability", range=[0, 1], tickformat=".0%"),
        yaxis_title="",
        height=max(320, len(d) * 24 + 80),
        margin=dict(l=10, r=40, t=50, b=10),
    )
    return fig


def _action_table(df: pd.DataFrame, assess: dict) -> pd.DataFrame:
    """High + Watch aircraft with their fused action and confirming drivers,
    ordered by action severity then probability."""
    rows = []
    for ac in df["ac_sn"] if "ac_sn" in df.columns else []:
        r = assess["rows"].get(ac)
        if not r or r["action"] == "—":
            continue
        rows.append({
            "Aircraft": _dnm(ac),
            "Probability": f"{r['prob']:.0%}" if pd.notna(r["prob"]) else "—",
            "Action": r["action"],
            "Confirming drivers": ", ".join(r["drivers"]) if r["drivers"] else "—",
            "_o": _ACTION_ORDER.index(r["action"]),
            "_p": r["prob"] if pd.notna(r["prob"]) else -1.0,
        })
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = (tab.sort_values(["_o", "_p"], ascending=[True, False])
                  .drop(columns=["_o", "_p"]))
    return tab


def _driver_percentiles(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Per-aircraft directional degradation percentile vs the side's own fleet
    for each present signal (0 = fleet-best, 100 = most degraded)."""
    sigs = _present_signals(df)
    idx = df["ac_sn"].tolist()
    pct = pd.DataFrame(index=idx, columns=[s[0] for s in sigs], dtype=float)
    for name, col, _unit, bad_dir, _cap in sigs:
        vals = pd.to_numeric(df[col], errors="coerce")
        n = vals.notna().sum()
        if n < 3:
            continue
        ref = vals.dropna()
        for ac_sn, v in zip(df["ac_sn"], vals):
            if pd.isna(v):
                continue
            if bad_dir == "up":
                pct.at[ac_sn, name] = float((ref < v).sum()) / len(ref) * 100.0
            else:
                pct.at[ac_sn, name] = float((ref > v).sum()) / len(ref) * 100.0
    return pct, sigs


def _driver_heatmap(df: pd.DataFrame, side: str) -> go.Figure | None:
    if df.empty or "sav_transient_prob" not in df.columns:
        return None
    pct, sigs = _driver_percentiles(df)
    if pct.empty or not sigs:
        return None
    order = (
        df.dropna(subset=["sav_transient_prob"])
        .sort_values("sav_transient_prob", ascending=False)["ac_sn"]
        .tolist()
    )
    order = [a for a in order if a in pct.index][:14]
    sig_names = [s[0] for s in sigs]
    z, text, y = [], [], []
    for ac_sn in order:
        row_vals, row_text = [], []
        for sname in sig_names:
            v = pct.at[ac_sn, sname] if sname in pct.columns else np.nan
            row_vals.append(None if pd.isna(v) else round(float(v), 1))
            row_text.append("" if pd.isna(v) else f"{round(float(v))}")
        z.append(row_vals)
        text.append(row_text)
        prob = df.loc[df["ac_sn"] == ac_sn, "sav_transient_prob"].iloc[0]
        y.append(f"{_dnm(ac_sn)}  ({prob:.0%})")
    fig = go.Figure(go.Heatmap(
        z=z, x=sig_names, y=y, zmin=0, zmax=100,
        colorscale="RdYlGn_r",
        text=text, texttemplate="%{text}",
        colorbar=dict(title="Degradation %ile"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Degradation: %{z:.0f} percentile<extra></extra>",
    ))
    fig.update_layout(
        title=f"Risk drivers — {side} (top {len(order)} by probability; aircraft probability in brackets)",
        height=max(320, len(order) * 32 + 130),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(side="top", tickangle=-30),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Confirmed Failures (ACARS-validated ground truth) helper ──────────────────
_CONFIRMED_TABLE = "e2_sav_confirmed_failures"
_DB_URI = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:Airline2024**@localhost:5432/postgres",
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_confirmed_failures() -> pd.DataFrame:
    """Read the project's ground-truth SAV removal validation table.

    Returns an empty DataFrame on any failure (missing table, connection
    error, zero rows) so the tab can show a polite empty state and never
    fabricate data.

    sqlalchemy is imported lazily so the page never hard-depends on a DB driver:
    in the isolated cloud deploy the on-prem Postgres is unreachable and this
    simply degrades to the empty state."""
    try:
        from sqlalchemy import create_engine, inspect, text
        engine = create_engine(_DB_URI)
    except Exception:
        return pd.DataFrame()
    try:
        if not inspect(engine).has_table(_CONFIRMED_TABLE):
            return pd.DataFrame()
        return pd.read_sql(text(f"SELECT * FROM {_CONFIRMED_TABLE}"), engine)
    except Exception:
        return pd.DataFrame()
    finally:
        engine.dispose()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_lh, tab_rh, tab_eda, tab_confirmed = st.tabs([
    ":material/leaderboard: Fleet Risk Ranking",
    ":material/settings: Engine 1 — LH",
    ":material/settings: Engine 2 — RH",
    ":material/straighten: Why this recommendation (EDA)",
    ":material/verified: Confirmed Failures (ACARS-validated)",
])

# ── Fleet Risk Ranking ────────────────────────────────────────────────────────
with tab_rank:
    st.subheader(":material/trending_up: Pre-failure probability and recommended action")
    st.caption(
        "Each aircraft's latest transient score, ranked. Bar color = recommended "
        "action; the dashed line is the High operating point. Fleet-wide — ignores "
        "the sidebar filter."
    )
    col_l, col_r = st.columns(2)
    for col_widget, df_side, title, assess in [
        (col_l, df_lh, "Engine 1 — LH", assess_lh),
        (col_r, df_rh, "Engine 2 — RH", assess_rh),
    ]:
        with col_widget:
            fig = _risk_bar(df_side, title, assess)
            if fig is None:
                st.info(f"No probability data for {title}.")
            else:
                st.plotly_chart(fig, use_container_width=True)
            atab = _action_table(df_side, assess)
            if not atab.empty:
                st.dataframe(atab, use_container_width=True, hide_index=True)
    st.caption(
        "**Recommended action** fuses the probability tier with physical "
        "confirmation. Color: red **Inspect** (model + a fleet-discriminating "
        "driver degraded) · orange **Investigate** (high probability, no confirming "
        "driver — possible sensor/context, verify first) · yellow **Monitor** (Watch "
        "band) · green normal. Confirming drivers are listed per aircraft above."
    )

# ── Per-engine driver views ───────────────────────────────────────────────────
for tab, df_side, side in [(tab_lh, df_lh_f, "LH"), (tab_rh, df_rh_f, "RH")]:
    with tab:
        if df_side.empty or "sav_transient_prob" not in df_side.columns:
            st.info(f"No transient data available for {side}.")
            continue

        fig = _driver_heatmap(df_side, side)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each cell is the aircraft's directional percentile vs the fleet "
                "for that signal: 0 = fleet-best, 100 = most degraded (a 'falling "
                "is bad' signal like peak N2 is inverted). This is the physical "
                "*why* behind the probability on the left."
            )

        # Probability vs the two strongest physical drivers
        st.markdown("**Probability vs key drivers**")
        scatter_sigs = [s for s in (("ss_osc_std", "Start oscillation (σ)", "AU"),
                                    ("time_since_installation", "Time since installation", "FC"))
                        if s[0] in df_side.columns and df_side[s[0]].notna().any()]
        if scatter_sigs:
            scols = st.columns(len(scatter_sigs))
            for sc, (col, label, unit) in zip(scols, scatter_sigs):
                d = df_side.dropna(subset=[col, "sav_transient_prob"]).copy()
                if d.empty:
                    continue
                d["tier"] = d["sav_transient_prob"].map(_tier)
                d["Display"] = d["ac_sn"].map(_dnm)
                fig_sc = px.scatter(
                    d, x=col, y="sav_transient_prob", color="tier",
                    color_discrete_map=_TIER_COLOR,
                    hover_name="Display",
                    category_orders={"tier": ["High", "Watch", "Normal"]},
                    labels={col: f"{label} ({unit})",
                            "sav_transient_prob": "Pre-failure probability",
                            "tier": "Tier"},
                )
                fig_sc.add_hline(y=_HIGH, line_dash="dash", line_color="#dc2626")
                fig_sc.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
                fig_sc.update_layout(
                    height=330, margin=dict(t=30, b=10, l=10, r=10),
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                    legend=dict(orientation="h", y=1.12),
                )
                with sc:
                    st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            f"{side}: {df_side['ac_sn'].nunique()} aircraft scored "
            f"(median of the last {int(df_side['n_flights'].median()) if 'n_flights' in df_side.columns and df_side['n_flights'].notna().any() else 'N'} starts each)."
        )

# ── Why this recommendation (EDA) ─────────────────────────────────────────────
with tab_eda:
    st.subheader(":material/analytics: Why is the model flagging this aircraft?")
    st.markdown(
        "Two complementary views of the *why*: a **per-aircraft driver breakdown** "
        "(how degraded each signal is vs the fleet) and a **distribution split** "
        "between the high-risk and the rest of the fleet — the same pre-failure EDA "
        "methodology used across the platform, applied to the transient drivers."
    )

    side_pick = st.radio("Engine side", ["LH", "RH"], horizontal=True, key="eda_side")
    df_eda = df_lh if side_pick == "LH" else df_rh

    if df_eda.empty or "sav_transient_prob" not in df_eda.columns:
        st.info("No probability data available — cannot run the analysis.")
    else:
        # ── (a) Per-aircraft driver breakdown ─────────────────────────────────
        pct, sigs = _driver_percentiles(df_eda)
        ranked = (
            df_eda.dropna(subset=["sav_transient_prob"])
            .sort_values("sav_transient_prob", ascending=False)
        )
        options = ranked["ac_sn"].tolist()
        if options and sigs:
            pick = st.selectbox(
                "Aircraft", options=options, format_func=_dnm, key="eda_pick",
                help="Defaults to the highest-probability aircraft on this engine.",
            )
            prob = float(df_eda.loc[df_eda["ac_sn"] == pick, "sav_transient_prob"].iloc[0])
            mc1, mc2 = st.columns([1, 3])
            mc1.metric(f"{_dnm(pick)} — {side_pick}", f"{prob:.0%}", _tier(prob))

            rows = []
            for name, col, unit, bad_dir, caption in sigs:
                v = pct.at[pick, name] if name in pct.columns else np.nan
                raw = pd.to_numeric(df_eda.loc[df_eda["ac_sn"] == pick, col], errors="coerce").iloc[0]
                if pd.isna(v):
                    continue
                rows.append({"Signal": name, "pct": float(v), "raw": raw, "unit": unit,
                             "dir": bad_dir})
            if rows:
                rows.sort(key=lambda r: r["pct"], reverse=True)
                fig_d = go.Figure(go.Bar(
                    x=[r["pct"] for r in rows],
                    y=[r["Signal"] for r in rows],
                    orientation="h",
                    marker_color=["#ef4444" if r["pct"] >= 75 else
                                  "#f59e0b" if r["pct"] >= 50 else "#22c55e" for r in rows],
                    text=[f"{r['pct']:.0f}%ile · {r['raw']:.2f} {r['unit']}" for r in rows],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Fleet percentile: %{x:.0f}<extra></extra>",
                ))
                fig_d.update_layout(
                    title=f"Driver breakdown — {_dnm(pick)} ({side_pick})",
                    xaxis=dict(title="Degradation percentile vs fleet", range=[0, 105]),
                    height=max(280, len(rows) * 40 + 90),
                    margin=dict(l=10, r=80, t=50, b=10),
                )
                fig_d.update_yaxes(autorange="reversed")
                with mc2:
                    st.plotly_chart(fig_d, use_container_width=True)
                st.caption(
                    "Bars near 100 are the signals driving this aircraft's score: "
                    + "; ".join(
                        f"**{r['Signal']}** — {SIGNALS[r['Signal']][3]}"
                        for r in rows[:2]
                    )
                )

        st.divider()

        # ── (b) Distribution split: high-risk vs rest ─────────────────────────
        st.markdown("**High-risk vs rest-of-fleet — signal distributions**")
        alert_mask = _is_alert(df_eda)
        high = df_eda[alert_mask]
        rest = df_eda[~alert_mask]
        if len(high) < 4 or len(rest) < 6:
            st.info(
                f"Not enough aircraft in both groups for a stable split "
                f"(high={len(high)}, rest={len(rest)}). Need ≥4 high-risk and ≥6 rest."
            )
        else:
            sigs_all = _present_signals(df_eda)
            plot_cols = st.columns(2)
            summary_rows = []
            for i, (name, col, unit, bad_dir, _cap) in enumerate(sigs_all):
                h = pd.to_numeric(high[col], errors="coerce").dropna()
                r = pd.to_numeric(rest[col], errors="coerce").dropna()
                if len(h) < 3 or len(r) < 4:
                    continue
                if bad_dir == "up":
                    ref = float(r.quantile(0.75))
                    sep = float((h > ref).mean())
                    confirms = float(h.median()) > ref
                    ref_label = "Rest P75"
                else:
                    ref = float(r.quantile(0.25))
                    sep = float((h < ref).mean())
                    confirms = float(h.median()) < ref
                    ref_label = "Rest P25"
                summary_rows.append({
                    "Signal": name,
                    "Direction": "↑ rising is bad" if bad_dir == "up" else "↓ falling is bad",
                    "Median — high risk": round(float(h.median()), 2),
                    "Median — rest": round(float(r.median()), 2),
                    ref_label: round(ref, 2),
                    "Separation": f"{sep:.0%}",
                    "Confirms?": "" if confirms else "weak",
                })
                fig_b = go.Figure()
                fig_b.add_trace(go.Box(y=r, name="Rest", marker_color="#22c55e",
                                       boxpoints="all", jitter=0.4, pointpos=0))
                fig_b.add_trace(go.Box(y=h, name="High risk", marker_color="#ef4444",
                                       boxpoints="all", jitter=0.4, pointpos=0))
                fig_b.update_layout(
                    title=f"{name} ({unit})", height=300,
                    margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
                )
                with plot_cols[i % 2]:
                    st.plotly_chart(fig_b, use_container_width=True)

            if summary_rows:
                st.subheader(":material/analytics: Signal separation summary")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "A signal *confirms* the high-risk call when the high-risk median "
                    "sits beyond the rest-of-fleet P75 (P25 for falling signals). "
                    "Confirming signals are the physical evidence behind the model — "
                    "an aircraft flagged by the model **and** degraded on a confirming "
                    "signal deserves priority inspection."
                )

# ── Confirmed Failures (ACARS-validated ground truth) ─────────────────────────
with tab_confirmed:
    st.subheader(":material/verified: ACARS-confirmed SAV removals — ground-truth validation")

    df_cf = _load_confirmed_failures()
    if df_cf.empty:
        st.info(
            "No ACARS-validated removal records available yet. Run the SAV "
            "ground-truth labelling step that populates `e2_sav_confirmed_failures`."
        )
    else:
        def _pick(*cands):
            for c in cands:
                if c in df_cf.columns:
                    return c
            return None

        c_tail = _pick("tail", "AC")
        c_sn = _pick("ac_sn", "AC_SN")
        c_eng = _pick("engine", "side", "position")
        c_date = _pick("removal_date", "TRANSACTION_DATE", "removal_dt")
        c_conf = _pick("confirmed_by_acars", "confirmed")
        c_lead = _pick("days_fault_to_removal", "lead_days", "lead_time_days")
        c_cov = _pick("fault_data_available")
        c_code = _pick("fault_codes", "fault_code", "matched_fault_code")

        work = df_cf.copy()
        if c_date:
            work[c_date] = pd.to_datetime(work[c_date], errors="coerce")
        if c_conf:
            work[c_conf] = work[c_conf].fillna(False).astype(bool)
        if c_cov:
            work[c_cov] = work[c_cov].fillna(False).astype(bool)
        if c_lead:
            work[c_lead] = pd.to_numeric(work[c_lead], errors="coerce")

        # KPI row — ONLY over rows with fault coverage (gap removals excluded,
        # never counted as NFF).
        covered = work[work[c_cov]] if c_cov else work.iloc[0:0]
        n_cov = len(covered)
        k_conf = int(covered[c_conf].sum()) if (c_conf and n_cov) else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Removals with fault coverage", n_cov)
        k2.metric("Confirmed by ACARS", k_conf)
        if n_cov >= 5:
            k3.metric("Confirmation rate", f"{100 * k_conf / n_cov:.0f}%")
        else:
            k3.metric("Confirmation rate", "—",
                      help="Suppressed — fewer than 5 removals with fault coverage.")
        if c_lead and k_conf:
            med_lead = covered.loc[covered[c_conf], c_lead].median()
            k4.metric("Median lead time",
                      "—" if pd.isna(med_lead) else f"{med_lead:.0f} d")
        else:
            k4.metric("Median lead time", "—")

        def _eng_label(v):
            s = str(v).strip().upper()
            if s in ("E1", "LH", "LEFT", "1", "ENG 1", "ENG1"):
                return "LH"
            if s in ("E2", "RH", "RIGHT", "2", "ENG 2", "ENG2"):
                return "RH"
            return s or "—"

        def _disp(tail, sn):
            key = str(sn).strip()
            if key.endswith(".0"):
                key = key[:-2]
            if key.isdigit() and len(key) >= 5:
                key = key[-5:]  # page key convention (5-digit TCRF suffix)
            name = _dnm(key) if key else ""
            if key and " · " in name:
                return name
            tail = str(tail).strip()
            if tail:
                return f"{tail} · {key}" if key else tail
            return name or "—"

        def _status(row):
            if c_cov and not row[c_cov]:
                return "— No fault coverage"
            if c_conf and row[c_conf]:
                return "Confirmed"
            return "NFF candidate"

        rows = []
        for _, r in work.iterrows():
            rows.append({
                "MSN": _disp(r[c_tail] if c_tail else "", r[c_sn] if c_sn else ""),
                "Engine": _eng_label(r[c_eng]) if c_eng else "—",
                "Removal date": (r[c_date].strftime("%d-%b-%Y")
                                 if c_date and pd.notna(r[c_date]) else "—"),
                "Status": _status(r),
                "Lead time (days)": (int(r[c_lead])
                                     if c_lead and pd.notna(r[c_lead]) else None),
                "Matched fault code": (
                    str(r[c_code]).strip()
                    if c_code and pd.notna(r[c_code]) and str(r[c_code]).strip()
                    else "—"),
                "_sort": r[c_date] if c_date else pd.NaT,
            })
        table = pd.DataFrame(rows)
        if "_sort" in table.columns:
            table = (table.sort_values("_sort", ascending=False, na_position="last")
                          .drop(columns="_sort"))
        table = table.head(30)
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.caption(
            "Ground-truth validation: TRAX SAV removals × FHDB fault messages "
            "(FIM SAV fault codes 801103M* / 710000M* / FDE 801 501 51-52) — the "
            "**same label the transient model is evaluated against** (honest GroupKFold "
            "AUC 0.74). FHDB fault coverage is discontinuous (gap 2024→2025), so "
            "removals without fault coverage are excluded from the rate, never "
            "counted as NFF. Median lead time is the fault→removal horizon."
        )
