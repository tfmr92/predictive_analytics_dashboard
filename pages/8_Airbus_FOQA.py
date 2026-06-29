"""
Airbus FOQA/MOQA — A320/A330 exceedance monitoring from decoded QAR/DAR flights.
Speed envelope (VMO/MMO/VLE), EGT takeoff/continuous limits, engine vibration
advisories and oil-system flags per flight.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, render_freshest_badge

st.set_page_config(page_title="Airbus FOQA/MOQA", layout="wide")

st.title(":material/monitoring: FOQA / MOQA — Airbus A320 & A330")
st.markdown(
    "Per-flight exceedance monitoring from decoded QAR/DAR data: speed envelope "
    "(VMO/MMO/VLE), EGT takeoff & continuous limits, N1/N2 vibration advisories "
    "and oil-system flags."
)

render_freshest_badge(
    ["airbus_a320_foqa_report.parquet", "airbus_a330_foqa_report.parquet"],
    label="Airbus FOQA report",
)

AC_COL = "tail_number"

# ── Single exceedance-flag spec source ─────────────────────────────────────────
# Binds EVERY ATA-05 flag emitted by airbus_foqa_compute_op._compute_flight to its
# value column(s), per-fleet limit key (resolved against _FLEET_LIMITS — the one
# limit source) and comparison direction. Severity, labels and the legacy
# _LIMIT_FLAGS / _ADVISORY_FLAGS / _FLAG_LABELS structures all derive from here, so
# there is no second parallel spec table to drift. value_cols=[] / limit_key=None
# means "flagged without a number" (discrete flag or no stored value). Keep in sync
# with the producer when flags are added/removed.
_FLAG_SPEC = [
    {"key": "vmo_exceeded",            "label": "VMO exceeded",             "severity": "Limit",    "value_cols": ["cas_max_kias"],                            "limit_key": "vmo_kias",          "direction": "up",   "phase": "In-flight"},
    {"key": "mmo_exceeded",            "label": "MMO exceeded",             "severity": "Limit",    "value_cols": ["mach_max"],                                "limit_key": "mmo",               "direction": "up",   "phase": "In-flight"},
    {"key": "vle_exceeded",            "label": "VLE (gear ext.) exceeded", "severity": "Limit",    "value_cols": [],                                          "limit_key": "vle_kias",          "direction": "up",   "phase": "Approach/Landing"},
    {"key": "egt_takeoff_exceeded",    "label": "EGT takeoff limit",        "severity": "Limit",    "value_cols": ["egt1_max_takeoff_c", "egt2_max_takeoff_c"],"limit_key": "egt_takeoff_c",     "direction": "up",   "phase": "Takeoff"},
    {"key": "egt_continuous_exceeded", "label": "EGT continuous limit",     "severity": "Limit",    "value_cols": ["egt1_max_c", "egt2_max_c"],                "limit_key": "egt_continuous_c",  "direction": "up",   "phase": "Climb/Cruise"},
    {"key": "n1_vib_limit_exc",        "label": "N1 vibration limit",       "severity": "Limit",    "value_cols": ["n1_vib_max"],                              "limit_key": "n1_vib_limit",      "direction": "up",   "phase": "In-flight"},
    {"key": "n2_vib_limit_exc",        "label": "N2 vibration limit",       "severity": "Limit",    "value_cols": ["n2_vib_max"],                              "limit_key": "n2_vib_limit",      "direction": "up",   "phase": "In-flight"},
    {"key": "tire_overspeed",          "label": "Tire overspeed",           "severity": "Limit",    "value_cols": ["tire_speed_max_kt"],                       "limit_key": "max_tire_speed_kt", "direction": "up",   "phase": "Ground roll"},
    {"key": "n1_vib_advisory",         "label": "N1 vibration advisory",    "severity": "Advisory", "value_cols": ["n1_vib_max"],                              "limit_key": "n1_vib_advisory",   "direction": "up",   "phase": "In-flight"},
    {"key": "n2_vib_advisory",         "label": "N2 vibration advisory",    "severity": "Advisory", "value_cols": ["n2_vib_max"],                              "limit_key": "n2_vib_advisory",   "direction": "up",   "phase": "In-flight"},
    {"key": "n3_vib_advisory",         "label": "N3 vibration advisory",    "severity": "Advisory", "value_cols": ["n3_vib_max"],                              "limit_key": None,                "direction": "up",   "phase": "In-flight"},
    {"key": "oil_low_press_flag",      "label": "Oil pressure low",         "severity": "Advisory", "value_cols": ["oil_press_min_1_psi", "oil_press_min_2_psi"],"limit_key": "oil_press_low_psi", "direction": "down", "phase": "Engine running"},
    {"key": "oil_high_press_flag",     "label": "Oil pressure high",        "severity": "Advisory", "value_cols": [],                                          "limit_key": "oil_press_high_psi","direction": "up",   "phase": "Engine running"},
    {"key": "oil_temp_high_flag",      "label": "Oil temperature high",     "severity": "Advisory", "value_cols": [],                                          "limit_key": "oil_temp_high_c",   "direction": "up",   "phase": "Engine running"},
    {"key": "oil_qty_low_flag",        "label": "Oil quantity low",         "severity": "Advisory", "value_cols": [],                                          "limit_key": "oil_qty_low_qt",    "direction": "down", "phase": "Engine running"},
    {"key": "fmw_active",              "label": "FMW fault (ACMS)",         "severity": "Advisory", "value_cols": [],                                          "limit_key": None,                "direction": None,   "phase": "—"},
]

_FLAG_LABELS = {s["key"]: s["label"] for s in _FLAG_SPEC}
_LIMIT_FLAGS = [s["key"] for s in _FLAG_SPEC if s["severity"] == "Limit"]
# fmw_active is excluded from the legacy advisory list to preserve the existing
# per-fleet "Advisories" metric scope; it is still surfaced via _FLAG_SPEC below.
_ADVISORY_FLAGS = [s["key"] for s in _FLAG_SPEC
                   if s["severity"] == "Advisory" and s["key"] != "fmw_active"]

# Single limit source — exact mirror of _LIMITS in airbus_foqa_compute_op.py
# (certified FCOM/AMM values); keep in sync. Feeds BOTH the engine-trend overlays
# and the Fleet Exceedance Summary. EGT/TGT thresholds are the producer's
# placeholder values (PIPC CFM/RR confirmation pending) — the trend overlays still
# omit EGT (_TREND_LIMITS has no EGT entry), but the producer emits the EGT flags
# against these constants, so the summary surfaces them.
_FLEET_LIMITS = {
    "A320FAM": {
        "vmo_kias": 350.0,
        "mmo": 0.82,
        "vle_kias": 280.0,
        "max_tire_speed_kt": 195.0,
        "n1_vib_advisory": 6.0,
        "n1_vib_limit": 5.0,
        "n2_vib_advisory": 4.3,
        "n2_vib_limit": 5.0,
        "oil_press_advisory_psi": 16.0,
        "oil_press_red_psi": 13.0,
        "oil_press_high_psi": 90.0,
        "oil_temp_high_c": 140.0,
        "oil_qty_low_qt": 3.0,
        "egt_takeoff_c": 950.0,
        "egt_continuous_c": 925.0,
    },
    "A330": {
        "vmo_kias": 330.0,
        "mmo": 0.86,
        "vle_kias": 250.0,
        "max_tire_speed_kt": 204.0,
        "n1_vib_advisory": 5.7,
        "n2_vib_advisory": 5.6,
        "oil_press_low_psi": 30.0,
        "oil_temp_high_c": 140.0,
        "oil_qty_low_qt": 3.0,
        "egt_takeoff_c": 900.0,
        "egt_continuous_c": 875.0,
    },
}

_FLEET_KEY = {"A320/A321": "A320FAM", "A330": "A330"}

_DIR_ARROW = {"up": "↑", "down": "↓"}


def _fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


@st.cache_data(ttl=300)
def _fleet_exceedance_summary(df_a320: pd.DataFrame, df_a330: pd.DataFrame, days_back: int):
    """Consolidated A320+A330 exceedance view for the window.

    Returns (kpi, top10) where kpi holds per-fleet aircraft-with-exceedance counts
    plus total breaching flights, and top10 is the worst breaching parameter per
    aircraft ranked by severity (Limit > Advisory) then breaching-flight count then
    margin — so flags without a numeric limit never sink out of the ranking."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    kpi = {"A320/A321": 0, "A330": 0, "flights": 0}
    rows = []
    for fleet_label, fleet_key, df in (
        ("A320/A321", "A320FAM", df_a320),
        ("A330", "A330", df_a330),
    ):
        if df.empty or "date" not in df.columns:
            continue
        sub = df[df["date"] >= cutoff]
        flag_keys = [s["key"] for s in _FLAG_SPEC if s["key"] in sub.columns]
        if sub.empty or not flag_keys:
            continue
        breach_any = sub[flag_keys].fillna(False).astype(bool).any(axis=1)
        kpi["flights"] += int(breach_any.sum())
        kpi[fleet_label] = int(sub.loc[breach_any, AC_COL].nunique())
        limits = _FLEET_LIMITS.get(fleet_key, {})

        for ac, g in sub.groupby(AC_COL):
            tb = int(g[flag_keys].fillna(False).astype(bool).any(axis=1).sum())
            cand = []
            for s in _FLAG_SPEC:
                if s["key"] not in g.columns:
                    continue
                bm = g[s["key"]].fillna(False).astype(bool)
                n = int(bm.sum())
                if n == 0:
                    continue
                gb = g[bm]
                val = None
                vcols = [c for c in s["value_cols"] if c in gb.columns]
                if vcols:
                    vv = gb[vcols].apply(pd.to_numeric, errors="coerce")
                    series = vv.min(axis=1) if s["direction"] == "down" else vv.max(axis=1)
                    if series.notna().any():
                        val = float(series.min() if s["direction"] == "down" else series.max())
                limit = limits.get(s["limit_key"]) if s["limit_key"] else None
                margin = (val - limit) if (val is not None and limit is not None) else None
                if margin is not None and limit not in (None, 0):
                    norm = margin / abs(limit) if s["direction"] == "up" else -margin / abs(limit)
                else:
                    norm = None
                cand.append({
                    "sev": 2 if s["severity"] == "Limit" else 1,
                    "has_num": 1 if margin is not None else 0,
                    "norm": norm if norm is not None else 0.0,
                    "n": n,
                    "label": s["label"],
                    "value": val,
                    "limit": limit,
                    "margin": margin,
                    "dir": _DIR_ARROW.get(s["direction"], "—"),
                    "phase": s["phase"],
                    "latest": gb["date"].max(),
                })
            if not cand:
                continue
            worst = max(cand, key=lambda c: (c["sev"], c["has_num"], c["norm"], c["n"]))
            max_sev = max(c["sev"] for c in cand)
            best_norm = max((c["norm"] for c in cand if c["sev"] == max_sev), default=0.0)
            rows.append({
                "_sev": max_sev, "_tb": tb, "_norm": best_norm,
                "Fleet": fleet_label,
                "Aircraft": str(ac),
                "Worst parameter": worst["label"],
                "Value": _fmt_num(worst["value"]),
                "Limit": _fmt_num(worst["limit"]),
                "Margin": f"{worst['margin']:+.2f}" if worst["margin"] is not None else "—",
                "Dir": worst["dir"],
                "Breaching flights": worst["n"],
                "Driving phase": worst["phase"],
                "Latest breach": worst["latest"].strftime("%d-%b-%Y") if pd.notna(worst["latest"]) else "—",
            })

    if not rows:
        return kpi, pd.DataFrame()
    top = (
        pd.DataFrame(rows)
        .sort_values(["_sev", "_tb", "_norm"], ascending=False)
        .head(10)
        .drop(columns=["_sev", "_tb", "_norm"])
        .reset_index(drop=True)
    )
    return kpi, top

# dashboard column -> list of (limit_key, severity, direction, label)
_TREND_LIMITS = {
    "n1_vib_max": [
        ("n1_vib_advisory", "amber", "up", "N1 vib advisory"),
        ("n1_vib_limit", "red", "up", "N1 vib AMM limit"),
    ],
    "n2_vib_max": [
        ("n2_vib_advisory", "amber", "up", "N2 vib advisory"),
        ("n2_vib_limit", "red", "up", "N2 vib AMM limit"),
    ],
    "oil_press_min_1_psi": [
        ("oil_press_advisory_psi", "amber", "down", "Oil press advisory"),
        ("oil_press_red_psi", "red", "down", "Oil press red"),
        ("oil_press_low_psi", "red", "down", "Oil press low — ENG shutdown"),
    ],
    "oil_press_min_2_psi": [
        ("oil_press_advisory_psi", "amber", "down", "Oil press advisory"),
        ("oil_press_red_psi", "red", "down", "Oil press red"),
        ("oil_press_low_psi", "red", "down", "Oil press low — ENG shutdown"),
    ],
}

_SEV_STYLE = {
    "amber": dict(line_color="#d97706", line_dash="dot"),
    "red": dict(line_color="#dc2626", line_dash="dash"),
}


def _add_limit_overlays(fig, fleet_key, col, ymin, ymax):
    """Overlay certified FCOM/AMM limit lines + a shaded exceedance band on a trend chart."""
    if fleet_key not in _FLEET_LIMITS or col not in _TREND_LIMITS:
        return
    limits = _FLEET_LIMITS.get(fleet_key, {})
    specs = _TREND_LIMITS.get(col, [])
    applicable = [
        (key, sev, direction, label)
        for key, sev, direction, label in specs
        if limits.get(key) is not None
    ]
    if not applicable:
        return

    red_vals = []
    direction = applicable[0][2]
    for key, sev, _dir, label in applicable:
        value = limits[key]
        style = _SEV_STYLE[sev]
        fig.add_hline(
            y=value, line_color=style["line_color"], line_dash=style["line_dash"],
            line_width=1.5, annotation_text=f"{label} ({value:g})",
            annotation_position="top left" if direction == "up" else "bottom left",
        )
        if sev == "red":
            red_vals.append(value)

    if red_vals:
        ymin_v = ymin if ymin is not None and not np.isnan(ymin) else None
        ymax_v = ymax if ymax is not None and not np.isnan(ymax) else None
        if direction == "up":
            y0 = max(red_vals)
            top_ref = ymax_v if ymax_v is not None else y0
            y1 = max(top_ref, y0 * 1.02)
        else:
            y0 = min(red_vals)
            bot_ref = ymin_v if ymin_v is not None else y0
            y1 = min(bot_ref, y0 * 0.98)
        fig.add_hrect(
            y0=y0, y1=y1, fillcolor="rgba(220,38,38,0.10)",
            layer="below", line_width=0,
        )


@st.cache_data(ttl=300)
def _load(filename: str) -> pd.DataFrame:
    df = load(filename)
    if df.empty:
        return df
    if "flight_datetime" in df.columns:
        df["date"] = pd.to_datetime(
            df["flight_datetime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
        )
    if AC_COL in df.columns:
        df[AC_COL] = df[AC_COL].astype(str).str.strip()
        df = df[df[AC_COL] != ""]
    return df.dropna(subset=["date"]).sort_values("date")


df_a320 = _load("airbus_a320_foqa_report.parquet")
df_a330 = _load("airbus_a330_foqa_report.parquet")

if df_a320.empty and df_a330.empty:
    st.error("No data yet. Run the `airbus_foqa_moqa_job` in Dagster.")
    st.stop()

st.caption(
    f"Reports loaded — A320/A321: {len(df_a320):,} flights · A330: {len(df_a330):,} flights"
)
if df_a320.empty:
    st.info("A320/A321 FOQA report is missing or empty — A320 views are unavailable.")
if df_a330.empty:
    st.info("A330 FOQA report is missing or empty — A330 views are unavailable.")

_latest_date = max([d["date"].max() for d in (df_a320, df_a330) if not d.empty], default=None)
if _latest_date is not None and pd.notna(_latest_date):
    st.caption(f"Data through {_latest_date.strftime('%d-%b-%Y')} · auto-refreshed hourly")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    days_back = st.slider("Days of history", 7, 365, 60)

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

# ── Fleet Exceedance Summary (consolidated A320 + A330) ────────────────────────
st.subheader(":material/rule: Fleet Exceedance Summary")
st.caption(
    f"Hard-limit and advisory ATA-05 events across both Airbus fleets in the last "
    f"{days_back} days — worst breaching parameter per aircraft, ranked by severity."
)

_kpi, _top = _fleet_exceedance_summary(df_a320, df_a330, days_back)

k1, k2, k3 = st.columns(3)
k1.metric("A320/A321 aircraft with exceedances", _kpi["A320/A321"])
k2.metric("A330 aircraft with exceedances", _kpi["A330"])
k3.metric("Breaching flights (both fleets)", f"{_kpi['flights']:,}")

if _top.empty:
    st.success(f"No exceedances recorded across A320/A330 in the last {days_back} days.")
else:
    st.dataframe(_top, use_container_width=True, hide_index=True)
    st.caption(
        "Margin is signed value − limit; Dir marks ↑ above / ↓ below the limit. "
        "Flags without a published numeric limit appear as — and stay ranked by "
        "severity. EGT/TGT thresholds are provisional (OEM PIPC confirmation pending)."
    )


def _exceedance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-aircraft totals: limit exceedances, advisories, flights."""
    limit_cols = [c for c in _LIMIT_FLAGS if c in df.columns]
    adv_cols = [c for c in _ADVISORY_FLAGS if c in df.columns]
    out = df.groupby(AC_COL).agg(Flights=("date", "count")).reset_index()
    if limit_cols:
        out["Limit exceedances"] = (
            df[limit_cols].fillna(False).astype(bool).sum(axis=1)
            .groupby(df[AC_COL]).sum().reindex(out[AC_COL]).values
        )
    else:
        out["Limit exceedances"] = 0
    if adv_cols:
        out["Advisories"] = (
            df[adv_cols].fillna(False).astype(bool).sum(axis=1)
            .groupby(df[AC_COL]).sum().reindex(out[AC_COL]).values
        )
    else:
        out["Advisories"] = 0
    return out.sort_values(["Limit exceedances", "Advisories"], ascending=False)


for fleet_label, df_fleet in [("A320/A321", df_a320), ("A330", df_a330)]:
    st.divider()
    st.header(f":material/insights: {fleet_label}")

    if df_fleet.empty:
        st.info(f"No {fleet_label} FOQA data available.")
        continue

    sub = df_fleet[df_fleet["date"] >= cutoff]
    if sub.empty:
        st.info(
            f"No {fleet_label} flights in the last {days_back} days "
            f"(latest record: {df_fleet['date'].max():%d-%b-%Y}). "
            "Widen the history window in the sidebar."
        )
        continue

    summary = _exceedance_summary(sub)
    n_ac = len(summary)
    n_flights = len(sub)
    n_limit = int(summary["Limit exceedances"].sum())
    n_adv = int(summary["Advisories"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aircraft monitored", n_ac)
    c2.metric("Flights analysed", f"{n_flights:,}")
    c3.metric("Limit exceedances", n_limit,
              help="VMO/MMO/VLE, EGT takeoff/continuous, vibration limit, tire overspeed")
    c4.metric("Advisories", n_adv,
              help="Vibration advisories and oil pressure/temperature/quantity flags")

    # Triage banner
    crit = summary[summary["Limit exceedances"] > 0]
    if not crit.empty:
        st.error(
            "**Limit exceedances this period:**\n\n" + "\n".join(
                f"- **{r[AC_COL]}** — {int(r['Limit exceedances'])} limit event(s), "
                f"{int(r['Advisories'])} advisory(ies) in {int(r['Flights'])} flights"
                for _, r in crit.head(10).iterrows()
            )
        )
    else:
        st.success(f"No limit exceedances in the last {days_back} days.")

    col_bar, col_types = st.columns(2)

    with col_bar:
        top = summary.head(10)
        if len(summary) > 10:
            st.caption(f"Top 10 of {len(summary)} aircraft.")
        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            y=top[AC_COL], x=top["Limit exceedances"], name="Limit exceedances",
            orientation="h", marker_color="#ef4444",
        ))
        fig_top.add_trace(go.Bar(
            y=top[AC_COL], x=top["Advisories"], name="Advisories",
            orientation="h", marker_color="#f59e0b",
        ))
        fig_top.update_layout(
            barmode="stack", title=f"Exceedances per aircraft — last {days_back} days",
            height=max(300, len(top) * 32), yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_types:
        type_counts = []
        for c in _LIMIT_FLAGS + _ADVISORY_FLAGS:
            if c in sub.columns:
                n = int(sub[c].fillna(False).astype(bool).sum())
                if n > 0:
                    type_counts.append({
                        "Type": _FLAG_LABELS.get(c, c),
                        "Events": n,
                        "Severity": "Limit" if c in _LIMIT_FLAGS else "Advisory",
                    })
        if type_counts:
            df_types = pd.DataFrame(type_counts).sort_values("Events")
            fig_types = px.bar(
                df_types, x="Events", y="Type", color="Severity", orientation="h",
                color_discrete_map={"Limit": "#ef4444", "Advisory": "#f59e0b"},
                title="Events by exceedance type",
            )
            fig_types.update_layout(height=max(300, len(df_types) * 32),
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.success("No exceedance events of any type in this window.")

    # ── Engine trend charts ────────────────────────────────────────────────────
    with st.expander(f"Engine trends — {fleet_label}", expanded=False):
        st.caption(
            "Dashed/dotted lines mark certified FCOM/AMM advisory (amber) and "
            "maintenance/red limits; shaded band is the exceedance zone (above for "
            "vibration, below for oil pressure). EGT limits are omitted pending OEM "
            "(CFM/RR) confirmation."
        )
        trend_cols = [
            ("egt1_max_c", "EGT max — Engine 1 (°C)"),
            ("egt2_max_c", "EGT max — Engine 2 (°C)"),
            ("n1_vib_max", "N1 vibration max (AU)"),
            ("n2_vib_max", "N2 vibration max (AU)"),
            ("oil_press_min_1_psi", "Oil pressure min — Engine 1 (psi)"),
            ("oil_press_min_2_psi", "Oil pressure min — Engine 2 (psi)"),
        ]
        plot_cols = st.columns(2)
        n_plotted = 0
        for col, title in trend_cols:
            if col not in sub.columns:
                continue
            df_plot = sub.dropna(subset=[col])
            if df_plot.empty:
                continue
            fig_t = px.scatter(
                df_plot, x="date", y=col, color=AC_COL, opacity=0.6,
                labels={col: title, "date": "", AC_COL: "Aircraft"},
                title=title,
            )
            fig_t.update_traces(marker_size=5)
            fig_t.update_layout(height=300, margin=dict(t=40, b=20, l=10, r=10),
                                xaxis=dict(tickformat="%d-%b-%y"), showlegend=n_ac <= 12)
            vals = df_plot[col].to_numpy(dtype=float)
            ymin = float(np.nanmin(vals)) if vals.size and not np.all(np.isnan(vals)) else None
            ymax = float(np.nanmax(vals)) if vals.size and not np.all(np.isnan(vals)) else None
            _add_limit_overlays(fig_t, _FLEET_KEY.get(fleet_label), col, ymin, ymax)
            with plot_cols[n_plotted % 2]:
                st.plotly_chart(fig_t, use_container_width=True)
            n_plotted += 1
        if n_plotted == 0:
            st.info("No engine trend columns available in this dataset.")

    # ── Per-flight detail table for flagged flights ────────────────────────────
    flag_cols_present = [c for c in _LIMIT_FLAGS + _ADVISORY_FLAGS if c in sub.columns]
    flagged = sub[sub[flag_cols_present].fillna(False).astype(bool).any(axis=1)] if flag_cols_present else pd.DataFrame()
    if not flagged.empty:
        with st.expander(f"{len(flagged)} flagged flight(s) — detail"):
            detail_cols = [c for c in
                           ["date", AC_COL, "exceedance_types", "mach_max", "cas_max_kias",
                            "egt1_max_c", "egt2_max_c", "n1_vib_max", "n2_vib_max",
                            "tire_speed_max_kt"]
                           if c in flagged.columns]
            st.dataframe(
                flagged[detail_cols].sort_values("date", ascending=False)
                .rename(columns={AC_COL: "Aircraft", "date": "Date"}),
                use_container_width=True, hide_index=True,
            )
