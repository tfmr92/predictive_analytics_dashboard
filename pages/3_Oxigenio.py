"""
Oxygen — Crew Oxygen System monitoring.
Real AMM thresholds from MTM-0051-00-Vol16_E2.35.L3.PDF:
  - 1155 PSI: minimum dispatch (OBSERVER OXY LO PRESS — cyan CAS)
  - 845 PSI:  CREW OXY LO PRESS — amber CAS (do not dispatch)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, make_prefix_map, display_name, clean_df

st.set_page_config(page_title="Oxygen System", layout="wide")

PSI_AMBER  = 845
PSI_CYAN   = 1155
PSI_CHARGE = 1850
RECHARGE_THRESHOLD = 150  # PSI increase > this between readings = cylinder swap/recharge

st.title("💨 Crew Oxygen System — ATA 35")
st.markdown(
    "Monitors oxygen cylinder pressure across the fleet using real AMM thresholds. "
    "Two CAS alert levels apply: **OBSERVER OXY LO PRESS** (cyan, below 1,155 PSI) "
    "and **CREW OXY LO PRESS** (amber, below 845 PSI — no dispatch)."
)

# ── Data ──────────────────────────────────────────────────────────────────────
df = load("e2_oxy_report.parquet")

if df.empty:
    st.error("No data yet. Run the `save_oxy_report` job in Dagster.")
    st.stop()

AC_COL = next((c for c in ("aircraftSerNum-1", "ac_sn") if c in df.columns), None)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
if AC_COL:
    df[AC_COL] = df[AC_COL].astype(str)

# Filter future dates and invalid serials
prefix_map = make_prefix_map()
df = clean_df(df, date_col="date", ac_col=AC_COL, prefix_map=prefix_map)

# Add display column (prefix · MSN) — falls back to MSN if ac_master unavailable
if AC_COL:
    df["_display"] = df[AC_COL].map(lambda msn: display_name(msn, prefix_map))
DISP_COL = "_display" if "_display" in df.columns else AC_COL
AC_COL = DISP_COL  # all charts/groupbys use the display name from here on

# ── Pre-compute alerted aircraft (before sidebar so we can show count) ────────
all_ac = sorted(df[DISP_COL].dropna().unique().tolist()) if AC_COL else []
alerted_msns: list = []

if "alert" in df.columns and DISP_COL:
    alerted_msns = sorted(
        df[df["alert"] == True][DISP_COL].dropna().unique().tolist()
    )
elif "psi" in df.columns and DISP_COL:
    latest_all = df.sort_values("date").groupby(DISP_COL).last()
    alerted_msns = sorted(latest_all[latest_all["psi"] < PSI_CYAN].index.tolist())

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 14, 365, 90)
    planning_horizon = st.slider(
        "Planning horizon (days)", 7, 180, 30,
        help="Aircraft forecast to cross the 845 PSI amber threshold within this "
             "many days are flagged for maintenance planning.",
    )

    alert_label = (
        f"🚨 Alerts only  ({len(alerted_msns)} aircraft)"
        if alerted_msns else "🚨 Alerts only  (none detected)"
    )
    alert_filter = st.checkbox(
        alert_label, value=False,
        help="Pre-select only aircraft with elevated daily leak rate or PSI below the cyan CAS threshold.",
    )
    default_ac = alerted_msns if (alert_filter and alerted_msns) else all_ac
    selected_ac = st.multiselect("Aircraft (MSN)", options=all_ac, default=default_ac)

    if alerted_msns:
        st.divider()
        st.subheader("⚠️ Alerted Aircraft")
        for msn in alerted_msns:
            st.markdown(f"- **{msn}**")

    st.divider()
    st.subheader("AMM Thresholds (ATA 35)")
    st.metric("Amber — CREW OXY LO PRESS", f"{PSI_AMBER} PSI",
              help="Below this level: no dispatch. QRH action required.")
    st.metric("Cyan — OBSERVER OXY LO PRESS", f"{PSI_CYAN} PSI",
              help="Below this level: observer may not have full oxygen supply.")
    st.metric("Max cylinder charge", f"{PSI_CHARGE} PSI")

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

if selected_ac and DISP_COL:
    df = df[df[DISP_COL].isin(selected_ac)]
if "date" in df.columns:
    df = df[df["date"] >= cutoff].dropna(subset=["date"]).sort_values("date")

# ── Alert level helpers ────────────────────────────────────────────────────────
def _alert_level(psi):
    if psi < PSI_AMBER:
        return "🔴 CREW OXY LO PRESS (Amber)"
    elif psi < PSI_CYAN:
        return "🟡 OBSERVER OXY LO PRESS (Cyan)"
    return "🟢 Normal"

def _alert_color(psi):
    if psi < PSI_AMBER:
        return "#ef4444"
    elif psi < PSI_CYAN:
        return "#f59e0b"
    return "#22c55e"

if "psi" in df.columns:
    df["alert_level"] = df["psi"].apply(_alert_level)

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_amber = n_cyan = n_normal = n_total = 0
avg_psi = df["psi"].mean() if "psi" in df.columns else None

if AC_COL and "psi" in df.columns:
    latest = df.sort_values("date").groupby(AC_COL).last()
    n_total  = len(latest)
    n_amber  = int((latest["psi"] < PSI_AMBER).sum())
    n_cyan   = int(((latest["psi"] >= PSI_AMBER) & (latest["psi"] < PSI_CYAN)).sum())
    n_normal = int((latest["psi"] >= PSI_CYAN).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 CREW OXY LO PRESS", n_amber,
          help=f"Latest PSI < {PSI_AMBER} — amber CAS, no dispatch")
c2.metric("🟡 OBSERVER OXY LO PRESS", n_cyan,
          help=f"Latest PSI {PSI_AMBER}–{PSI_CYAN - 1} — cyan CAS")
c3.metric("🟢 Normal", n_normal,
          help=f"Latest PSI ≥ {PSI_CYAN}")
if avg_psi is not None:
    c4.metric("Fleet avg PSI (period)", f"{avg_psi:.0f}")

st.divider()

# ── Chart 1: PSI over time — alerted aircraft highlighted, recharge annotated ─
st.subheader("1. Oxygen Pressure (PSI) Over Time")
st.caption(
    "Aircraft with elevated leak rate are shown in **red** with thicker lines; normal aircraft in gray. "
    f"🔵 triangle markers indicate a recharge or cylinder swap event (PSI increase > {RECHARGE_THRESHOLD} PSI). "
    "Click any legend entry to show/hide that aircraft."
)

if "psi" in df.columns and AC_COL:
    df_psi = df.dropna(subset=["psi", AC_COL]).sort_values([AC_COL, "date"]).copy()

    # Recharge detection: positive PSI jump between consecutive readings per aircraft
    df_psi["_psi_diff"] = df_psi.groupby(AC_COL)["psi"].diff()
    recharges = df_psi[df_psi["_psi_diff"] > RECHARGE_THRESHOLD].copy()

    # Color map: alerted MSNs → red, others → slate gray
    alerted_in_view = set(
        df_psi[df_psi["alert"] == True][AC_COL].unique().tolist()
        if "alert" in df_psi.columns
        else df_psi[df_psi["psi"] < PSI_CYAN][AC_COL].unique().tolist()
    )
    color_map = {
        msn: ("#ef4444" if msn in alerted_in_view else "#94a3b8")
        for msn in df_psi[AC_COL].unique()
    }

    fig_psi = px.line(
        df_psi, x="date", y="psi",
        color=AC_COL,
        color_discrete_map=color_map,
        labels={"psi": "Pressure (PSI)", "date": "", AC_COL: "MSN"},
        title="Crew Oxygen Pressure — PSI Timeline",
        custom_data=[AC_COL, "alert_level"] if "alert_level" in df_psi.columns else [AC_COL],
    )
    fig_psi.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d-%b-%Y}<br>PSI: %{y:.0f}<extra></extra>"
    )

    # Thicker lines for alerted aircraft
    for msn in alerted_in_view:
        fig_psi.update_traces(selector=dict(name=str(msn)), line=dict(width=2.5))

    # Recharge markers
    if not recharges.empty:
        fig_psi.add_scatter(
            x=recharges["date"],
            y=recharges["psi"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="royalblue", opacity=0.85),
            name="Recharge / Swap",
            customdata=recharges[[AC_COL, "_psi_diff"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> — Recharge/Swap<br>"
                "%{x|%d-%b-%Y}<br>PSI after: %{y:.0f}<br>+%{customdata[1]:.0f} PSI<extra></extra>"
            ),
        )

    # AMM zones
    fig_psi.add_hrect(y0=0, y1=PSI_AMBER, fillcolor="rgba(239,68,68,0.07)",
                      line_width=0, annotation_text="No dispatch zone", annotation_position="top left")
    fig_psi.add_hrect(y0=PSI_AMBER, y1=PSI_CYAN, fillcolor="rgba(245,158,11,0.07)",
                      line_width=0, annotation_text="Reduced capability", annotation_position="top left")
    fig_psi.add_hline(y=PSI_AMBER, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"CREW OXY LO PRESS ({PSI_AMBER} PSI)",
                      annotation_position="bottom right")
    fig_psi.add_hline(y=PSI_CYAN, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"OBSERVER OXY LO PRESS ({PSI_CYAN} PSI)",
                      annotation_position="top right")
    fig_psi.update_layout(
        height=460,
        xaxis=dict(tickformat="%d-%b-%y"),
        legend_title_text="MSN",
        yaxis=dict(range=[0, PSI_CHARGE + 100]),
    )
    st.plotly_chart(fig_psi, use_container_width=True)

    if alerted_in_view:
        st.info(
            f"**Elevated leak rate:** {', '.join(sorted(alerted_in_view))} — "
            "review daily PSI drop trend and plan cylinder inspection."
        )

elif "delta_press" in df.columns and AC_COL:
    st.info(
        "Absolute PSI column (`psi`) not present — showing daily pressure drop. "
        "Re-run `save_oxy_report` to populate the PSI column."
    )

st.divider()

# ── Chart 2: Latest PSI per aircraft ──────────────────────────────────────────
st.subheader("2. Aircraft Status — Latest Reading vs. AMM Limits")
st.caption(
    "Most recent pressure reading per aircraft. "
    "Red bars require immediate maintenance before next departure. "
    "Hover for average daily leak rate."
)

if "psi" in df.columns and AC_COL:
    latest_status = (
        df.dropna(subset=[AC_COL, "psi"])
        .sort_values("date")
        .groupby(AC_COL)
        .last()[["psi"]]
        .reset_index()
        .sort_values("psi")
    )
    latest_status["alert_level"] = latest_status["psi"].apply(_alert_level)
    latest_status["color"] = latest_status["psi"].apply(_alert_color)

    if "delta_press" in df.columns:
        avg_leak = df.groupby(AC_COL)["delta_press"].mean().rename("avg_leak")
        latest_status = latest_status.join(avg_leak, on=AC_COL)
        latest_status["_hover"] = latest_status.apply(
            lambda r: (
                f"{r[AC_COL]}: {r['psi']:.0f} PSI  |  "
                f"Avg leak: {r.get('avg_leak', 0):.1f} PSI/day"
            ),
            axis=1,
        )
    else:
        latest_status["_hover"] = (
            latest_status[AC_COL].astype(str) + ": " + latest_status["psi"].round(0).astype(str) + " PSI"
        )

    fig_status = go.Figure(go.Bar(
        y=latest_status[AC_COL].astype(str),
        x=latest_status["psi"],
        orientation="h",
        marker_color=latest_status["color"],
        text=latest_status["alert_level"],
        textposition="outside",
        hovertext=latest_status["_hover"],
        hoverinfo="text",
    ))
    fig_status.add_vline(x=PSI_AMBER, line_dash="dash", line_color="#ef4444",
                         annotation_text=f"Amber ({PSI_AMBER} PSI)")
    fig_status.add_vline(x=PSI_CYAN, line_dash="dot", line_color="#f59e0b",
                         annotation_text=f"Cyan ({PSI_CYAN} PSI)")
    fig_status.update_layout(
        title="Latest oxygen pressure per aircraft",
        xaxis_title="Pressure (PSI)",
        yaxis_title="MSN",
        height=max(320, len(latest_status) * 34),
        xaxis=dict(range=[0, PSI_CHARGE + 100]),
        margin=dict(l=10, r=160, t=40, b=10),
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.divider()

# ── Chart 3: Daily leak rate per aircraft vs fleet mean ───────────────────────
st.subheader("3. Daily Leak Rate per Aircraft — vs. Fleet Mean")
st.caption(
    "Average PSI drop per day for each aircraft over the selected period. "
    "Aircraft above the fleet alert threshold (mean + 1σ) are highlighted in red — "
    "these show a steeper pressure-loss curve and should be prioritised for cylinder inspection."
)

if "delta_press" in df.columns and AC_COL:
    leak_per_ac = (
        df.dropna(subset=["delta_press", AC_COL])
        .groupby(AC_COL)["delta_press"]
        .agg(avg="mean", n_obs="count")
        .reset_index()
        .sort_values("avg", ascending=False)
    )
    fleet_mean = leak_per_ac["avg"].mean()
    fleet_std  = leak_per_ac["avg"].std()
    threshold  = fleet_mean + fleet_std
    leak_per_ac["_color"] = leak_per_ac["avg"].apply(
        lambda v: "#ef4444" if v > threshold else "#64748b"
    )

    fig_leak = go.Figure(go.Bar(
        y=leak_per_ac[AC_COL].astype(str),
        x=leak_per_ac["avg"],
        orientation="h",
        marker_color=leak_per_ac["_color"],
        text=leak_per_ac["avg"].round(1).astype(str) + " PSI/day",
        textposition="outside",
        customdata=leak_per_ac["n_obs"],
        hovertemplate="%{y}: %{x:.2f} PSI/day  (n=%{customdata} obs)<extra></extra>",
    ))
    fig_leak.add_vline(x=fleet_mean, line_dash="solid", line_color="#64748b",
                       annotation_text=f"Fleet mean ({fleet_mean:.1f})",
                       annotation_position="top right")
    fig_leak.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Alert threshold ({threshold:.1f})",
                       annotation_position="top right")
    fig_leak.update_layout(
        title="Average daily PSI leak rate per aircraft",
        xaxis_title="Avg PSI drop / day",
        yaxis_title="MSN",
        height=max(320, len(leak_per_ac) * 34),
        margin=dict(l=10, r=180, t=40, b=10),
    )
    st.plotly_chart(fig_leak, use_container_width=True)

st.divider()

# ── Chart 4: Recharge / Swap Frequency ────────────────────────────────────────
if "psi" in df.columns and AC_COL:
    df_sorted = df.dropna(subset=["psi", AC_COL]).sort_values([AC_COL, "date"]).copy()
    df_sorted["_psi_diff"] = df_sorted.groupby(AC_COL)["psi"].diff()
    recharge_counts = (
        df_sorted[df_sorted["_psi_diff"] > RECHARGE_THRESHOLD]
        .groupby(AC_COL)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    if not recharge_counts.empty:
        st.subheader("4. Cylinder Recharge / Swap Frequency")
        st.caption(
            f"Number of times PSI increased by more than {RECHARGE_THRESHOLD} PSI (recharge or cylinder swap) "
            "per aircraft in the selected period. "
            "High frequency combined with fast PSI decay is a strong indicator of cylinder leakage."
        )
        fleet_avg_rc = recharge_counts["n"].mean()
        recharge_counts["_color"] = recharge_counts["n"].apply(
            lambda n: "#ef4444" if n > fleet_avg_rc * 1.5 else ("#f59e0b" if n > fleet_avg_rc else "#64748b")
        )
        fig_rc = go.Figure(go.Bar(
            x=recharge_counts[AC_COL].astype(str),
            y=recharge_counts["n"],
            marker_color=recharge_counts["_color"],
            text=recharge_counts["n"],
            textposition="outside",
            hovertemplate="%{x}: %{y} recharge/swap events<extra></extra>",
        ))
        fig_rc.add_hline(
            y=fleet_avg_rc, line_dash="dot", line_color="#64748b",
            annotation_text=f"Fleet avg ({fleet_avg_rc:.1f})", annotation_position="right",
        )
        fig_rc.update_layout(
            xaxis_title="MSN",
            yaxis_title="Recharge / Swap Events",
            height=360,
        )
        st.plotly_chart(fig_rc, use_container_width=True)
        st.divider()

# ── Dispatch forecast ─────────────────────────────────────────────────────────
if "psi" in df.columns and "delta_press" in df.columns and AC_COL:
    st.subheader("5. Dispatch Forecast — Days Until Threshold")
    st.caption(
        "Estimated days until each aircraft crosses the 1,155 PSI (cyan) or 845 PSI (amber) "
        "threshold, based on the average daily pressure drop over the selected period."
    )

    today = pd.Timestamp.now().normalize()
    forecast_rows = []

    for msn, grp in df.dropna(subset=["psi", "delta_press", AC_COL]).groupby(AC_COL):
        current_psi = grp.sort_values("date")["psi"].iloc[-1]
        avg_drop = grp["delta_press"].mean()
        if avg_drop > 0:
            days_to_cyan  = max(0, (current_psi - PSI_CYAN)  / avg_drop)
            days_to_amber = max(0, (current_psi - PSI_AMBER) / avg_drop)
            est_amber_date = today + pd.Timedelta(days=days_to_amber)
            est_amber_str  = est_amber_date.strftime("%d-%b-%Y")
        else:
            days_to_cyan  = float("inf")
            days_to_amber = float("inf")
            est_amber_date = None
            est_amber_str  = "—"
        forecast_rows.append({
            "MSN": msn,
            "Current PSI": round(current_psi),
            "Avg Drop (PSI/day)": round(avg_drop, 2),
            "Days → Cyan (1155)": "—" if days_to_cyan == float("inf") else int(days_to_cyan),
            "Days → Amber (845)": "—" if days_to_amber == float("inf") else int(days_to_amber),
            "Estimated date (845)": est_amber_str,
            "Status": _alert_level(current_psi),
            "_days_to_amber": days_to_amber,
            "_est_amber_date": est_amber_date,
            "_below_amber": current_psi < PSI_AMBER,
        })

    if forecast_rows:
        df_fc = pd.DataFrame(forecast_rows).sort_values("Current PSI")

        immediate = df_fc[df_fc["_below_amber"]]
        upcoming  = df_fc[
            (~df_fc["_below_amber"])
            & (df_fc["_days_to_amber"] > 0)
            & (df_fc["_days_to_amber"] != float("inf"))
            & (df_fc["_days_to_amber"] <= planning_horizon)
        ].sort_values("_days_to_amber")

        if not immediate.empty:
            msn_list = ", ".join(immediate["MSN"].astype(str).tolist())
            st.error(
                f"**Immediate — no dispatch, QRH action required.** "
                f"Below {PSI_AMBER} PSI (amber): {msn_list}"
            )
        if not upcoming.empty:
            lines = [
                f"- **{r['MSN']}** → est. {r['_est_amber_date'].strftime('%d-%b-%Y')} "
                f"({int(r['_days_to_amber'])} days)"
                for _, r in upcoming.iterrows()
            ]
            st.warning(
                f"**Plan maintenance within {planning_horizon} days** — forecast to cross "
                f"the {PSI_AMBER} PSI amber threshold:\n" + "\n".join(lines)
            )
        if immediate.empty and upcoming.empty:
            st.success(
                f"No aircraft below {PSI_AMBER} PSI or forecast to cross it within "
                f"{planning_horizon} days. No dispatch action required."
            )

        display_cols = [
            "MSN", "Current PSI", "Avg Drop (PSI/day)",
            "Days → Cyan (1155)", "Days → Amber (845)",
            "Estimated date (845)", "Status",
        ]

        def _color_rows(row):
            if row["Current PSI"] < PSI_AMBER:
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            elif row["Current PSI"] < PSI_CYAN:
                return ["background-color: rgba(245,158,11,0.12)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_fc[display_cols].style.apply(_color_rows, axis=1),
            use_container_width=True,
            hide_index=True,
        )

# ── Top pressure-drop events ──────────────────────────────────────────────────
if "delta_press" in df.columns and "date" in df.columns:
    top_cols = ["date", AC_COL, "delta_press"] + (["psi"] if "psi" in df.columns else [])
    top_cols = [c for c in top_cols if c in df.columns]
    top = (
        df.nlargest(15, "delta_press")[top_cols]
        .rename(columns={
            "date": "Date",
            AC_COL: "MSN",
            "delta_press": "Pressure Drop (PSI)",
            "psi": "Current PSI",
        })
    )
    with st.expander("Top 15 highest single-day pressure drop events"):
        st.dataframe(top, use_container_width=True, hide_index=True)

st.divider()

# ── Section 6: Life Analysis — PSI Drop Histogram + Weibull ──────────────────
st.subheader("6. Life Analysis — Oxygen Charge Duration")
st.caption(
    "Left: distribution of PSI drops per flight (shows variance in daily consumption). "
    "Right: Weibull model fitted on the days between consecutive recharge events — "
    "B10/B50 guide proactive maintenance scheduling."
)

col_hist, col_weib = st.columns(2)

with col_hist:
    if "delta_press" in df.columns:
        _drops = df.dropna(subset=["delta_press"]).copy()
        _drops = _drops[_drops["delta_press"] > 0]
        if not _drops.empty:
            _fleet_avg_drop = _drops["delta_press"].mean()
            fig_hist = px.histogram(
                _drops, x="delta_press",
                nbins=30,
                labels={"delta_press": "PSI Drop per Flight", "count": "Flights"},
                title="PSI Drop per Flight — Distribution",
                color_discrete_sequence=["#64748b"],
            )
            fig_hist.add_vline(
                x=_fleet_avg_drop, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Fleet avg ({_fleet_avg_drop:.1f} PSI/flt)",
                annotation_position="top right",
            )
            fig_hist.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No PSI drop data in current selection.")
    else:
        st.info("`delta_press` column not available.")

with col_weib:
    if "psi" in df.columns and AC_COL:
        try:
            from scipy.stats import weibull_min as _weibull
            import numpy as _np

            _ds = (
                df.dropna(subset=["psi", AC_COL, "date"])
                .sort_values([AC_COL, "date"])
                .copy()
            )
            _ds["_pdiff"] = _ds.groupby(AC_COL)["psi"].diff()
            _intervals: list[float] = []
            for _, _grp in _ds.groupby(AC_COL):
                _rdates = _grp.loc[_grp["_pdiff"] > RECHARGE_THRESHOLD, "date"].tolist()
                for _i in range(1, len(_rdates)):
                    _d = (_rdates[_i] - _rdates[_i - 1]).days
                    if _d > 0:
                        _intervals.append(float(_d))

            if len(_intervals) >= 5:
                _arr = _np.array(_intervals)
                _shape, _loc, _scale = _weibull.fit(_arr, floc=0)
                _b10 = _weibull.ppf(0.10, _shape, _loc, _scale)
                _b50 = _weibull.ppf(0.50, _shape, _loc, _scale)
                _x   = _np.linspace(0, _arr.max() * 1.15, 300)
                _pdf = _weibull.pdf(_x, _shape, _loc, _scale)

                fig_w = go.Figure()
                fig_w.add_histogram(
                    x=_arr, name="Observed",
                    histnorm="probability density",
                    marker_color="steelblue", opacity=0.55,
                )
                fig_w.add_scatter(
                    x=_x, y=_pdf, mode="lines",
                    name=f"Weibull (β={_shape:.2f}, η={_scale:.0f}d)",
                    line=dict(color="#f59e0b", width=2.5),
                )
                fig_w.add_vline(x=_b10, line_dash="dot", line_color="#ef4444",
                                annotation_text=f"B10 = {_b10:.0f}d",
                                annotation_position="top right")
                fig_w.add_vline(x=_b50, line_dash="dot", line_color="#64748b",
                                annotation_text=f"B50 = {_b50:.0f}d",
                                annotation_position="top right")
                fig_w.update_layout(
                    title=f"Charge Service Life — Weibull  (n={len(_arr)} intervals)",
                    xaxis_title="Days Between Recharges",
                    yaxis_title="Probability Density",
                    height=340,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_w, use_container_width=True)
                st.caption(
                    f"**B10 = {_b10:.0f} days** — 10% of charges deplete by this point.  "
                    f"**B50 = {_b50:.0f} days** — median charge service life."
                )
            else:
                st.info(
                    f"Need ≥ 5 recharge intervals for Weibull fitting "
                    f"(found {len(_intervals)}). Expand the history window."
                )
        except ImportError:
            st.info("scipy not installed — add `scipy>=1.12.0` to requirements.txt.")
    else:
        st.info("PSI column not available for Weibull analysis.")
