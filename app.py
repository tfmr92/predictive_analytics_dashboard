"""
Home — Azul Fleet Predictive Maintenance Dashboard
Summary KPIs + mini trend charts for all monitored systems (E2, ATR, Airbus).
"""

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, render_freshest_badge, render_empty_state

st.set_page_config(
    page_title="Azul Fleet — Predictive Maintenance",
    layout="wide",
)

st.title(":material/flight: Azul Fleet — Predictive Maintenance")
st.caption("E195-E2 · ATR 72 · A320 / A330 · Refreshed automatically · data lag ≤ 1 h")

render_freshest_badge(
    ["e2_sav_transient_lh_report.parquet", "e2_sav_transient_rh_report.parquet"],
    label="SAV report",
)
render_freshest_badge(["e2_wnb_report.parquet"], label="Wheels & Brakes report")
render_freshest_badge(["e2_oxy_report.parquet"], label="Oxygen report")

# ── Load data ──────────────────────────────────────────────────────────────────
df_sav_lh = load("e2_sav_transient_lh_report.parquet")
df_sav_rh = load("e2_sav_transient_rh_report.parquet")
df_wnb    = load("e2_wnb_report.parquet")
df_oxy    = load("e2_oxy_report.parquet")
df_fuel   = load("e2_fuel_report.parquet")

for df in (df_sav_lh, df_sav_rh, df_wnb, df_oxy, df_fuel):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

# E2 SAV is sourced from the ACARS-validated start-transient model (the same report
# 0_Fleet_Overview.py and 1_SAV.py trust), not the dead aggregate. The transient
# parquet carries no explicit alert column, so mirror 0_Fleet_Overview.py's
# _sav_transient_alert (prob >= the documented calibrated High band) and guard the
# call with a recency window so a stale flight's score cannot become a permanent
# false alert on the landing page.
_SAV_HIGH = 0.60          # mirrors 0_Fleet_Overview.py _SAV_HIGH — calibrated High band
_SAV_RECENCY_DAYS = 30    # last_flight_dt must be within this window to alert


# SAV (E2) — ACARS-validated start-transient model (the same report 1_SAV.py trusts).
# One row per aircraft (median of the last N starts) with a calibrated pre-failure
# probability and a last_flight_dt; no explicit alert column. We mirror 1_SAV.py
# _is_alert's fallback (prob >= _SAV_HIGH) and gate it on a recency window so a stale
# flight's score cannot become a permanent false alert. Returns {ac_sn: 1|0} — the
# SAME shape the old aggregate produced, so every downstream consumer is untouched.
def _sav_transient_alert(df: pd.DataFrame) -> dict:
    if (df is None or df.empty
            or "ac_sn" not in df.columns
            or "sav_transient_prob" not in df.columns):
        return {}
    d = df.copy()
    d["ac_sn"] = (
        d["ac_sn"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    d["sav_transient_prob"] = pd.to_numeric(d["sav_transient_prob"], errors="coerce")
    if "last_flight_dt" in d.columns:
        d["last_flight_dt"] = pd.to_datetime(d["last_flight_dt"], errors="coerce")
    else:
        d["last_flight_dt"] = pd.NaT
    # Latest row per aircraft (transient is already 1 row/ac; sort is a safety net).
    d = d.sort_values("last_flight_dt").groupby("ac_sn").last().reset_index()
    recency_floor = pd.Timestamp(date.today() - timedelta(days=_SAV_RECENCY_DAYS))
    alert: dict = {}
    for _, row in d.iterrows():
        p, lf = row["sav_transient_prob"], row["last_flight_dt"]
        is_alert = (
            pd.notna(p) and p >= _SAV_HIGH
            and pd.notna(lf) and lf >= recency_floor
        )
        alert[row["ac_sn"]] = 1 if is_alert else 0
    return alert


# ── Fleet KPIs ────────────────────────────────────────────────────────────────
sav_lh_alert = sum(_sav_transient_alert(df_sav_lh).values())
sav_rh_alert = sum(_sav_transient_alert(df_sav_rh).values())

wnb_hard = 0
if not df_wnb.empty:
    for col in ("NormAccel_lh", "NormAccel_rh"):
        if col in df_wnb.columns:
            wnb_hard += int((df_wnb[col] > 1.4).sum())

oxy_ac_col = next((c for c in ("aircraftSerNum-1", "ac_sn") if c in df_oxy.columns), None)
oxy_below_psi = 0
if oxy_ac_col and "psi" in df_oxy.columns:
    latest_oxy = df_oxy.dropna(subset=["psi"]).sort_values("date").groupby(oxy_ac_col).last()
    oxy_below_psi = int((latest_oxy["psi"] < 1800).sum())
elif not df_oxy.empty and "alert" in df_oxy.columns and oxy_ac_col:
    oxy_below_psi = int(df_oxy.sort_values("date").groupby(oxy_ac_col).last()["alert"].sum())

if df_sav_lh.empty and df_sav_rh.empty:
    render_empty_state(
        ["e2_sav_transient_lh_report.parquet", "e2_sav_transient_rh_report.parquet"],
        label="SAV report",
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("SAV alerts — LH", sav_lh_alert,
          help="Aircraft with predicted pre-failure on left starter valve (latest flight)")
c2.metric("SAV alerts — RH", sav_rh_alert,
          help="Aircraft with predicted pre-failure on right starter valve (latest flight)")
c3.metric("Hard landings (W&B)", wnb_hard,
          help="Total landings above 1.4 g in the loaded dataset")
c4.metric("Oxy below threshold", oxy_below_psi,
          help="Aircraft with latest pressure reading < 1800 PSI")

st.divider()

# ── Mini trend charts ──────────────────────────────────────────────────────────
st.subheader(":material/trending_up: Fleet Trends")
left, right = st.columns(2)

# SAV — top 10 aircraft by pre-failure probability (LH+RH combined). The transient
# report is one row per aircraft (median of last N starts), not a per-flight time
# series, so a weekly resample no longer applies — CLAUDE.md's "never list the whole
# fleet" rule means only the top 10 by probability are shown.
def _top_sav_bar(df_lh: pd.DataFrame, df_rh: pd.DataFrame):
    frames = []
    for df, side in ((df_lh, "LH"), (df_rh, "RH")):
        if df is None or df.empty or "sav_transient_prob" not in df.columns or "ac_sn" not in df.columns:
            continue
        d = df.copy()
        d["ac_sn"] = d["ac_sn"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        d["sav_transient_prob"] = pd.to_numeric(d["sav_transient_prob"], errors="coerce")
        d = d.dropna(subset=["sav_transient_prob"])
        if d.empty:
            continue
        d["side"] = side
        frames.append(d[["ac_sn", "sav_transient_prob", "side"]])
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None
    combined = combined.sort_values("sav_transient_prob", ascending=False).head(10)
    combined["label"] = combined["ac_sn"] + " (" + combined["side"] + ")"
    combined = combined.sort_values("sav_transient_prob", ascending=True)
    fig = go.Figure(go.Bar(
        y=combined["label"],
        x=combined["sav_transient_prob"],
        orientation="h",
        marker_color=combined["sav_transient_prob"].apply(
            lambda p: "#ef4444" if p >= _SAV_HIGH else "#22c55e"
        ),
        text=[f"{p:.0%}" for p in combined["sav_transient_prob"]],
        textposition="outside",
    ))
    fig.add_vline(x=_SAV_HIGH, line_dash="dash", line_color="#dc2626",
                  annotation_text="High", annotation_position="top")
    fig.update_layout(
        title="SAV — top pre-failure probability by aircraft",
        xaxis=dict(title="Pre-failure probability", range=[0, 1], tickformat=".0%"),
        yaxis_title="",
        height=max(260, len(combined) * 24 + 80),
        margin=dict(l=10, r=40, t=50, b=10),
    )
    return fig


with left:
    fig = _top_sav_bar(df_sav_lh, df_sav_rh)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SAV LH data not available.")

# Oxy — absolute PSI trend
with right:
    psi_col = "psi" if "psi" in df_oxy.columns else None
    if not df_oxy.empty and "date" in df_oxy.columns and psi_col and oxy_ac_col:
        fig2 = px.line(
            df_oxy.dropna(subset=["date", psi_col]).sort_values("date"),
            x="date", y=psi_col,
            color=oxy_ac_col,
            title="Crew Oxygen — pressure PSI per aircraft",
            labels={psi_col: "PSI", "date": "", oxy_ac_col: "MSN"},
        )
        fig2.add_hline(y=1800, line_dash="dash", line_color="red",
                       annotation_text="1800 PSI min", annotation_position="top right")
        fig2.update_layout(
            height=260,
            xaxis=dict(tickformat="%d-%b-%y"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif not df_oxy.empty and "date" in df_oxy.columns and "delta_press" in df_oxy.columns:
        fig2 = px.line(
            df_oxy.dropna(subset=["date"]).sort_values("date"),
            x="date", y="delta_press",
            color=oxy_ac_col if oxy_ac_col else None,
            title="Crew Oxygen — daily pressure drop",
            labels={"delta_press": "Drop (PSI)", "date": ""},
        )
        fig2.update_layout(height=260, xaxis=dict(tickformat="%d-%b-%y"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Oxygen data not available.")

st.info("Use the sidebar to navigate to detailed dashboards for each system.")
