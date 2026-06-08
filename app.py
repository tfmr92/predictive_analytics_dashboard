"""
Home — Azul Fleet Predictive Maintenance Dashboard
Summary KPIs + mini trend charts for all monitored systems (E2, ATR, Airbus).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.drive_loader import load

st.set_page_config(
    page_title="Azul Fleet — Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Azul Fleet — Predictive Maintenance")
st.caption("E195-E2 · ATR 72 · A320 / A330 · Refreshed automatically · data lag ≤ 1 h")

# ── Load data ──────────────────────────────────────────────────────────────────
df_sav_lh = load("e2_sav_lh_report.parquet")
df_sav_rh = load("e2_sav_rh_report.parquet")
df_wnb    = load("e2_wnb_report.parquet")
df_oxy    = load("e2_oxy_report.parquet")
df_fuel   = load("e2_fuel_report.parquet")

for df in (df_sav_lh, df_sav_rh, df_wnb, df_oxy, df_fuel):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")


def _alert_aircraft(df: pd.DataFrame, pred_col: str, ac_col: str = "ac_sn") -> int:
    """Latest flight for each aircraft — how many are in alert."""
    if df.empty or pred_col not in df.columns or ac_col not in df.columns:
        return 0
    return int(df.sort_values("date").groupby(ac_col).last()[pred_col].eq(1).sum())


# ── Fleet KPIs ────────────────────────────────────────────────────────────────
sav_lh_alert = _alert_aircraft(df_sav_lh, "pre_lh_sav_failure_prediction")
sav_rh_alert = _alert_aircraft(df_sav_rh, "pre_rh_sav_failure_prediction")

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 SAV alerts — LH", sav_lh_alert,
          help="Aircraft with predicted pre-failure on left starter valve (latest flight)")
c2.metric("🔴 SAV alerts — RH", sav_rh_alert,
          help="Aircraft with predicted pre-failure on right starter valve (latest flight)")
c3.metric("⚠️ Hard landings (W&B)", wnb_hard,
          help="Total landings above 1.4 g in the loaded dataset")
c4.metric("💨 Oxy below threshold", oxy_below_psi,
          help="Aircraft with latest pressure reading < 1800 PSI")

st.divider()

# ── Mini trend charts ──────────────────────────────────────────────────────────
st.subheader("Fleet Trends")
left, right = st.columns(2)

# SAV LH — weekly alert rate
with left:
    if not df_sav_lh.empty and "date" in df_sav_lh.columns and "pre_lh_sav_failure_prediction" in df_sav_lh.columns:
        weekly = (
            df_sav_lh.dropna(subset=["date"])
            .set_index("date")
            .resample("W")["pre_lh_sav_failure_prediction"]
            .mean()
            .reset_index()
        )
        weekly.columns = ["Week", "Alert Rate"]
        fig = px.area(
            weekly, x="Week", y="Alert Rate",
            title="SAV LH — weekly % flights in alert",
            color_discrete_sequence=["#ef4444"],
        )
        fig.update_layout(
            yaxis_tickformat=".0%",
            height=260,
            xaxis=dict(tickformat="%d-%b-%y"),
        )
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
