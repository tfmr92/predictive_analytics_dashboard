import pandas as pd
import plotly.express as px
import streamlit as st

from utils.drive_loader import load

st.set_page_config(
    page_title="Azul E2 — Manutenção Preditiva",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Azul E2 — Manutenção Preditiva")
st.caption("Atualizado automaticamente a cada execução do Dagster · dados com até 1h de defasagem")

# ── Carregar dados ────────────────────────────────────────────────────────────
df_sav_lh  = load("e2_sav_lh_report.parquet")
df_sav_rh  = load("e2_sav_rh_report.parquet")
df_wnb     = load("e2_wnb_report.parquet")
df_oxy     = load("e2_oxy_report.parquet")
df_fuel    = load("e2_fuel_report.parquet")


def _risk_count(df: pd.DataFrame, pred_col: str) -> int:
    if df.empty or pred_col not in df.columns:
        return 0
    recent = df.sort_values("date").groupby("ac_sn").tail(30)
    return int((recent[pred_col] == 1).sum())


def _alert_aircraft(df: pd.DataFrame) -> int:
    if df.empty or "alert" not in df.columns:
        return 0
    latest = df.sort_values("date").groupby("aircraftSerNum-1").last()
    return int(latest["alert"].sum())


# ── KPIs resumo ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

sav_lh_risk = _risk_count(df_sav_lh, "pre_lh_sav_failure_prediction")
sav_rh_risk = _risk_count(df_sav_rh, "pre_rh_sav_failure_prediction")

wnb_hard = 0
if not df_wnb.empty:
    for col in ("NormAccel_lh", "NormAccel_rh"):
        if col in df_wnb.columns:
            wnb_hard += int((df_wnb[col] > 1.4).sum())

oxy_alerts = _alert_aircraft(df_oxy)

col1.metric("🔴 Alertas SAV (LH)", sav_lh_risk, help="Voos com predição de pré-falha no Starter esquerdo")
col2.metric("🔴 Alertas SAV (RH)", sav_rh_risk, help="Voos com predição de pré-falha no Starter direito")
col3.metric("⚠️ Pousos duros (W&B)", wnb_hard, help="Pousos com aceleração acima de 1,4 g")
col4.metric("💨 Alertas Oxigênio", oxy_alerts, help="Aeronaves com alerta de queda de pressão")

st.divider()

# ── Mini-gráficos de tendência ─────────────────────────────────────────────────
st.subheader("Tendência geral")

left, right = st.columns(2)

# SAV: % de voos com alerta por semana
with left:
    if not df_sav_lh.empty and "date" in df_sav_lh.columns:
        df_sav_lh["date"] = pd.to_datetime(df_sav_lh["date"], errors="coerce")
        weekly = (
            df_sav_lh.dropna(subset=["date"])
            .set_index("date")
            .resample("W")["pre_lh_sav_failure_prediction"]
            .mean()
            .reset_index()
        )
        weekly.columns = ["Semana", "% Alertas LH"]
        fig = px.area(
            weekly, x="Semana", y="% Alertas LH",
            title="SAV LH — % de voos com alerta por semana",
            color_discrete_sequence=["#ef4444"],
        )
        fig.update_layout(yaxis_tickformat=".0%", height=260)
        st.plotly_chart(fig, use_container_width=True)

# Oxy: delta de pressão ao longo do tempo
with right:
    if not df_oxy.empty and "date" in df_oxy.columns and "delta_press" in df_oxy.columns:
        df_oxy["date"] = pd.to_datetime(df_oxy["date"], errors="coerce")
        fig2 = px.line(
            df_oxy.dropna(subset=["date"]).sort_values("date"),
            x="date", y="delta_press",
            color="aircraftSerNum-1" if "aircraftSerNum-1" in df_oxy.columns else None,
            title="Oxigênio — perda de pressão por voo",
            labels={"delta_press": "Queda de pressão (PSI)", "date": ""},
        )
        fig2.update_layout(height=260)
        st.plotly_chart(fig2, use_container_width=True)

st.info("Use o menu lateral para acessar os dashboards detalhados de cada sistema.")
