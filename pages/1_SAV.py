"""
Página SAV — Air Turbine Starter (Motor Starter)

O starter arranca o motor no início de cada voo. Quando ele começa a se
desgastar, o motor demora mais para ligar, a temperatura sobe e a velocidade
de rotação cai — sinais que o modelo detecta antes de a peça falhar de vez.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="SAV — Motor Starter", layout="wide")

st.title("⚙️ Motor Starter (SAV)")
st.markdown(
    "O **starter** é o motor que dá a partida ao motor principal. "
    "Falhas no starter causam atrasos ou AOG. "
    "O modelo detecta sinais de desgaste **até 80 voos antes** da falha."
)

# ── Dados ─────────────────────────────────────────────────────────────────────
df_lh = load("e2_sav_lh_report.parquet")
df_rh = load("e2_sav_rh_report.parquet")

if df_lh.empty and df_rh.empty:
    st.error("Dados ainda não disponíveis. Execute o job `save_sav_report` no Dagster.")
    st.stop()

for df in (df_lh, df_rh):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Filtro de aeronave
all_ac = sorted(
    set(df_lh["ac_sn"].dropna().astype(int).unique().tolist() if "ac_sn" in df_lh.columns else [])
    | set(df_rh["ac_sn"].dropna().astype(int).unique().tolist() if "ac_sn" in df_rh.columns else [])
)
selected = st.multiselect("Filtrar aeronave(s)", options=all_ac, default=all_ac[:10] if len(all_ac) > 10 else all_ac)

if selected:
    if "ac_sn" in df_lh.columns:
        df_lh = df_lh[df_lh["ac_sn"].astype(int).isin(selected)]
    if "ac_sn" in df_rh.columns:
        df_rh = df_rh[df_rh["ac_sn"].astype(int).isin(selected)]

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
def _latest_risk(df: pd.DataFrame, pred_col: str):
    """Retorna DataFrame com última predição por aeronave."""
    if df.empty or pred_col not in df.columns:
        return pd.DataFrame()
    return (
        df.dropna(subset=["date", "ac_sn"])
        .sort_values("date")
        .groupby("ac_sn")
        .last()[[pred_col, "date"]]
        .reset_index()
    )


risk_lh = _latest_risk(df_lh, "pre_lh_sav_failure_prediction")
risk_rh = _latest_risk(df_rh, "pre_rh_sav_failure_prediction")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aeronaves monitoradas (LH)", len(risk_lh))
c2.metric("🔴 Em alerta (LH)", int((risk_lh["pre_lh_sav_failure_prediction"] == 1).sum()) if not risk_lh.empty else 0)
c3.metric("Aeronaves monitoradas (RH)", len(risk_rh))
c4.metric("🔴 Em alerta (RH)", int((risk_rh["pre_rh_sav_failure_prediction"] == 1).sum()) if not risk_rh.empty else 0)

# ── Gráfico 1: situação atual por aeronave ─────────────────────────────────────
st.subheader("Situação atual — último voo registrado")
st.caption("Verde = sem alerta   |   Vermelho = modelo detectou pré-falha")

col_a, col_b = st.columns(2)

for side, risk_df, pred_col, title in [
    ("LH", risk_lh, "pre_lh_sav_failure_prediction", "Starter Esquerdo (LH)"),
    ("RH", risk_rh, "pre_rh_sav_failure_prediction", "Starter Direito (RH)"),
]:
    with (col_a if side == "LH" else col_b):
        if risk_df.empty:
            st.info(f"Sem dados para {title}.")
            continue
        risk_df = risk_df.copy()
        risk_df["status"] = risk_df[pred_col].map({0: "Normal", 1: "Alerta"})
        risk_df["cor"] = risk_df[pred_col].map({0: "#22c55e", 1: "#ef4444"})
        risk_df["ac_sn"] = risk_df["ac_sn"].astype(str)
        risk_df = risk_df.sort_values(pred_col, ascending=False)

        fig = go.Figure(go.Bar(
            y=risk_df["ac_sn"],
            x=risk_df[pred_col],
            orientation="h",
            marker_color=risk_df["cor"],
            text=risk_df["status"],
            textposition="inside",
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(tickvals=[0, 1], ticktext=["Normal", "Alerta"], range=[0, 1.1]),
            yaxis_title="Aeronave",
            height=max(300, len(risk_df) * 28),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Gráfico 2: linha do tempo de alertas ──────────────────────────────────────
st.subheader("Histórico de alertas — quando cada aeronave esteve em risco")
st.caption("Cada linha é uma aeronave. Pontos vermelhos = voos com alerta detectado.")

tab_lh, tab_rh = st.tabs(["Starter Esquerdo (LH)", "Starter Direito (RH)"])

for tab, df, pred_col, speed_col, itt_col, label in [
    (tab_lh, df_lh, "pre_lh_sav_failure_prediction", "starterSpeed-1a_max", "ittFADEC-1a_max", "LH"),
    (tab_rh, df_rh, "pre_rh_sav_failure_prediction", "starterSpeed-3a_max", "ittFADEC-3a_max", "RH"),
]:
    with tab:
        if df.empty or pred_col not in df.columns:
            st.info("Sem dados.")
            continue

        df = df.dropna(subset=["date"]).copy()
        df["ac_sn"] = df["ac_sn"].astype(str)
        df["Resultado"] = df[pred_col].map({0: "Normal", 1: "⚠️ Alerta"})

        # Timeline
        fig_hist = px.scatter(
            df, x="date", y="ac_sn",
            color="Resultado",
            color_discrete_map={"Normal": "#86efac", "⚠️ Alerta": "#ef4444"},
            labels={"date": "Data", "ac_sn": "Aeronave"},
            title=f"Starter {label} — alertas ao longo do tempo",
        )
        fig_hist.update_traces(marker_size=6)
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        col1, col2 = st.columns(2)

        # Velocidade do starter
        if speed_col in df.columns:
            with col1:
                fig_spd = px.line(
                    df.sort_values("date"), x="date", y=speed_col,
                    color="ac_sn",
                    title="Velocidade de rotação do Starter (%RPM)",
                    labels={speed_col: "Velocidade (%)", "date": "", "ac_sn": "Aeronave"},
                )
                fig_spd.add_hrect(
                    y0=0, y1=df[speed_col].quantile(0.10),
                    fillcolor="red", opacity=0.08,
                    annotation_text="zona de risco", annotation_position="top left",
                )
                fig_spd.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_spd, use_container_width=True)
                st.caption("⬇️ Velocidade caindo ao longo do tempo indica desgaste progressivo.")

        # Temperatura ITT
        if itt_col in df.columns:
            with col2:
                fig_itt = px.line(
                    df.sort_values("date"), x="date", y=itt_col,
                    color="ac_sn",
                    title="Temperatura no arranque — ITT (°C)",
                    labels={itt_col: "Temperatura (°C)", "date": "", "ac_sn": "Aeronave"},
                )
                p90 = df[itt_col].quantile(0.90)
                fig_itt.add_hline(
                    y=p90, line_dash="dot", line_color="orange",
                    annotation_text="limite atenção (P90)", annotation_position="top right",
                )
                fig_itt.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_itt, use_container_width=True)
                st.caption("⬆️ Temperatura acima do normal pode indicar arranque quente (hot-start).")
