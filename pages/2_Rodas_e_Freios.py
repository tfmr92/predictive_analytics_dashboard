"""
Página Wheels & Brakes — Rodas e Freios

Monitora a dureza dos pousos e o desgaste das rodas.
Pousos duros aceleram o desgaste de pneus, freios e estrutura.
O modelo prevê remoção de cada posição de roda antes da falha.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Rodas & Freios", layout="wide")

st.title("🛞 Rodas e Freios (Wheels & Brakes)")
st.markdown(
    "Monitoramento da **dureza dos pousos** e do desgaste das rodas. "
    "Um pouso duro (acima de **1,4 g**) desgasta freios, pneus e a estrutura da aeronave. "
    "O modelo identifica quais rodas estão próximas da necessidade de remoção."
)

# ── Dados ─────────────────────────────────────────────────────────────────────
df = load("e2_wnb_report.parquet")

if df.empty:
    st.error("Dados ainda não disponíveis. Execute o job `save_wheel_brake_report` no Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "ac_sn" in df.columns:
    df["ac_sn"] = df["ac_sn"].astype(str)

# Filtros
all_ac = sorted(df["ac_sn"].dropna().unique().tolist()) if "ac_sn" in df.columns else []
selected = st.multiselect("Filtrar aeronave(s)", options=all_ac, default=all_ac[:10] if len(all_ac) > 10 else all_ac)
if selected and "ac_sn" in df.columns:
    df = df[df["ac_sn"].isin(selected)]

_HARD_LIMIT = 1.4   # g
_SEVERE_LIMIT = 2.0  # g

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
hard_lh = int((df["NormAccel_lh"] > _HARD_LIMIT).sum()) if "NormAccel_lh" in df.columns else 0
hard_rh = int((df["NormAccel_rh"] > _HARD_LIMIT).sum()) if "NormAccel_rh" in df.columns else 0
bounce_max = int(df[["bouncing_count_lh", "bouncing_count_rh"]].max().max()) if all(
    c in df.columns for c in ("bouncing_count_lh", "bouncing_count_rh")
) else 0

pred_cols = [c for c in df.columns if c.startswith("prediction_")]
total_alerts = int(df[pred_cols].eq(1).any(axis=1).sum()) if pred_cols else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pousos duros (LH > 1,4 g)", hard_lh)
c2.metric("Pousos duros (RH > 1,4 g)", hard_rh)
c3.metric("Máx. de ricochetes em um pouso", bounce_max)
c4.metric("🔴 Voos com alerta de remoção", total_alerts)

# ── Gráfico 1: dureza dos pousos ──────────────────────────────────────────────
st.subheader("Dureza dos pousos por voo")
st.caption(f"Linha laranja = {_HARD_LIMIT} g (limite atenção)  |  Linha vermelha = {_SEVERE_LIMIT} g (limite crítico)")

tab_lh, tab_rh = st.tabs(["Trem Principal Esquerdo (LH)", "Trem Principal Direito (RH)"])

for tab, acol, label in [
    (tab_lh, "NormAccel_lh", "LH"),
    (tab_rh, "NormAccel_rh", "RH"),
]:
    with tab:
        if acol not in df.columns:
            st.info("Coluna não disponível.")
            continue
        df_plot = df.dropna(subset=["date", acol]).copy()
        df_plot["Classificação"] = pd.cut(
            df_plot[acol],
            bins=[-999, _HARD_LIMIT, _SEVERE_LIMIT, 999],
            labels=["Normal", "Atenção", "Crítico"],
        )
        color_map = {"Normal": "#22c55e", "Atenção": "#f59e0b", "Crítico": "#ef4444"}
        fig = px.scatter(
            df_plot.sort_values("date"),
            x="date", y=acol,
            color="Classificação",
            color_discrete_map=color_map,
            hover_data={"ac_sn": True} if "ac_sn" in df_plot.columns else {},
            labels={acol: "Aceleração no pouso (g)", "date": ""},
            title=f"Força no pouso — Trem {label}",
        )
        fig.add_hline(y=_HARD_LIMIT, line_dash="dot", line_color="orange",
                      annotation_text="atenção", annotation_position="top right")
        fig.add_hline(y=_SEVERE_LIMIT, line_dash="dot", line_color="red",
                      annotation_text="crítico", annotation_position="top right")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

# ── Gráfico 2: ricochetes por aeronave ────────────────────────────────────────
st.subheader("Ricochetes no pouso por aeronave")
st.caption("Cada ricochete = a aeronave voltou a subir após tocar a pista. Indica pouso instável.")

if all(c in df.columns for c in ("bouncing_count_lh", "bouncing_count_rh", "ac_sn")):
    bounce_agg = (
        df.groupby("ac_sn")[["bouncing_count_lh", "bouncing_count_rh"]]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("bouncing_count_lh", ascending=False)
    )
    fig_bounce = go.Figure()
    fig_bounce.add_bar(x=bounce_agg["ac_sn"], y=bounce_agg["bouncing_count_lh"],
                       name="LH", marker_color="#3b82f6")
    fig_bounce.add_bar(x=bounce_agg["ac_sn"], y=bounce_agg["bouncing_count_rh"],
                       name="RH", marker_color="#8b5cf6")
    fig_bounce.update_layout(
        barmode="group",
        title="Média de ricochetes por aeronave",
        xaxis_title="Aeronave", yaxis_title="Ricochetes (média)",
        height=340,
    )
    st.plotly_chart(fig_bounce, use_container_width=True)

# ── Gráfico 3: previsão de remoção por posição ────────────────────────────────
st.subheader("Previsão de remoção por posição de roda")
st.caption("Vermelho = modelo detectou sinais de desgaste acelerado nesta posição.")

_POS_LABELS = {
    "prediction_mlg1": "MLG 1\n(LH frente)",
    "prediction_mlg2": "MLG 2\n(LH trás)",
    "prediction_mlg3": "MLG 3\n(RH frente)",
    "prediction_mlg4": "MLG 4\n(RH trás)",
    "prediction_nlg_lh": "NLG LH\n(nariz esq.)",
    "prediction_nlg_rh": "NLG RH\n(nariz dir.)",
}

available_preds = [c for c in _POS_LABELS if c in df.columns]
if available_preds and "ac_sn" in df.columns:
    heatmap_data = (
        df.groupby("ac_sn")[available_preds]
        .apply(lambda g: (g == 1).mean())
        .rename(columns=_POS_LABELS)
        .reset_index()
    )
    heatmap_melted = heatmap_data.melt(id_vars="ac_sn", var_name="Posição", value_name="% alertas")
    fig_heat = px.density_heatmap(
        heatmap_melted, x="Posição", y="ac_sn", z="% alertas",
        color_continuous_scale=["#dcfce7", "#fef9c3", "#fca5a5", "#ef4444"],
        labels={"ac_sn": "Aeronave", "% alertas": "% voos em alerta"},
        title="Mapa de calor — risco de remoção por aeronave e posição",
    )
    fig_heat.update_layout(height=max(300, len(heatmap_data) * 30))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Top 10 pousos mais duros ────────────────────────────────────────────────
st.subheader("Top 10 pousos mais duros do período")
if "NormAccel_lh" in df.columns and "date" in df.columns:
    top = (
        df.nlargest(10, "NormAccel_lh")[["date", "ac_sn", "NormAccel_lh", "NormAccel_rh",
                                          "bouncing_count_lh", "bouncing_count_rh"]]
        .rename(columns={
            "date": "Data", "ac_sn": "Aeronave",
            "NormAccel_lh": "Aceleração LH (g)", "NormAccel_rh": "Aceleração RH (g)",
            "bouncing_count_lh": "Ricochetes LH", "bouncing_count_rh": "Ricochetes RH",
        })
    )
    st.dataframe(top, use_container_width=True, hide_index=True)
