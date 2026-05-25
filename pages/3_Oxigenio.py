"""
Página Oxigênio — Sistema de Oxigênio da Tripulação

O cilindro de oxigênio da tripulação perde pressão naturalmente ao longo
do tempo. Uma queda maior do que a esperada indica vazamento e exige
inspeção antes do próximo voo.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Oxigênio da Tripulação", layout="wide")

st.title("💨 Oxigênio da Tripulação")
st.markdown(
    "O **cilindro de oxigênio** da tripulação perde uma pequena quantidade de pressão a cada voo. "
    "Quando a queda é maior do que o esperado, pode indicar **vazamento** — "
    "situação que exige manutenção antes do próximo voo."
)

# ── Dados ─────────────────────────────────────────────────────────────────────
df = load("e2_oxy_report.parquet")

if df.empty:
    st.error("Dados ainda não disponíveis. Execute o job `save_oxy_report` no Dagster.")
    st.stop()

AC_COL = "aircraftSerNum-1" if "aircraftSerNum-1" in df.columns else None

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if AC_COL:
    df[AC_COL] = df[AC_COL].astype(str)

all_ac = sorted(df[AC_COL].dropna().unique().tolist()) if AC_COL else []
selected = st.multiselect("Filtrar aeronave(s)", options=all_ac, default=all_ac[:10] if len(all_ac) > 10 else all_ac)
if selected and AC_COL:
    df = df[df[AC_COL].isin(selected)]

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_alert = 0
n_total = 0
if AC_COL and "alert" in df.columns:
    latest = df.sort_values("date").groupby(AC_COL).last()
    n_alert = int(latest["alert"].sum())
    n_total = len(latest)

avg_loss = df["delta_press"].mean() if "delta_press" in df.columns else 0
max_loss = df["delta_press"].max() if "delta_press" in df.columns else 0

c1, c2, c3 = st.columns(3)
c1.metric("🔴 Aeronaves em alerta", n_alert,
          help="Aeronaves com queda de pressão acima do limiar")
c2.metric("Perda média por voo (PSI)", f"{avg_loss:.2f}")
c3.metric("Maior perda registrada (PSI)", f"{max_loss:.2f}")

# ── Gráfico 1: status atual por aeronave ─────────────────────────────────────
st.subheader("Status atual de cada aeronave")
st.caption("Verde = pressão normal   |   Vermelho = alerta de queda acima do esperado")

if AC_COL and "alert" in df.columns and "delta_press" in df.columns:
    status_df = (
        df.sort_values("date")
        .groupby(AC_COL)
        .last()[["delta_press", "alert"]]
        .reset_index()
        .sort_values("delta_press", ascending=False)
    )
    status_df["Status"] = status_df["alert"].map({True: "⚠️ Alerta", False: "Normal"})
    status_df["cor"] = status_df["alert"].map({True: "#ef4444", False: "#22c55e"})
    status_df[AC_COL] = status_df[AC_COL].astype(str)

    fig_status = go.Figure(go.Bar(
        y=status_df[AC_COL],
        x=status_df["delta_press"],
        orientation="h",
        marker_color=status_df["cor"],
        text=status_df["Status"],
        textposition="inside",
    ))

    if "smoothed" in df.columns:
        limiar = df["delta_press"].mean() + df["delta_press"].std()
        fig_status.add_vline(
            x=limiar, line_dash="dot", line_color="orange",
            annotation_text="limiar de alerta", annotation_position="top right",
        )

    fig_status.update_layout(
        title="Perda de pressão no último voo por aeronave",
        xaxis_title="Queda de pressão (PSI)",
        yaxis_title="Aeronave",
        height=max(300, len(status_df) * 28),
    )
    st.plotly_chart(fig_status, use_container_width=True)

# ── Gráfico 2: histórico de perda de pressão ─────────────────────────────────
st.subheader("Histórico de perda de pressão por voo")
st.caption("A linha tracejada é a média. Pontos acima da linha laranja acionam alerta.")

if "delta_press" in df.columns and "date" in df.columns:
    df_sorted = df.dropna(subset=["date"]).sort_values("date")

    fig_hist = px.line(
        df_sorted,
        x="date",
        y="smoothed" if "smoothed" in df.columns else "delta_press",
        color=AC_COL if AC_COL else None,
        labels={
            "smoothed": "Queda de pressão (suavizada)",
            "delta_press": "Queda de pressão (PSI)",
            "date": "",
            AC_COL: "Aeronave",
        },
        title="Tendência de perda de pressão de oxigênio",
    )

    if "delta_press" in df.columns:
        limiar = df["delta_press"].mean() + df["delta_press"].std()
        fig_hist.add_hline(
            y=limiar, line_dash="dot", line_color="orange",
            annotation_text="limiar de alerta", annotation_position="top right",
        )

    # Marcar voos com alerta
    if "alert" in df_sorted.columns and AC_COL:
        alertas = df_sorted[df_sorted["alert"] == True]
        if not alertas.empty:
            y_col = "smoothed" if "smoothed" in alertas.columns else "delta_press"
            fig_hist.add_scatter(
                x=alertas["date"], y=alertas[y_col],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Voo com alerta",
            )

    fig_hist.update_layout(height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Gráfico 3: voos com maior perda ──────────────────────────────────────────
st.subheader("Voos com maior perda de pressão")
st.caption("Estes voos exigem atenção prioritária da manutenção.")

if "delta_press" in df.columns and "date" in df.columns:
    top_cols = ["date", AC_COL, "delta_press"] if AC_COL else ["date", "delta_press"]
    top_cols = [c for c in top_cols if c in df.columns]
    top = (
        df.nlargest(15, "delta_press")[top_cols]
        .rename(columns={
            "date": "Data",
            AC_COL: "Aeronave",
            "delta_press": "Perda de pressão (PSI)",
        })
    )
    st.dataframe(top, use_container_width=True, hide_index=True)
