"""
Página Combustível — Consumo por fase de voo

Monitora quanto combustível cada aeronave consome em cada fase do voo.
Um consumo acima do esperado na cruzeiro pode indicar degradação dos motores.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Consumo de Combustível", layout="wide")

st.title("⛽ Consumo de Combustível")
st.markdown(
    "Acompanha quanto combustível cada aeronave usa em cada fase do voo. "
    "Um consumo crescente na **cruzeiro** pode indicar degradação dos motores, "
    "problemas aerodinâmicos ou ajuste de plano de voo ineficiente."
)

# ── Dados ─────────────────────────────────────────────────────────────────────
df = load("e2_fuel_report.parquet")

if df.empty:
    st.error("Dados ainda não disponíveis. Execute o job `save_fuel_consumption_report` no Dagster.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

AC_COL = "ac_sn" if "ac_sn" in df.columns else None

# Fases disponíveis no dataset
_PHASE_MAP = {
    "taxi_out":      "Taxi (saída)",
    "take_off":      "Decolagem",
    "second_segment": "2º segmento",
    "initial_climb": "Subida inicial",
    "climb":         "Subida",
    "cruise":        "Cruzeiro",
    "descent":       "Descida",
    "approach":      "Aproximação",
    "final_approach": "Aprox. final",
    "landing":       "Pouso",
    "taxi_in":       "Taxi (chegada)",
}

# Identificar colunas de combustível disponíveis
burn_cols = {}
for phase_en, phase_pt in _PHASE_MAP.items():
    for eng in (1, 2):
        col = f"{phase_en}fuelMeterFuelBurn{eng}Kg"
        if col in df.columns:
            burn_cols[col] = (phase_pt, f"Motor {eng}")

if AC_COL:
    all_ac = sorted(df[AC_COL].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Filtrar aeronave(s)", options=all_ac,
                              default=all_ac[:10] if len(all_ac) > 10 else all_ac)
    if selected:
        df = df[df[AC_COL].astype(str).isin(selected)]

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
cruise_cols = [c for c in burn_cols if "cruise" in c]
total_burn_cols = list(burn_cols.keys())

avg_cruise = df[cruise_cols].sum(axis=1).mean() if cruise_cols else 0
avg_total  = df[total_burn_cols].sum(axis=1).mean() if total_burn_cols else 0
n_flights  = len(df)

c1, c2, c3 = st.columns(3)
c1.metric("Voos analisados", f"{n_flights:,}")
c2.metric("Combustível médio na cruzeiro (kg)", f"{avg_cruise:.0f}")
c3.metric("Combustível médio total por voo (kg)", f"{avg_total:.0f}")

# ── Gráfico 1: distribuição por fase ──────────────────────────────────────────
st.subheader("Onde o combustível é gasto?")
st.caption("Mostra a distribuição média do combustível consumido em cada fase do voo.")

if burn_cols:
    phase_totals = {}
    for col, (phase_pt, motor) in burn_cols.items():
        label = f"{phase_pt} ({motor})"
        phase_totals[label] = df[col].mean()

    phase_df = pd.DataFrame(list(phase_totals.items()), columns=["Fase", "Média (kg)"])
    phase_df = phase_df.sort_values("Média (kg)", ascending=False)

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = px.pie(
            phase_df, names="Fase", values="Média (kg)",
            title="Proporção por fase (médio)",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            phase_df, y="Fase", x="Média (kg)",
            orientation="h",
            title="Consumo médio por fase (kg)",
            color="Média (kg)",
            color_continuous_scale=["#bbf7d0", "#fbbf24", "#ef4444"],
        )
        fig_bar.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Gráfico 2: eficiência por aeronave na cruzeiro ────────────────────────────
st.subheader("Eficiência na cruzeiro por aeronave")
st.caption("Menor consumo na cruzeiro = motor mais eficiente. Barra vermelha = acima da média da frota.")

if cruise_cols and AC_COL:
    df["combustivel_cruzeiro_total"] = df[cruise_cols].sum(axis=1)
    eff = (
        df.groupby(AC_COL)["combustivel_cruzeiro_total"]
        .mean()
        .reset_index()
        .rename(columns={"combustivel_cruzeiro_total": "Combustível médio cruzeiro (kg)"})
        .sort_values("Combustível médio cruzeiro (kg)", ascending=False)
    )
    media_frota = eff["Combustível médio cruzeiro (kg)"].mean()
    eff["cor"] = eff["Combustível médio cruzeiro (kg)"].apply(
        lambda x: "#ef4444" if x > media_frota * 1.05 else "#22c55e"
    )

    fig_eff = go.Figure(go.Bar(
        x=eff["Combustível médio cruzeiro (kg)"],
        y=eff[AC_COL].astype(str),
        orientation="h",
        marker_color=eff["cor"],
    ))
    fig_eff.add_vline(
        x=media_frota, line_dash="dash", line_color="gray",
        annotation_text="média da frota", annotation_position="top right",
    )
    fig_eff.update_layout(
        title="Combustível médio na cruzeiro por aeronave",
        xaxis_title="kg", yaxis_title="Aeronave",
        height=max(300, len(eff) * 28),
    )
    st.plotly_chart(fig_eff, use_container_width=True)

# ── Gráfico 3: tendência mensal de consumo ────────────────────────────────────
st.subheader("Tendência mensal de consumo na cruzeiro")
st.caption("Um aumento constante ao longo dos meses pode indicar degradação dos motores.")

if cruise_cols and "date" in df.columns:
    df["combustivel_cruzeiro_total"] = df[cruise_cols].sum(axis=1)
    monthly = (
        df.dropna(subset=["date"])
        .set_index("date")
        .resample("M")["combustivel_cruzeiro_total"]
        .mean()
        .reset_index()
        .rename(columns={"date": "Mês", "combustivel_cruzeiro_total": "Combustível médio (kg)"})
    )

    fig_trend = px.line(
        monthly, x="Mês", y="Combustível médio (kg)",
        title="Consumo médio na cruzeiro — tendência mensal da frota",
        markers=True,
        color_discrete_sequence=["#3b82f6"],
    )
    # Linha de tendência
    if len(monthly) > 2:
        import numpy as np
        x_num = (monthly["Mês"] - monthly["Mês"].min()).dt.days
        z = np.polyfit(x_num, monthly["Combustível médio (kg)"].fillna(0), 1)
        trend_y = np.polyval(z, x_num)
        fig_trend.add_scatter(
            x=monthly["Mês"], y=trend_y,
            mode="lines", name="Tendência",
            line=dict(dash="dot", color="orange"),
        )
    fig_trend.update_layout(height=340)
    st.plotly_chart(fig_trend, use_container_width=True)
