# Fleet Briefing — 2026-05-28

## 🚨 Immediate Actions Required

**Data indisponível para avaliação completa** — os parquet files de SAV (LH/RH), W&B, OXY e FUEL não foram encontrados em `D:\parquet_files\`. Nenhuma ação baseada em preditivos pode ser emitida para esses sistemas hoje.

> Ação imediata: verificar se os jobs `save_sav_report`, `save_wnb_report`, `save_oxy_report` e `save_fuel_consumption_report` executaram com sucesso nas últimas 24h via Dagster UI (http://localhost:3000).

## ⚠️ Monitor Closely (next 5 flights)

- **FOQA — 402 voos registrados** nos últimos 30 dias, mas o resumo estatístico foi truncado antes dos valores reais (min/max/mean). Não é possível confirmar se exceedances ocorreram sem os dados completos.
- Colunas de exceedance presentes: `itt_lh/rh_takeoff_exceedance`, `n2_vib_lh/rh_amber`, `n2_vib_lh/rh_borescope`, `hard_landing_flag`, `oil_press_lh/rh_low` — flags existem mas contagens True/False não foram recebidas nesta sessão.

## ✅ Fleet Health Overview

A infraestrutura FOQA está operacional com 402 registros de voo coletados. No entanto, a ausência dos relatórios preditivos de SAV, W&B, OXY e FUEL impede uma avaliação de saúde completa da frota hoje. O pipeline de geração de parquet precisa ser investigado antes do próximo briefing.

## 📈 Trend Highlights

- **SAV:** Indisponível — parquet não encontrado
- **W&B:** Indisponível — parquet não encontrado
- **Engine (FOQA):** 402 voos capturados; estatísticas de ITT, N2 vib e CAS presentes no schema mas valores não recebidos nesta execução
- **Oxygen:** Indisponível — parquet não encontrado

## 🔧 Recommended Maintenance Actions

| Aircraft | System | Priority | Action |
|----------|--------|----------|--------|
| Toda frota | Pipeline dados | P1 | Verificar falhas nos jobs Dagster: `save_sav_report`, `save_wnb_report`, `save_oxy_report`, `save_fuel_consumption_report` |
| Toda frota | FOQA | P2 | Reprocessar briefing após restaurar parquets — verificar exceedances ITT/N2 vib/hard landing nos 402 voos |
| — | — | — | Sem ações de manutenção específicas por aeronave possíveis hoje |

---
*Nota: Briefing gerado com dados parciais. Parquets ausentes indicam possível falha nos schedules ou caminho `D:\parquet_files\` inacessível. Prioridade: restaurar pipeline antes do briefing das 13:00.*