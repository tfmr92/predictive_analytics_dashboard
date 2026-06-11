"""
Airbus FOQA/MOQA — A320/A330 exceedance monitoring from decoded QAR/DAR flights.
Speed envelope (VMO/MMO/VLE), EGT takeoff/continuous limits, engine vibration
advisories and oil-system flags per flight.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load

st.set_page_config(page_title="Airbus FOQA/MOQA", layout="wide")

st.title("🔍 FOQA / MOQA — Airbus A320 & A330")
st.markdown(
    "Per-flight exceedance monitoring from decoded QAR/DAR data: speed envelope "
    "(VMO/MMO/VLE), EGT takeoff & continuous limits, N1/N2 vibration advisories "
    "and oil-system flags."
)

AC_COL = "tail_number"

# Exceedance flags grouped by severity for triage
_LIMIT_FLAGS = [
    "vmo_exceeded", "mmo_exceeded", "vle_exceeded",
    "egt_takeoff_exceeded", "egt_continuous_exceeded",
    "n1_vib_limit_exc", "n2_vib_limit_exc", "tire_overspeed",
]
_ADVISORY_FLAGS = [
    "n1_vib_advisory", "n2_vib_advisory", "n3_vib_advisory",
    "oil_low_press_flag", "oil_high_press_flag", "oil_temp_high_flag", "oil_qty_low_flag",
]

_FLAG_LABELS = {
    "vmo_exceeded": "VMO exceeded",
    "mmo_exceeded": "MMO exceeded",
    "vle_exceeded": "VLE (gear ext.) exceeded",
    "egt_takeoff_exceeded": "EGT takeoff limit",
    "egt_continuous_exceeded": "EGT continuous limit",
    "n1_vib_limit_exc": "N1 vibration limit",
    "n2_vib_limit_exc": "N2 vibration limit",
    "tire_overspeed": "Tire overspeed",
    "n1_vib_advisory": "N1 vibration advisory",
    "n2_vib_advisory": "N2 vibration advisory",
    "n3_vib_advisory": "N3 vibration advisory",
    "oil_low_press_flag": "Oil pressure low",
    "oil_high_press_flag": "Oil pressure high",
    "oil_temp_high_flag": "Oil temperature high",
    "oil_qty_low_flag": "Oil quantity low",
}


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

_latest_date = max([d["date"].max() for d in (df_a320, df_a330) if not d.empty], default=None)
if _latest_date is not None and pd.notna(_latest_date):
    st.caption(f"Data through {_latest_date.strftime('%d-%b-%Y')} · auto-refreshed hourly")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 7, 365, 60)

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)


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
    st.header(f"✈️ {fleet_label}")

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
    c3.metric("🔴 Limit exceedances", n_limit,
              help="VMO/MMO/VLE, EGT takeoff/continuous, vibration limit, tire overspeed")
    c4.metric("🟡 Advisories", n_adv,
              help="Vibration advisories and oil pressure/temperature/quantity flags")

    # Triage banner
    crit = summary[summary["Limit exceedances"] > 0]
    if not crit.empty:
        st.error(
            "**🚨 Limit exceedances this period:**\n\n" + "\n".join(
                f"- **{r[AC_COL]}** — {int(r['Limit exceedances'])} limit event(s), "
                f"{int(r['Advisories'])} advisory(ies) in {int(r['Flights'])} flights"
                for _, r in crit.head(10).iterrows()
            )
        )
    else:
        st.success(f"✅ No limit exceedances in the last {days_back} days.")

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
    with st.expander(f"📈 Engine trends — {fleet_label}", expanded=False):
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
            with plot_cols[n_plotted % 2]:
                st.plotly_chart(fig_t, use_container_width=True)
            n_plotted += 1
        if n_plotted == 0:
            st.info("No engine trend columns available in this dataset.")

    # ── Per-flight detail table for flagged flights ────────────────────────────
    flag_cols_present = [c for c in _LIMIT_FLAGS + _ADVISORY_FLAGS if c in sub.columns]
    flagged = sub[sub[flag_cols_present].fillna(False).astype(bool).any(axis=1)] if flag_cols_present else pd.DataFrame()
    if not flagged.empty:
        with st.expander(f"📋 {len(flagged)} flagged flight(s) — detail"):
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
