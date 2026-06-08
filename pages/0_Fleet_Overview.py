"""
Fleet Overview — Multi-fleet predictive maintenance status.
One row per aircraft, color-coded by worst alert across all monitored systems.
Covers E195-E2 (SAV, W&B, Oxygen, FOQA) + A320/A330 (FOQA).
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, make_prefix_map, display_name, clean_df

st.set_page_config(page_title="Fleet Overview", layout="wide", page_icon="🗺️")

PSI_CYAN  = 1155   # Oxygen: Observer Oxy Lo Press — cyan CAS
PSI_AMBER = 845    # Oxygen: Crew Oxy Lo Press — amber CAS / no dispatch

# ── Load all parquets (each is non-fatal if unavailable) ──────────────────────
@st.cache_data(ttl=300)
def _load_all() -> dict:
    datasets = {}
    for key, filename in {
        "sav_lh":  "e2_sav_lh_report.parquet",
        "sav_rh":  "e2_sav_rh_report.parquet",
        "oxy":     "e2_oxy_report.parquet",
        "foqa":    "e2_foqa_report.parquet",
        "wnb":     "e2_wnb_report.parquet",
        "fuel":    "e2_fuel_report.parquet",
        "a320":    "airbus_a320_foqa_report.parquet",
        "a330":    "airbus_a330_foqa_report.parquet",
    }.items():
        try:
            df = load(filename)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            datasets[key] = df
        except Exception:
            datasets[key] = pd.DataFrame()
    return datasets


data = _load_all()
prefix_map = make_prefix_map()

# Filter future dates and invalid serials from all E2 datasets
_e2_keys = ("sav_lh", "sav_rh", "oxy", "foqa", "wnb", "fuel")
for _k in _e2_keys:
    if _k in data and not data[_k].empty:
        _ac = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in data[_k].columns), None)
        data[_k] = clean_df(data[_k], date_col="date", ac_col=_ac, prefix_map=prefix_map)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back = st.slider("Days of history", 7, 90, 30)
    fleet_filter = st.multiselect(
        "Fleet",
        options=["E2", "A320", "A330"],
        default=["E2", "A320", "A330"],
    )

cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)


def _latest(df: pd.DataFrame, ac_col: str) -> pd.DataFrame:
    """Latest record per aircraft within the selected time window."""
    if df.empty or ac_col not in df.columns:
        return pd.DataFrame()
    sub = df[df["date"] >= cutoff] if "date" in df.columns else df
    if sub.empty:
        return pd.DataFrame()
    return sub.sort_values("date").groupby(ac_col).last().reset_index()


def _dnm(msn) -> str:
    return display_name(str(msn), prefix_map)


# ── Build per-system alert snapshots ─────────────────────────────────────────

# SAV LH
sav_lh_latest = _latest(data["sav_lh"], "ac_sn")
sav_lh_alert: dict = {}
if not sav_lh_latest.empty and "pre_lh_sav_failure_prediction" in sav_lh_latest.columns:
    sav_lh_alert = dict(zip(
        sav_lh_latest["ac_sn"].astype(str),
        sav_lh_latest["pre_lh_sav_failure_prediction"].astype(int),
    ))

# SAV RH
sav_rh_latest = _latest(data["sav_rh"], "ac_sn")
sav_rh_alert: dict = {}
if not sav_rh_latest.empty and "pre_rh_sav_failure_prediction" in sav_rh_latest.columns:
    sav_rh_alert = dict(zip(
        sav_rh_latest["ac_sn"].astype(str),
        sav_rh_latest["pre_rh_sav_failure_prediction"].astype(int),
    ))

# Oxygen
oxy_ac_col = next((c for c in ("aircraftSerNum-1", "ac_sn") if c in data["oxy"].columns), None)
oxy_psi: dict = {}
oxy_alert: dict = {}
if oxy_ac_col and "psi" in data["oxy"].columns and not data["oxy"].empty:
    oxy_lat = _latest(data["oxy"], oxy_ac_col)
    if not oxy_lat.empty:
        oxy_psi = dict(zip(oxy_lat[oxy_ac_col].astype(str), oxy_lat["psi"]))
        oxy_alert = {
            msn: (2 if psi < PSI_AMBER else 1 if psi < PSI_CYAN else 0)
            for msn, psi in oxy_psi.items()
        }

# FOQA E2
foqa_ac_col = "ac_sn" if "ac_sn" in data["foqa"].columns else None
foqa_exceedances: dict = {}
foqa_flag_cols = [
    "itt_lh_takeoff_exceedance", "itt_rh_takeoff_exceedance",
    "n2_vib_lh_amber", "n2_vib_rh_amber",
    "hard_landing_flag", "vmo_exceedance", "vle_exceedance", "apu_egt_exceedance",
]
if foqa_ac_col and not data["foqa"].empty:
    foqa_sub = data["foqa"][data["foqa"]["date"] >= cutoff] if "date" in data["foqa"].columns else data["foqa"]
    present_flags = [c for c in foqa_flag_cols if c in foqa_sub.columns]
    if present_flags:
        foqa_exceedances = (
            foqa_sub[present_flags].fillna(False).astype(bool).sum(axis=1)
            .groupby(foqa_sub[foqa_ac_col].astype(str))
            .sum()
            .to_dict()
        )

# FOQA Airbus
airbus_ac_col = "tail_number" if "tail_number" in data["a320"].columns else "ac_sn" if "ac_sn" in data["a320"].columns else None
airbus_alerts: dict = {}
for fleet_key, fleet_label in [("a320", "A320"), ("a330", "A330")]:
    df_ab = data[fleet_key]
    if df_ab.empty or airbus_ac_col is None or airbus_ac_col not in df_ab.columns:
        continue
    ab_sub = df_ab[df_ab["date"] >= cutoff] if "date" in df_ab.columns else df_ab
    exc_cols = [c for c in ab_sub.columns if c.endswith("_exceedance") or c.endswith("_flag")]
    if exc_cols:
        counts = (
            ab_sub[exc_cols].fillna(False).astype(bool).sum(axis=1)
            .groupby(ab_sub[airbus_ac_col].astype(str))
            .sum()
        )
        for tail, n in counts.items():
            airbus_alerts[f"{fleet_label}:{tail}"] = int(n)

# ── Collect all aircraft ──────────────────────────────────────────────────────
all_e2 = sorted(set(
    list(sav_lh_alert.keys())
    + list(sav_rh_alert.keys())
    + list(oxy_alert.keys())
    + list(foqa_exceedances.keys())
))

# ── KPI row ───────────────────────────────────────────────────────────────────
st.title("🗺️ Fleet Overview")
st.caption(f"Multi-fleet predictive maintenance · {date.today().strftime('%d-%b-%Y')} · {days_back}-day window")

n_sav_alert = sum(1 for v in {**sav_lh_alert, **sav_rh_alert}.values() if v == 1)
n_oxy_red   = sum(1 for v in oxy_alert.values() if v == 2)
n_oxy_amber = sum(1 for v in oxy_alert.values() if v == 1)
n_foqa      = sum(1 for v in foqa_exceedances.values() if v > 0)
n_airbus    = sum(1 for v in airbus_alerts.values() if v > 0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("✈️ E2 aircraft tracked", len(all_e2))
c2.metric("🔴 SAV alerts (LH+RH)", n_sav_alert,
          help="Aircraft with predicted pre-failure on starter valve (latest flight)")
c3.metric("💨 Oxy — no dispatch", n_oxy_red,
          help=f"Latest PSI < {PSI_AMBER} — amber CAS, no departure")
c4.metric("🔍 E2 FOQA exceedances", n_foqa,
          help=f"Aircraft with ≥1 engine/aircraft exceedance in last {days_back} days")
c5.metric("🛫 Airbus FOQA alerts", n_airbus,
          help=f"A320/A330 aircraft with exceedances in last {days_back} days")

st.divider()

# ── Immediate Actions banner ──────────────────────────────────────────────────
immediate_items = []

for msn in all_e2:
    if sav_lh_alert.get(msn) == 1:
        immediate_items.append(f"**{_dnm(msn)}** — SAV LH: predicted pre-failure (check ATS valve ATA 80)")
    if sav_rh_alert.get(msn) == 1:
        immediate_items.append(f"**{_dnm(msn)}** — SAV RH: predicted pre-failure (check ATS valve ATA 80)")
    if oxy_alert.get(msn) == 2:
        psi = oxy_psi.get(msn, 0)
        immediate_items.append(f"**{_dnm(msn)}** — Oxygen: {psi:.0f} PSI < {PSI_AMBER} PSI — no dispatch, QRH action required (ATA 35)")

for key, n in airbus_alerts.items():
    if n > 0:
        fleet, tail = key.split(":", 1)
        immediate_items.append(f"**{fleet} {tail}** — FOQA: {n} exceedance(s) in last {days_back} days")

if immediate_items:
    st.error("**🚨 Immediate Actions Required**\n\n" + "\n\n".join(f"- {x}" for x in immediate_items))

# Watch list
watch_items = []
for msn in all_e2:
    if oxy_alert.get(msn) == 1:
        psi = oxy_psi.get(msn, 0)
        watch_items.append(f"**{_dnm(msn)}** — Oxygen: {psi:.0f} PSI ({PSI_AMBER}–{PSI_CYAN}) — monitor, possible OBSERVER OXY LO PRESS")
    if foqa_exceedances.get(msn, 0) > 0:
        n = foqa_exceedances[msn]
        watch_items.append(f"**{_dnm(msn)}** — FOQA: {n} exceedance event(s) this period")

if watch_items:
    st.warning("**⚠️ Monitor Closely**\n\n" + "\n\n".join(f"- {x}" for x in watch_items))

if not immediate_items and not watch_items:
    st.success(f"✅ No critical alerts in the last {days_back} days across all monitored systems.")

st.divider()

# ── Fleet Health Matrix ───────────────────────────────────────────────────────
st.subheader("E2 Fleet — Health Matrix")
st.caption(
    "Latest status per aircraft × system. "
    "🔴 = alert/action required · 🟡 = monitor · 🟢 = normal · — = no data in period"
)

if "E2" in fleet_filter and all_e2:
    matrix_rows = []
    for msn in sorted(all_e2):
        dn = _dnm(msn)

        # SAV LH
        sav_lh_cell = "🔴" if sav_lh_alert.get(msn) == 1 else ("🟢" if msn in sav_lh_alert else "—")
        # SAV RH
        sav_rh_cell = "🔴" if sav_rh_alert.get(msn) == 1 else ("🟢" if msn in sav_rh_alert else "—")
        # Oxygen
        oa = oxy_alert.get(msn)
        oxy_cell = ("🔴" if oa == 2 else "🟡" if oa == 1 else ("🟢" if msn in oxy_alert else "—"))
        psi_val = f"{oxy_psi[msn]:.0f}" if msn in oxy_psi else "—"
        # FOQA
        foqa_n = foqa_exceedances.get(msn, None)
        foqa_cell = ("🔴" if foqa_n and foqa_n > 0 else ("🟢" if foqa_n == 0 else "—"))

        # Overall worst
        cells = [sav_lh_cell, sav_rh_cell, oxy_cell, foqa_cell]
        if "🔴" in cells:
            worst = "🔴 Critical"
        elif "🟡" in cells:
            worst = "🟡 Monitor"
        elif "🟢" in cells:
            worst = "🟢 Normal"
        else:
            worst = "— No data"

        matrix_rows.append({
            "Aircraft": dn,
            "MSN": msn,
            "Overall": worst,
            "SAV LH": sav_lh_cell,
            "SAV RH": sav_rh_cell,
            "Oxygen": f"{oxy_cell} {psi_val} PSI".strip(),
            "FOQA": f"{foqa_cell} ({foqa_n})" if foqa_n is not None else foqa_cell,
        })

    if matrix_rows:
        df_matrix = pd.DataFrame(matrix_rows).sort_values(
            "Overall",
            key=lambda s: s.map({"🔴 Critical": 0, "🟡 Monitor": 1, "🟢 Normal": 2, "— No data": 3}),
        )

        def _color_matrix(row):
            o = row.get("Overall", "")
            if "🔴" in o:
                return ["background-color: rgba(239,68,68,0.12)"] * len(row)
            elif "🟡" in o:
                return ["background-color: rgba(245,158,11,0.10)"] * len(row)
            return [""] * len(row)

        display_cols = ["Aircraft", "Overall", "SAV LH", "SAV RH", "Oxygen", "FOQA"]
        st.dataframe(
            df_matrix[display_cols].style.apply(_color_matrix, axis=1),
            use_container_width=True,
            hide_index=True,
        )

# ── Airbus FOQA summary ───────────────────────────────────────────────────────
for fleet_key, fleet_label in [("a320", "A320"), ("a330", "A330")]:
    if fleet_label not in fleet_filter:
        continue
    df_ab = data[fleet_key]
    if df_ab.empty:
        continue

    st.divider()
    st.subheader(f"{fleet_label} Fleet — FOQA Exceedance Summary")

    ab_col = airbus_ac_col
    if ab_col is None or ab_col not in df_ab.columns:
        st.info(f"No tail number column found in {fleet_label} FOQA data.")
        continue

    ab_sub = df_ab[df_ab["date"] >= cutoff] if "date" in df_ab.columns else df_ab
    exc_cols = [c for c in ab_sub.columns if c.endswith("_exceedance") or c.endswith("_flag")]
    if not exc_cols:
        st.info(f"No exceedance columns found in {fleet_label} FOQA data.")
        continue

    ab_counts = (
        ab_sub.assign(_total=ab_sub[exc_cols].fillna(False).astype(bool).sum(axis=1))
        .groupby(ab_col)["_total"].sum()
        .reset_index()
        .rename(columns={ab_col: "Aircraft", "_total": "Exceedances"})
        .sort_values("Exceedances", ascending=False)
    )
    ab_counts["Display"] = ab_counts["Aircraft"].apply(
        lambda t: display_name(str(t), prefix_map)
    )

    fig_ab = go.Figure(go.Bar(
        x=ab_counts["Display"],
        y=ab_counts["Exceedances"],
        marker_color=ab_counts["Exceedances"].apply(
            lambda n: "#ef4444" if n > 5 else "#f59e0b" if n > 0 else "#22c55e"
        ),
        text=ab_counts["Exceedances"],
        textposition="outside",
        hovertemplate="%{x}: %{y} exceedance event(s)<extra></extra>",
    ))
    fig_ab.update_layout(
        title=f"{fleet_label} — Exceedance events per aircraft (last {days_back} days)",
        xaxis_title="Aircraft",
        yaxis_title="Total exceedance events",
        height=340,
    )
    st.plotly_chart(fig_ab, use_container_width=True)

# ── Data freshness ────────────────────────────────────────────────────────────
st.divider()
with st.expander("Data freshness — last update per system"):
    for label, df_key, ac_col_name in [
        ("SAV LH", "sav_lh", "ac_sn"),
        ("SAV RH", "sav_rh", "ac_sn"),
        ("Oxygen", "oxy", oxy_ac_col),
        ("FOQA E2", "foqa", "ac_sn"),
        ("A320 FOQA", "a320", airbus_ac_col),
        ("A330 FOQA", "a330", airbus_ac_col),
    ]:
        df_chk = data[df_key]
        if df_chk.empty or "date" not in df_chk.columns:
            st.write(f"- **{label}**: no data")
            continue
        latest_dt = df_chk["date"].max()
        age = (pd.Timestamp.now() - latest_dt).days if pd.notna(latest_dt) else None
        status = f"✅" if age is not None and age <= 2 else "⚠️"
        st.write(
            f"- **{label}**: {status} last record {latest_dt.strftime('%d-%b-%Y') if pd.notna(latest_dt) else '—'}"
            + (f" ({age}d ago)" if age is not None else "")
        )
