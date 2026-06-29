"""
Fleet Overview — Multi-fleet predictive maintenance status.
One row per aircraft, color-coded by worst alert across all monitored systems.
Covers E195-E2 (SAV, W&B, Oxygen, FOQA, Fuel) + A320/A330 (FOQA).
"""

import math
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, make_prefix_map, display_name, clean_df, get_file_mtime

st.set_page_config(page_title="Fleet Overview", layout="wide")

PSI_CYAN  = 1155   # Oxygen: Observer Oxy Lo Press — cyan CAS
PSI_AMBER = 845    # Oxygen: Crew Oxy Lo Press — amber CAS / no dispatch
FORECAST_HORIZON_DAYS = 30  # fixed planning horizon for upcoming inspections

# E2 SAV is now sourced from the ACARS-validated start-transient model (the same
# report 1_SAV.py trusts), not the dead aggregate. The transient parquet carries no
# explicit alert column, so mirror 1_SAV.py _is_alert's fallback (prob >= the
# documented calibrated High band) and guard the call with a recency window so a
# stale flight's score cannot become a permanent false alert on the landing page.
_SAV_HIGH = 0.60          # mirrors 1_SAV.py _THRESHOLD_HIGH — calibrated High band
_SAV_RECENCY_DAYS = 30    # last_flight_dt must be within this window to alert

# Pipeline sources for the health panel: (label, df_key, report filename, producing Dagster job).
# 'Last refresh' keys off the Drive mtime of each filename — the producing job runs on a fixed
# schedule regardless of new flights, so a stale mtime means the job stopped, not that the fleet is idle.
PIPELINE_SOURCES = [
    ("SAV LH (E2)",         "sav_lh",      "e2_sav_transient_lh_report.parquet", "save_sav_transient_report"),
    ("SAV RH (E2)",         "sav_rh",      "e2_sav_transient_rh_report.parquet", "save_sav_transient_report"),
    ("SAV A320 — Eng 1",    "sav_a320_e1", "airbus_sav_eng1_report.parquet",  "save_airbus_sav_report"),
    ("SAV A320 — Eng 2",    "sav_a320_e2", "airbus_sav_eng2_report.parquet",  "save_airbus_sav_report"),
    ("Oxygen (E2)",         "oxy",         "e2_oxy_report.parquet",           "save_oxy_report"),
    ("FOQA E2",             "foqa",        "e2_foqa_report.parquet",          "e2_foqa_moqa_job"),
    ("A320 FOQA",           "a320",        "airbus_a320_foqa_report.parquet", "airbus_foqa_moqa_job"),
    ("A330 FOQA",           "a330",        "airbus_a330_foqa_report.parquet", "airbus_foqa_moqa_job"),
    ("Wheels & Brakes (E2)","wnb",         "e2_wnb_report.parquet",           "save_wheel_brake_report"),
    ("Fuel (E2)",           "fuel",        "e2_fuel_report.parquet",          "save_fuel_consumption_report"),
]

# ── Load all parquets (each is non-fatal if unavailable) ──────────────────────
@st.cache_data(ttl=300)
def _load_all() -> tuple[dict, dict]:
    datasets, errors = {}, {}
    for key, filename in {
        "sav_lh":      "e2_sav_transient_lh_report.parquet",
        "sav_rh":      "e2_sav_transient_rh_report.parquet",
        "oxy":         "e2_oxy_report.parquet",
        "foqa":        "e2_foqa_report.parquet",
        "wnb":         "e2_wnb_report.parquet",
        "fuel":        "e2_fuel_report.parquet",
        "a320":        "airbus_a320_foqa_report.parquet",
        "a330":        "airbus_a330_foqa_report.parquet",
        "sav_a320_e1": "airbus_sav_eng1_report.parquet",
        "sav_a320_e2": "airbus_sav_eng2_report.parquet",
    }.items():
        try:
            df = load(filename)
            # Some parquets carry 'flight_datetime' instead of 'date'
            # (Airbus FOQA: compact string YYYYMMDDHHMMSS; Airbus SAV: datetime64)
            if "date" not in df.columns and "flight_datetime" in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df["flight_datetime"]):
                    df["date"] = df["flight_datetime"]
                else:
                    df["date"] = pd.to_datetime(
                        df["flight_datetime"].astype(str), format="%Y%m%d%H%M%S",
                        errors="coerce",
                    )
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            datasets[key] = df
            if df.empty:
                errors[key] = "loaded 0 rows (file missing on Drive or unreadable)"
        except Exception as exc:
            datasets[key] = pd.DataFrame()
            errors[key] = f"{type(exc).__name__}: {exc}"
    return datasets, errors


data, data_errors = _load_all()
prefix_map = make_prefix_map()

# Filter future dates and invalid serials from all E2 datasets
_e2_keys = ("sav_lh", "sav_rh", "oxy", "foqa", "wnb", "fuel")
for _k in _e2_keys:
    if _k in data and not data[_k].empty:
        _ac = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in data[_k].columns), None)
        data[_k] = clean_df(data[_k], date_col="date", ac_col=_ac, prefix_map=prefix_map)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
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


@st.cache_data(ttl=300)
def _oxy_maintenance_forecast(df_oxy: pd.DataFrame, ac_col: str, cutoff_ts: pd.Timestamp) -> dict:
    """Fleet-wide forecast of aircraft projected to cross the 845 PSI amber
    (no-dispatch) threshold, reusing the oxygen dispatch-forecast method with the
    documented robustness lessons applied:
      - recency guard: only flights within the selected history window
      - minimum sample: require >= 4 in-window readings per aircraft
      - smoothed current reading: current_psi = median of last 5 PSI values
      - median daily drop (not mean) from delta_press; skip non-positive drop
    Returns {"rows": [...within horizon...], "n_evaluated": <int forecastable a/c>}.
    """
    if df_oxy is None or df_oxy.empty or not ac_col:
        return {"rows": [], "n_evaluated": 0}
    if (ac_col not in df_oxy.columns
            or "psi" not in df_oxy.columns
            or "delta_press" not in df_oxy.columns):
        return {"rows": [], "n_evaluated": 0}

    sub = df_oxy.copy()
    if "date" in sub.columns:
        sub = sub[sub["date"] >= cutoff_ts]
    sub = sub.dropna(subset=["psi", "delta_press", ac_col])
    if sub.empty:
        return {"rows": [], "n_evaluated": 0}

    today = pd.Timestamp.now().normalize()
    rows: list = []
    n_evaluated = 0

    for msn, grp in sub.groupby(ac_col):
        grp = grp.sort_values("date") if "date" in grp.columns else grp
        if len(grp) < 4:                            # minimum sample guard
            continue
        current_psi = float(grp["psi"].tail(5).median())   # smoothed current reading
        daily_drop = float(grp["delta_press"].median())    # median, not mean
        if daily_drop <= 0:                          # skip non-positive drop
            continue
        n_evaluated += 1
        days_to_amber = (current_psi - PSI_AMBER) / daily_drop
        if days_to_amber <= 0 or days_to_amber > FORECAST_HORIZON_DAYS:
            continue
        rows.append({
            "msn": str(msn),
            "current_psi": current_psi,
            "daily_drop": daily_drop,
            "days_to_amber": days_to_amber,
            "est_date": today + pd.Timedelta(days=days_to_amber),
        })

    rows.sort(key=lambda r: r["days_to_amber"])
    return {"rows": rows, "n_evaluated": n_evaluated}


@st.cache_data(ttl=300)
def _wnb_alerts(df_wnb: pd.DataFrame, cutoff_ts: pd.Timestamp) -> dict:
    """Wheels & Brakes (ATA 32) removal alerts from the LONG-format W&B report.

    The W&B report carries one row per (aircraft, brake/gear position) — up to 6
    positions per tail — so collapsing with _latest would drop 5 of 6 positions.
    Instead this takes the latest record per (aircraft, position), then OR-aggregates
    the binary prediction columns across positions: a tail is flagged if ANY position
    predicts a removal. Returns {msn: 1|0}; empty dict if no data / no prediction cols.
    """
    if df_wnb is None or df_wnb.empty:
        return {}
    ac_col = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in df_wnb.columns), None)
    if ac_col is None or "position" not in df_wnb.columns:
        return {}
    pred_cols = [c for c in df_wnb.columns if c.startswith("prediction_")]
    if not pred_cols:
        return {}

    sub = df_wnb[df_wnb["date"] >= cutoff_ts] if "date" in df_wnb.columns else df_wnb
    if sub.empty:
        return {}

    latest = (
        sub.sort_values("date").groupby([ac_col, "position"]).last()
        if "date" in sub.columns
        else sub.groupby([ac_col, "position"]).last()
    )
    # NaN-skipping max == OR across positions
    agg = latest.reset_index().groupby(ac_col)[pred_cols].max()

    wnb_alert: dict = {}
    for msn, row in agg.iterrows():
        row_max = row.max()  # NaN if all positions/cols are NaN
        if pd.isna(row_max):
            continue
        wnb_alert[str(msn)] = 1 if float(row_max) == 1 else 0
    return wnb_alert


@st.cache_data(ttl=300)
def _fuel_alerts(df_fuel: pd.DataFrame, cutoff_ts: pd.Timestamp) -> tuple[dict, dict]:
    """E2 cruise-burn engine-degradation alerts vs each tail's OWN baseline.

    Reuses the validated own-baseline method from 4_Fuel.py Section 5: per aircraft,
    total cruise burn = sum of the cruise*fuelMeterFuelBurn*Kg columns, normalized to a
    kg/h RATE via time_sec_cruise when present (else absolute kg, so route length no
    longer confounds engine health). Baseline = median of each tail's earliest
    ceil(30%)/>=5 in-window flights; recent = mean of its last 5 flights. Level 2 (red)
    when recent > baseline x 1.10, level 1 (amber) when > x 1.05, else 0. Requires
    >= 5 flights and a strictly positive baseline. Returns ({ac_sn: level},
    {ac_sn: pct_above}); empty dicts if data / cruise columns are missing.
    """
    if df_fuel is None or df_fuel.empty:
        return {}, {}
    ac_col = next((c for c in ("ac_sn", "aircraftSerNum-1") if c in df_fuel.columns), None)
    if ac_col is None or "date" not in df_fuel.columns:
        return {}, {}
    cruise_cols = [
        c for c in df_fuel.columns
        if c.startswith("cruise") and "fuelMeterFuelBurn" in c and c.endswith("Kg")
    ]
    if not cruise_cols:
        return {}, {}

    sub = df_fuel[df_fuel["date"] >= cutoff_ts].copy()
    if sub.empty:
        return {}, {}

    for c in cruise_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    cruise_total = sub[cruise_cols].sum(axis=1)

    # Normalize to kg/h when cruise duration is available; else absolute kg.
    if "time_sec_cruise" in sub.columns:
        dur_hr = pd.to_numeric(sub["time_sec_cruise"], errors="coerce") / 3600.0
        metric = (cruise_total / dur_hr).where(dur_hr > 0)
    else:
        metric = cruise_total

    sub = sub.assign(_fuel_metric=metric).dropna(subset=["_fuel_metric", ac_col, "date"])
    if sub.empty:
        return {}, {}

    fuel_alert: dict = {}
    fuel_pct: dict = {}
    for msn, grp in sub.groupby(ac_col):
        grp = grp.sort_values("date")
        if len(grp) < 5:                                   # minimum sample guard
            continue
        n_base = min(len(grp), max(5, int(math.ceil(len(grp) * 0.30))))
        baseline = float(grp["_fuel_metric"].iloc[:n_base].median())
        if baseline <= 0:                                  # need a positive reference
            continue
        recent_n = min(5, len(grp))
        recent_mean = float(grp["_fuel_metric"].iloc[-recent_n:].mean())
        pct_above = (recent_mean - baseline) / baseline * 100.0
        level = 2 if recent_mean > baseline * 1.10 else (1 if recent_mean > baseline * 1.05 else 0)
        fuel_alert[str(msn)] = level
        fuel_pct[str(msn)] = pct_above
    return fuel_alert, fuel_pct


# ── Build per-system alert snapshots ─────────────────────────────────────────

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


sav_lh_alert = _sav_transient_alert(data["sav_lh"])
sav_rh_alert = _sav_transient_alert(data["sav_rh"])

# Coverage of the transient model across both engines — surfaced below as a visible
# assertion so a coverage collapse fails loud, never as silent zero-alerts.
n_sav_cov = len(set(sav_lh_alert) | set(sav_rh_alert))

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
def _airbus_col(df: pd.DataFrame) -> str | None:
    return next((c for c in ("tail_number", "ac_sn") if c in df.columns), None)

airbus_ac_col = _airbus_col(data["a320"]) or _airbus_col(data["a330"])
airbus_alerts: dict = {}
for fleet_key, fleet_label in [("a320", "A320"), ("a330", "A330")]:
    df_ab = data[fleet_key]
    _ab_col = _airbus_col(df_ab)
    if df_ab.empty or _ab_col is None:
        continue
    airbus_ac_col = _ab_col
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

# Wheels & Brakes (ATA 32) — long-format OR aggregation across all 6 positions
wnb_alert = _wnb_alerts(data["wnb"], cutoff)

# Fuel (ATA 73/76) — cruise-burn engine-degradation vs each tail's own baseline
fuel_alert, fuel_pct = _fuel_alerts(data["fuel"], cutoff)

# SAV A320 — latest-start pre-failure prediction per engine
sav_a320_alerts: dict = {}   # {normalized tail: [eng_labels]} — any-engine pre-failure
sav_a320_known: set = set()  # normalized tails with any SAV record (for matrix "—" fallback)
for _key, _eng in [("sav_a320_e1", "Eng 1"), ("sav_a320_e2", "Eng 2")]:
    _df_sa = data.get(_key, pd.DataFrame())
    if _df_sa.empty or "sav_failure_pred" not in _df_sa.columns or "aircraft_id" not in _df_sa.columns:
        continue
    _df_sa = _df_sa[_df_sa["aircraft_id"].astype(str).str.strip() != ""]
    sav_a320_known.update(_df_sa["aircraft_id"].astype(str).str.strip())
    _lat = _df_sa.sort_values("date").groupby("aircraft_id").last()
    for tail in _lat.index[_lat["sav_failure_pred"].eq(1)]:
        sav_a320_alerts.setdefault(str(tail).strip(), []).append(_eng)

# ── Collect all aircraft ──────────────────────────────────────────────────────
all_e2 = sorted(set(
    list(sav_lh_alert.keys())
    + list(sav_rh_alert.keys())
    + list(oxy_alert.keys())
    + list(foqa_exceedances.keys())
    + list(wnb_alert.keys())
    + list(fuel_alert.keys())
))

# ── Predictive catches (commercial KPI) ───────────────────────────────────────
# Distinct-aircraft sets of critical predictive catches across the full fleet.
# Each catch is a potential unscheduled removal / AOG flagged before failure.
sav_catch_msns = {
    msn for msn in all_e2
    if sav_lh_alert.get(msn) == 1 or sav_rh_alert.get(msn) == 1
}
oxy_nodispatch_msns = {msn for msn, v in oxy_alert.items() if v == 2}
wnb_catch_msns = {msn for msn, v in wnb_alert.items() if v == 1}
# Fuel red-tier (>10% above own cruise-burn baseline) — engine-degradation catch
fuel_catch_msns = {msn for msn, v in fuel_alert.items() if v == 2}
airbus_catch = {key for key, n in airbus_alerts.items() if n > 5}
# A320 starter-valve pre-failure catches (binary sav_failure_pred == 1), keyed to
# match airbus_catch ("A320:<tail>") so a tail flagged by both FOQA and SAV counts once.
sav_a320_catch = {f"A320:{tail}" for tail in sav_a320_alerts}
# Fuel cruise-burn own-baseline is excluded from the headline: it is a heuristic with no link to a confirmed failure (SAV/W&B/A320 SAV = model predictions, oxy amber = no-dispatch CAS, Airbus FOQA = certified-limit exceedances are the asserted catches).
asserted_catches = len(sav_catch_msns | oxy_nodispatch_msns | wnb_catch_msns) + len(airbus_catch | sav_a320_catch)
fuel_advisory_n = len(fuel_catch_msns)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.title(":material/dashboard: Fleet Overview")
st.caption(f"Multi-fleet predictive maintenance · {date.today().strftime('%d-%b-%Y')} · {days_back}-day window")

n_sav_alert = len(sav_catch_msns)
n_oxy_red   = sum(1 for v in oxy_alert.values() if v == 2)
n_oxy_amber = sum(1 for v in oxy_alert.values() if v == 1)
n_foqa      = sum(1 for v in foqa_exceedances.values() if v > 0)
n_airbus    = sum(1 for v in airbus_alerts.values() if v > 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("E2 aircraft tracked", len(all_e2))
c2.metric("SAV alerts (LH+RH)", n_sav_alert,
          help="Aircraft with predicted pre-failure on starter valve (latest flight)")
c3.metric("Oxy — no dispatch", n_oxy_red,
          help=f"Latest PSI < {PSI_AMBER} — amber CAS, no departure")
c4.metric("E2 FOQA exceedances", n_foqa,
          help=f"Aircraft with ≥1 engine/aircraft exceedance in last {days_back} days")
c5.metric("Airbus FOQA alerts", n_airbus,
          help=f"A320/A330 aircraft with exceedances in last {days_back} days")
c6.metric("Unsched. removals flagged", asserted_catches,
          help="Distinct aircraft flagged with a critical predictive catch this period "
               "(E2 SAV LH/RH pre-failure, oxygen no-dispatch, W&B brake/gear removal, "
               "plus Airbus red-tier >5 exceedances and A320 starter-valve pre-failure). "
               "Each is a potential unscheduled "
               "removal / AOG caught before failure — a candidate save, not an asserted one.")

st.caption(
    "**Commercial impact** — each flagged aircraft is a potential unscheduled "
    "removal or AOG caught predictively before in-service failure this period. "
    "Specific tails are listed in the Immediate Actions banner below. "
    f"Separately, {fuel_advisory_n} aircraft are on a fuel cruise-burn advisory "
    "(>10% above their own baseline) — a monitor-tier heuristic confounded by "
    "weight/altitude/wind, not an asserted predictive catch."
)

# SAV (E2) coverage assertion — the SAV signal above is sourced from the
# ACARS-validated transient model. Make the scored-aircraft count visible, and fail
# LOUD if coverage collapses so a stalled job never reads as a clean zero-alert fleet.
st.caption(
    f":material/dataset: SAV (E2): {n_sav_cov} aircraft scored by the "
    "ACARS-validated transient model."
)
if n_sav_cov == 0:
    st.warning(
        "SAV (E2): the start-transient model scored zero aircraft — the "
        "`save_sav_transient_report` job may be stopped or its features table is "
        "empty. Starter-valve predictions are unavailable until it refreshes; the "
        "absence of SAV alerts here does NOT mean the fleet is clear."
    )

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
    if wnb_alert.get(msn) == 1:
        immediate_items.append(f"**{_dnm(msn)}** — W&B: predicted brake/gear removal (inspect landing gear ATA 32)")

for key, n in airbus_alerts.items():
    if n > 0:
        fleet, tail = key.split(":", 1)
        immediate_items.append(f"**{fleet} {tail}** — FOQA: {n} exceedance(s) in last {days_back} days")

for tail, engs in sorted(sav_a320_alerts.items()):
    immediate_items.append(
        f"**A320 {tail}** — SAV {' + '.join(engs)}: predicted pre-failure "
        "(inspect starter air valve ATA 80)"
    )

if immediate_items:
    st.error("**Immediate Actions Required**\n\n" + "\n\n".join(f"- {x}" for x in immediate_items))

# Watch list
watch_items = []
for msn in all_e2:
    if oxy_alert.get(msn) == 1:
        psi = oxy_psi.get(msn, 0)
        watch_items.append(f"**{_dnm(msn)}** — Oxygen: {psi:.0f} PSI ({PSI_AMBER}–{PSI_CYAN}) — monitor, possible OBSERVER OXY LO PRESS")
    if foqa_exceedances.get(msn, 0) > 0:
        n = foqa_exceedances[msn]
        watch_items.append(f"**{_dnm(msn)}** — FOQA: {n} exceedance event(s) this period")
    if fuel_alert.get(msn) in (1, 2):
        fuel_icon = "Critical" if fuel_alert.get(msn) == 2 else "Monitor"
        watch_items.append(f"{fuel_icon} **{_dnm(msn)}** — Fuel: +{fuel_pct.get(msn, 0):.0f}% vs own cruise-burn baseline (advisory — monitor engine degradation, ATA 73/76)")

if watch_items:
    st.warning("**Monitor Closely**\n\n" + "\n\n".join(f"- {x}" for x in watch_items))

if not immediate_items and not watch_items:
    st.success(f"No critical alerts in the last {days_back} days across all monitored systems.")

# ── Upcoming Maintenance Forecast (Tier 3 — upcoming inspections) ─────────────
st.subheader(":material/query_stats: Oxygen Servicing Forecast")
st.caption(
    f"Forward-looking oxygen servicing calendar. Projects each E2 aircraft's "
    f"days-to-amber from the median daily PSI drop (PSI/day) and the smoothed current pressure "
    f"(median of the last 5 readings) within the {days_back}-day window, and flags "
    f"those forecast to cross the {PSI_AMBER} PSI amber (no-dispatch) threshold inside "
    f"a {FORECAST_HORIZON_DAYS}-day planning horizon."
)

_oxy_df = data["oxy"]
_oxy_cols_ok = (
    oxy_ac_col is not None
    and not _oxy_df.empty
    and "psi" in _oxy_df.columns
    and "delta_press" in _oxy_df.columns
)

if not _oxy_cols_ok:
    st.info(
        "Oxygen-pressure history is unavailable (no PSI / daily-drop columns in the "
        "current window) — upcoming-maintenance forecast cannot be computed."
    )
else:
    _fc = _oxy_maintenance_forecast(_oxy_df, oxy_ac_col, cutoff)
    _fc_rows = _fc["rows"]
    if _fc_rows:
        _lines = [
            f"- **{_dnm(r['msn'])}** → est. servicing {r['est_date'].strftime('%d-%b-%Y')} "
            f"({int(r['days_to_amber'])} days remaining) · PSI path "
            f"{r['current_psi']:.0f} → {PSI_AMBER} at {r['daily_drop']:.1f} PSI/day · "
            f"plan ATA 35 oxygen cylinder servicing"
            for r in _fc_rows
        ]
        st.warning(
            f"**Plan oxygen servicing within {FORECAST_HORIZON_DAYS} days** — the "
            f"following aircraft are forecast to cross the {PSI_AMBER} PSI amber "
            f"(no-dispatch) threshold:\n" + "\n".join(_lines)
        )
    elif _fc["n_evaluated"] > 0:
        st.success(
            f"No E2 aircraft forecast to cross the {PSI_AMBER} PSI amber threshold "
            f"within the next {FORECAST_HORIZON_DAYS} days "
            f"({_fc['n_evaluated']} aircraft evaluated)."
        )
    else:
        st.info(
            f"Insufficient oxygen-pressure history in the last {days_back} days to forecast "
            f"upcoming servicing (need ≥ 4 readings per aircraft with a positive daily drop). "
            f"Expand the history window in the sidebar."
        )

st.divider()

# ── Fleet Health Matrix ───────────────────────────────────────────────────────
st.subheader(":material/grid_view: E2 Fleet — Health Matrix")
st.caption(
    "Latest status per aircraft × system. "
    "Critical = alert/action required · Monitor = monitor · Normal = normal · — = no data in period"
)

if "E2" in fleet_filter and all_e2:
    matrix_rows = []
    for msn in sorted(all_e2):
        dn = _dnm(msn)

        # SAV LH
        sav_lh_cell = "Critical" if sav_lh_alert.get(msn) == 1 else ("Normal" if msn in sav_lh_alert else "—")
        # SAV RH
        sav_rh_cell = "Critical" if sav_rh_alert.get(msn) == 1 else ("Normal" if msn in sav_rh_alert else "—")
        # Oxygen
        oa = oxy_alert.get(msn)
        oxy_cell = ("Critical" if oa == 2 else "Monitor" if oa == 1 else ("Normal" if msn in oxy_alert else "—"))
        psi_val = f"{oxy_psi[msn]:.0f}" if msn in oxy_psi else "—"
        # FOQA
        foqa_n = foqa_exceedances.get(msn, None)
        foqa_cell = ("Critical" if foqa_n and foqa_n > 0 else ("Normal" if foqa_n == 0 else "—"))
        # Wheels & Brakes (ATA 32)
        wnb_cell = "Critical" if wnb_alert.get(msn) == 1 else ("Normal" if msn in wnb_alert else "—")
        # Fuel (ATA 73/76) — cruise-burn degradation vs own baseline
        fa = fuel_alert.get(msn)
        fuel_cell = ("Critical" if fa == 2 else "Monitor" if fa == 1 else ("Normal" if msn in fuel_alert else "—"))

        # Overall worst
        cells = [sav_lh_cell, sav_rh_cell, oxy_cell, foqa_cell, wnb_cell, fuel_cell]
        if "Critical" in cells:
            worst = "Critical"
        elif "Monitor" in cells:
            worst = "Monitor"
        elif "Normal" in cells:
            worst = "Normal"
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
            "W&B": wnb_cell,
            "Fuel": f"{fuel_cell} +{fuel_pct[msn]:.0f}%" if msn in fuel_pct else fuel_cell,
        })

    if matrix_rows:
        df_matrix = pd.DataFrame(matrix_rows).sort_values(
            "Overall",
            key=lambda s: s.map({"Critical": 0, "Monitor": 1, "Normal": 2, "— No data": 3}),
        )

        def _color_matrix(row):
            o = row.get("Overall", "")
            if "Critical" in o:
                return ["background-color: rgba(239,68,68,0.12)"] * len(row)
            elif "Monitor" in o:
                return ["background-color: rgba(245,158,11,0.10)"] * len(row)
            return [""] * len(row)

        display_cols = ["Aircraft", "Overall", "SAV LH", "SAV RH", "Oxygen", "FOQA", "W&B", "Fuel"]
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
    st.subheader(f":material/warning: {fleet_label} Fleet — FOQA Exceedance Summary")

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
    _n_total_ab = len(ab_counts)
    # Full per-tail FOQA exceedance counts (before top-10 truncation) for the A320 matrix
    if fleet_label == "A320":
        _a320_foqa_counts = dict(zip(
            ab_counts["Aircraft"].astype(str).str.strip(),
            ab_counts["Exceedances"].astype(int),
        ))
    ab_counts = ab_counts.head(10)
    if _n_total_ab > 10:
        st.caption(f"Showing top 10 of {_n_total_ab} aircraft by exceedance count.")
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

    # ── A320 Health Matrix (FOQA + SAV per tail) — A320 only ──────────────────
    # A330 keeps the bar chart only (no SAV signal), avoiding a redundant re-present.
    if fleet_label == "A320":
        st.markdown("#### A320 Fleet — Health Matrix")
        st.caption(
            "Per-tail triage across the two monitored A320 systems. "
            "Critical = alert/action required · Monitor = monitor · Normal = normal · — = no data. "
            "FOQA Critical = >5 exceedance events · SAV Critical = predicted starter-valve pre-failure."
        )
        hm_rows = []
        # Anchor on the UNION of FOQA and SAV tails so a SAV-only pre-failure
        # tail (no FOQA record in the window) still surfaces as Critical.
        for _tail in sorted(set(_a320_foqa_counts) | sav_a320_known):
            _tail_s = str(_tail).strip()
            _n = _a320_foqa_counts.get(_tail_s)
            # FOQA cell — reuse the file's >5 red / >0 amber / 0 green convention
            # when a count exists; otherwise — (no false for a missing record).
            if _n is not None:
                foqa_cell = "Critical" if _n > 5 else ("Monitor" if _n > 0 else "Normal")
                foqa_display = f"{foqa_cell} ({int(_n)})"
            else:
                foqa_cell = "—"
                foqa_display = "—"
            # SAV cell — alert / known-no-alert / no record (— never a false )
            if _tail_s in sav_a320_alerts:
                sav_cell = "Critical"
            elif _tail_s in sav_a320_known:
                sav_cell = "Normal"
            else:
                sav_cell = "—"

            cells = [foqa_cell, sav_cell]
            if "Critical" in cells:
                worst = "Critical"
            elif "Monitor" in cells:
                worst = "Monitor"
            elif "Normal" in cells:
                worst = "Normal"
            else:
                worst = "— No data"

            hm_rows.append({
                "Aircraft": display_name(_tail_s, prefix_map),
                "Overall": worst,
                "FOQA": foqa_display,
                "SAV": sav_cell,
            })

        if hm_rows:
            df_hm = pd.DataFrame(hm_rows).sort_values(
                "Overall",
                key=lambda s: s.map(
                    {"Critical": 0, "Monitor": 1, "Normal": 2, "— No data": 3}
                ),
            )
            _n_hm = len(df_hm)
            df_hm = df_hm.head(10)
            if _n_hm > 10:
                st.caption(f"Showing top 10 of {_n_hm} A320 aircraft by status.")

            def _color_hm(row):
                o = row.get("Overall", "")
                if "Critical" in o:
                    return ["background-color: rgba(239,68,68,0.12)"] * len(row)
                elif "Monitor" in o:
                    return ["background-color: rgba(245,158,11,0.10)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_hm.style.apply(_color_hm, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No A320 aircraft available to build the health matrix in the selected window."
            )

# ── Pipeline health ───────────────────────────────────────────────────────────
st.divider()
st.subheader(":material/health_and_safety: Pipeline health")

_now_utc = pd.Timestamp.now(tz="UTC")
_health_rows = []
n_total = len(PIPELINE_SOURCES)
n_stale = 0
for label, df_key, filename, producing_job in PIPELINE_SOURCES:
    df_chk = data.get(df_key, pd.DataFrame())
    mtime = get_file_mtime(filename)
    if mtime is None:
        age_h = None
        status = "Unavailable"
        refresh = "—"
        n_stale += 1
    else:
        age_h = (_now_utc - mtime) / pd.Timedelta(hours=1)
        refresh = f"{int(round(age_h))}h ago"
        if age_h > 48:
            status = "Stale"
            n_stale += 1
        elif age_h > 24:
            status = "Aging"
        else:
            status = "Active"

    if not df_chk.empty and "date" in df_chk.columns:
        last_dt = df_chk["date"].max()
        last_flight = last_dt.strftime("%d-%b-%Y") if pd.notna(last_dt) else "—"
        rows = len(df_chk)
    else:
        last_flight = "—"
        rows = 0

    _health_rows.append({
        "System": label,
        "Status": status,
        "Producing job": producing_job,
        "Last refresh": refresh,   # Drive mtime age — is the job alive?
        "Last flight": last_flight,  # last event date — secondary data-coverage context
        "Rows": rows,
    })

if n_stale > 0:
    st.warning(
        f"{n_stale} of {n_total} data sources have not refreshed in 48h+ — "
        "the producing job(s) may be stopped; predictions on the affected "
        "pages may be outdated."
    )
else:
    st.success("All data sources refreshed within 48h.")

with st.expander("Data freshness — last refresh per system"):
    st.dataframe(
        pd.DataFrame(_health_rows),
        use_container_width=True,
        hide_index=True,
    )
