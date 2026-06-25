"""
FOQA / MOQA — Engine & Aircraft Exceedance Monitoring (ATA 05)
Monitors engine trends (ITT, N2 vibration, oil pressure) and aircraft
exceedances (hard landing, VMO, gear overspeed, APU EGT) per the E195-E2
knowledge base and PW1900G limits.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.drive_loader import load, clean_df, make_prefix_map, display_name

st.set_page_config(page_title="FOQA / MOQA", layout="wide")

# AMM limits (PW1900G / E195-E2 knowledge_base.json)
ITT_CONTINUOUS_C   = 1006
ITT_TAKEOFF_C      = 1054
N2_VIB_AMBER       = 4.00
N2_VIB_BORESCOPE   = 4.08
N2_VIB_REMOVE      = 4.31
OIL_PRESS_MIN      = 50.3
VMO_KIAS           = 330
VLE_KIAS           = 265  # gear-extended speed limit (local constant)

# Per-aircraft drill-down spec: (label, flag column, recorded-value column, AMM limit, recommended action, direction)
# limit 'HARD_LANDING' = use the per-row hard_landing_g_limit; None = no fixed AMM limit
# direction 'high' = trips by exceeding above the limit; 'low' = trips by falling below the limit
DRILLDOWN_SPECS = [
    ("ITT Takeoff LH (°C)",  'itt_lh_takeoff_exceedance', 'max_itt_lh_takeoff',      ITT_TAKEOFF_C, "review ATA 05 task", 'high'),
    ("ITT Takeoff RH (°C)",  'itt_rh_takeoff_exceedance', 'max_itt_rh_takeoff',      ITT_TAKEOFF_C, "review ATA 05 task", 'high'),
    ("ITT Continuous LH (°C)", 'itt_lh_continuous_exceedance', 'max_itt_lh_climb',   ITT_CONTINUOUS_C, "review ATA 05 continuous limit", 'high'),
    ("ITT Continuous RH (°C)", 'itt_rh_continuous_exceedance', 'max_itt_rh_climb',   ITT_CONTINUOUS_C, "review ATA 05 continuous limit", 'high'),
    ("Low Oil Press LH (psig)", 'oil_press_lh_low',        'min_oil_press_lh',        OIL_PRESS_MIN, "check oil system per AMM", 'low'),
    ("Low Oil Press RH (psig)", 'oil_press_rh_low',        'min_oil_press_rh',        OIL_PRESS_MIN, "check oil system per AMM", 'low'),
    ("N2 Vibration LH (AU)", 'n2_vib_lh_amber',           'max_n2_vib_lh',           N2_VIB_AMBER,  "borescope per AMM", 'high'),
    ("N2 Vibration RH (AU)", 'n2_vib_rh_amber',           'max_n2_vib_rh',           N2_VIB_AMBER,  "borescope per AMM", 'high'),
    ("Hard landing (g)",     'hard_landing_flag',         'max_normal_accel_landing', 'HARD_LANDING', "review ATA 05 task", 'high'),
    ("VMO (kias)",           'vmo_exceedance',            'max_cas_kias',            VMO_KIAS,      "review ATA 05 task", 'high'),
    ("VLE gear (kias)",      'vle_exceedance',            'max_cas_gear_down',       VLE_KIAS,      "review ATA 05 task", 'high'),
    ("APU EGT (°C)",         'apu_egt_exceedance',        'max_apu_egt',             None,          "MPP7166_05-50-55", 'high'),
]


@st.cache_data(ttl=300)
def compute_margin_to_limit(df_in, ac_col, param_specs):
    """Per-MSN margin to AMM limit using the median of each tail's last 5 flights
    by date (min 3 flights). Robust to single-reading sensor noise."""
    if df_in is None or df_in.empty or not ac_col or ac_col not in df_in.columns:
        return pd.DataFrame()
    if 'date' not in df_in.columns:
        return pd.DataFrame()

    rows = []
    for param, label, limit in param_specs:
        if param not in df_in.columns:
            continue
        sub = df_in.dropna(subset=[param, 'date', ac_col])
        if sub.empty:
            continue
        for msn, g in sub.groupby(ac_col):
            last5 = g.sort_values('date').tail(5)
            if len(last5) < 3:
                continue
            median_val = float(last5[param].median())
            margin = limit - median_val
            pct = (median_val / limit * 100) if limit else float('nan')
            rows.append({
                'MSN': str(msn),
                'Parameter': label,
                'Recent median (last 5)': round(median_val, 2),
                'AMM limit': limit,
                'Margin': round(margin, 2),
                '% of limit': round(pct, 1),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Margin', ascending=True).reset_index(drop=True)


@st.cache_data(ttl=300)
def drilldown_counts_by_msn(df_in, ac_col):
    """Total exceedance-flag trips per aircraft, used to pick the busiest tail by default."""
    if df_in is None or df_in.empty or not ac_col or ac_col not in df_in.columns:
        return pd.Series(dtype=int)
    flag_cols = [f for _, f, _, _, _, _ in DRILLDOWN_SPECS if f in df_in.columns]
    if not flag_cols:
        return pd.Series(dtype=int)
    trips = df_in[flag_cols].fillna(False).astype(bool).sum(axis=1)
    return trips.groupby(df_in[ac_col]).sum().astype(int)


@st.cache_data(ttl=300)
def build_drilldown_events(df_in, ac_col, msn):
    """Chronological exceedance events for one aircraft, built only from flag columns
    present in the data. Each row pairs the recorded value with its AMM limit and action."""
    if df_in is None or df_in.empty or not ac_col or ac_col not in df_in.columns:
        return pd.DataFrame()
    sub = df_in[df_in[ac_col] == msn]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for label, flag_col, value_col, limit, action, direction in DRILLDOWN_SPECS:
        if flag_col not in sub.columns:
            continue
        flagged = sub[sub[flag_col].fillna(False).astype(bool)]
        if flagged.empty:
            continue
        for _, r in flagged.iterrows():
            rec_val = r.get(value_col) if value_col in flagged.columns else None
            if limit == 'HARD_LANDING':
                amm_limit = r.get('hard_landing_g_limit') if 'hard_landing_g_limit' in flagged.columns else None
            else:
                amm_limit = limit
            limit_str = None
            if pd.notna(amm_limit):
                arrow = '▼ below' if direction == 'low' else '▲ above'
                limit_str = f"{arrow} {round(float(amm_limit), 2)}"
            rows.append({
                'Date': r.get('date'),
                'Parameter': label,
                'Limit': limit_str,
                'Recorded value': round(float(rec_val), 2) if pd.notna(rec_val) else None,
                'AMM limit': round(float(amm_limit), 2) if pd.notna(amm_limit) else None,
                'Recommended action': action,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Date', ascending=False).reset_index(drop=True)


# Two-sided 0.975 Student-t multipliers (95% CI) by degrees of freedom.
_T95_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def _t95(dof):
    """Two-sided 0.975 Student-t multiplier for the given degrees of freedom.

    Falls back to ~2.04 (the large-sample / z≈1.96-but-conservative value) for
    dof > 30, where the t distribution is essentially normal.
    """
    if dof in _T95_TABLE:
        return _T95_TABLE[dof]
    if dof < 1:
        return _T95_TABLE[1]
    return 2.04


@st.cache_data(ttl=300)
def _n2vib_forecast(df_in, ac_col):
    """Per-engine linear ECTM trend toward the certified 4.00 AU amber limit.

    For each engine side present in the data, fits an OLS line to a tail's dated N2
    vibration history (>=6 flights) and projects the amber crossing. The uncertainty
    is the trend fit's 95% confidence interval (derived from the slope standard error),
    not a daily-volatility quantile. Only coherent upward trends qualify (R^2 >= 0.30,
    positive slope, present value below the amber limit); abrupt step-changes give a
    low R^2 and are intentionally excluded — they are caught by the exceedance flags.
    """
    if df_in is None or df_in.empty or not ac_col or ac_col not in df_in.columns:
        return []
    if 'date' not in df_in.columns:
        return []

    results = []
    for side, param in [('LH', 'max_n2_vib_lh'), ('RH', 'max_n2_vib_rh')]:
        if param not in df_in.columns:
            continue
        for ac_sn, g in df_in.groupby(ac_col):
            sub = g.dropna(subset=[param, 'date']).sort_values('date')
            n = len(sub)
            if n < 6:
                continue
            x = (sub['date'] - sub['date'].min()).dt.days.astype(float).to_numpy()
            y = sub[param].astype(float).to_numpy()
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            x_mean = float(x.mean())
            x_max = float(x.max())
            Sxx = float(((x - x_mean) ** 2).sum())
            if Sxx <= 0:
                continue
            slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
            y_hat = np.polyval([slope, intercept], x)
            sse = float(((y - y_hat) ** 2).sum())
            sst = float(((y - y.mean()) ** 2).sum())
            if sst <= 0:
                continue
            r2 = 1.0 - sse / sst
            if r2 < 0.30 or slope <= 0:
                continue
            current_fit = float(np.polyval([slope, intercept], x_max))
            if current_fit >= N2_VIB_AMBER:
                continue

            residual_std = float(np.sqrt(sse / (n - 2)))
            slope_se = residual_std / np.sqrt(Sxx)
            t_mult = _t95(n - 2)
            slope_hi = slope + t_mult * slope_se            # steepest within 95% CI
            slope_lo = max(slope - t_mult * slope_se, 1e-9)  # shallowest within 95% CI

            gap = N2_VIB_AMBER - current_fit
            days_to_amber = gap / slope
            earliest_days = gap / slope_hi
            latest_days = None if slope_lo <= 1e-9 else gap / slope_lo  # CI includes zero -> not significant

            # Anchor the projection at the LAST flight (x_max) where current_fit /
            # days_to_amber are evaluated — matches the chart's amber crossing.
            last_flight_date = sub['date'].max()
            low_conf = r2 < 0.50

            results.append({
                'ac_sn': ac_sn,
                'side': side,
                'n': n,
                'r2': r2,
                'low_conf': low_conf,
                'current_fit': current_fit,
                'slope_per_month': slope * 30.0,
                'days_to_amber': days_to_amber,
                't_mult': t_mult,
                'earliest_date': last_flight_date + pd.Timedelta(days=earliest_days),
                'expected_date': last_flight_date + pd.Timedelta(days=days_to_amber),
                'latest_date': None if latest_days is None else last_flight_date + pd.Timedelta(days=latest_days),
                # fields needed to redraw the fit + CI band on the per-engine chart
                'slope': slope,
                'intercept': intercept,
                'residual_std': residual_std,
                'Sxx': Sxx,
                'x_mean': x_mean,
                'x_min_date': sub['date'].min(),
            })
    return results


st.title(":material/monitoring: FOQA / MOQA — ATA 05 Exceedance & Engine Trends")
st.caption("Engine: ITT, N2 vibration, oil pressure · Aircraft: hard landing, VMO, gear overspeed")

df = load("e2_foqa_report.parquet")

if df.empty:
    st.info("No FOQA data available yet. The `e2_foqa_moqa_job` pipeline has not processed any files.")
    st.stop()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

if 'date' in df.columns and df['date'].notna().any():
    latest_flight = df['date'].max()
    age_days = (date.today() - latest_flight.date()).days
    if age_days <= 3:
        st.success(f"Latest flight ingested {latest_flight.date():%d-%b-%Y} ({age_days} day(s) ago)")
    else:
        st.warning(f"No new flights ingested in {age_days} days — pipeline may be stale (latest {latest_flight.date():%d-%b-%Y})")
else:
    st.info("No dated flights available to assess data freshness.")

ac_col = 'ac_sn' if 'ac_sn' in df.columns else None

# Filter future dates and invalid serials
_foqa_prefix_map = make_prefix_map()
df = clean_df(df, date_col='date', ac_col=ac_col, prefix_map=_foqa_prefix_map)

# Unfiltered snapshot — safety triage must not inherit cosmetic sidebar filters
df_full = df.copy()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/insights: Filters")
    if ac_col and df[ac_col].notna().any():
        all_ac = sorted(df[ac_col].dropna().unique())
        sel_ac = st.multiselect("Aircraft (MSN)", all_ac, default=all_ac)
        df = df[df[ac_col].isin(sel_ac)]

    if 'date' in df.columns and df['date'].notna().any():
        min_d = df['date'].min().date()
        max_d = df['date'].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d))
        if len(date_range) == 2:
            df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

# ── Fleet KPIs ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

n_exc   = int(df.get('any_exceedance', pd.Series()).fillna(False).sum()) if 'any_exceedance' in df.columns else 0
n_hard  = int(df.get('hard_landing_flag', pd.Series()).fillna(False).sum()) if 'hard_landing_flag' in df.columns else 0
n_vib   = int((df.get('n2_vib_lh_amber', pd.Series(False)) | df.get('n2_vib_rh_amber', pd.Series(False))).fillna(False).sum())
n_itt   = int((df.get('itt_lh_takeoff_exceedance', pd.Series(False)) | df.get('itt_rh_takeoff_exceedance', pd.Series(False))).fillna(False).sum())
n_vmo   = int(df.get('vmo_exceedance', pd.Series(False)).fillna(False).sum())

c1.metric("Flights w/ exceedance", n_exc)
c2.metric("Hard landings", n_hard)
c3.metric("N2 vib amber+", n_vib)
c4.metric("ITT exceedances", n_itt)
c5.metric("VMO events", n_vmo)

st.divider()

# ── Fleet Exceedance Heatmap ──────────────────────────────────────────────────
st.subheader(":material/grid_view: Fleet Exceedance Heatmap")
st.caption("Count of flights per aircraft (MSN) that tripped each exceedance flag in the filtered period.")

if not ac_col or df[ac_col].notna().sum() == 0:
    st.info("No aircraft (MSN) identifier available to build the heatmap.")
else:
    param_specs = [
        ("ITT T/O", lambda d: d.get('itt_lh_takeoff_exceedance', pd.Series(False, index=d.index)).fillna(False)
                              | d.get('itt_rh_takeoff_exceedance', pd.Series(False, index=d.index)).fillna(False)),
        ("ITT Cont.", lambda d: d.get('itt_lh_continuous_exceedance', pd.Series(False, index=d.index)).fillna(False)
                                | d.get('itt_rh_continuous_exceedance', pd.Series(False, index=d.index)).fillna(False)),
        ("Low oil press", lambda d: d.get('oil_press_lh_low', pd.Series(False, index=d.index)).fillna(False)
                                    | d.get('oil_press_rh_low', pd.Series(False, index=d.index)).fillna(False)),
        ("N2 vib amber", lambda d: d.get('n2_vib_lh_amber', pd.Series(False, index=d.index)).fillna(False)
                                   | d.get('n2_vib_rh_amber', pd.Series(False, index=d.index)).fillna(False)),
        ("Hard landing", lambda d: d.get('hard_landing_flag', pd.Series(False, index=d.index)).fillna(False)),
        ("VMO", lambda d: d.get('vmo_exceedance', pd.Series(False, index=d.index)).fillna(False)),
        ("VLE gear", lambda d: d.get('vle_exceedance', pd.Series(False, index=d.index)).fillna(False)),
        ("APU EGT", lambda d: d.get('apu_egt_exceedance', pd.Series(False, index=d.index)).fillna(False)),
    ]

    present_flag_cols = {
        'itt_lh_takeoff_exceedance', 'itt_rh_takeoff_exceedance',
        'itt_lh_continuous_exceedance', 'itt_rh_continuous_exceedance',
        'oil_press_lh_low', 'oil_press_rh_low',
        'n2_vib_lh_amber', 'n2_vib_rh_amber',
        'hard_landing_flag', 'vmo_exceedance', 'vle_exceedance', 'apu_egt_exceedance',
    } & set(df.columns)

    if not present_flag_cols:
        st.success("No exceedance flag columns available to map.")
    else:
        active_params = []
        for label, fn in param_specs:
            flags = fn(df).astype(bool)
            if flags.any():
                active_params.append((label, flags))

        if not active_params:
            st.success("No exceedances recorded for any aircraft in the selected period.")
        else:
            msns = sorted(df[ac_col].dropna().unique())
            param_labels = [lbl for lbl, _ in active_params]
            z = []
            for msn in msns:
                msn_mask = df[ac_col] == msn
                z.append([int((flags & msn_mask).sum()) for _, flags in active_params])

            if sum(sum(row) for row in z) == 0:
                st.success("No exceedances recorded for any aircraft in the selected period.")
            else:
                fig_hm = go.Figure(go.Heatmap(
                    z=z,
                    x=param_labels,
                    y=[str(m) for m in msns],
                    colorscale=[[0.0, '#ffffff'], [0.5, '#ffbf00'], [1.0, '#d62728']],
                    hovertemplate="%{y} — %{x}: %{z} flights<extra></extra>",
                    colorbar=dict(title="Flights"),
                ))
                fig_hm.update_layout(
                    height=max(280, 40 * len(msns) + 120),
                    xaxis_title="Exceedance parameter",
                    yaxis_title="Aircraft (MSN)",
                )
                st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ── Per-Aircraft Exceedance Drill-Down ─────────────────────────────────────────
st.subheader(":material/search: Per-Aircraft Exceedance Drill-Down")
st.caption("Pick a tail (defaults to the one with the most exceedances) to see its chronological "
           "events, each paired with the AMM limit and a recommended task.")

if not ac_col or df[ac_col].notna().sum() == 0:
    st.info("No aircraft (MSN) identifier available to drill down.")
else:
    msn_options = sorted(df[ac_col].dropna().unique())
    counts = drilldown_counts_by_msn(df, ac_col)
    default_idx = 0
    if not counts.empty and counts.max() > 0:
        top_msn = counts.idxmax()
        if top_msn in msn_options:
            default_idx = msn_options.index(top_msn)

    sel_msn = st.selectbox("Aircraft (MSN)", msn_options, index=default_idx, format_func=lambda m: str(m))
    events = build_drilldown_events(df, ac_col, sel_msn)

    if events.empty:
        st.success(f"No exceedances recorded for aircraft {sel_msn} in the selected period.")
    else:
        most_recent = pd.to_datetime(events['Date'], errors='coerce').max()
        m1, m2 = st.columns(2)
        m1.metric("Exceedance events", len(events))
        m2.metric("Most recent event", f"{most_recent:%d-%b-%Y}" if pd.notna(most_recent) else "—")

        disp = events.copy()
        disp['Date'] = pd.to_datetime(disp['Date'], errors='coerce').dt.strftime('%d-%b-%Y')
        st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()

tab_eng, tab_acft, tab_apu, tab_events = st.tabs([":material/trending_up: Engine Trends", ":material/warning: Aircraft Exceedances", ":material/insights: APU", ":material/insights: Event Log"])


# ── ENGINE TRENDS ─────────────────────────────────────────────────────────────
with tab_eng:
    st.subheader(":material/settings: Margin to AMM Limit — Engine Triage (fleet-wide)")
    st.caption("Closest-to-limit first. Uses the median of each tail's last 5 flights (min 3) — "
               "robust to sensor noise — and ignores the sidebar filters so no engine is hidden.")

    margin_specs = [
        ('max_itt_lh_takeoff', 'ITT Takeoff LH (°C)', ITT_TAKEOFF_C),
        ('max_itt_rh_takeoff', 'ITT Takeoff RH (°C)', ITT_TAKEOFF_C),
        ('max_n2_vib_lh',      'N2 Vibration LH (AU)', N2_VIB_AMBER),
        ('max_n2_vib_rh',      'N2 Vibration RH (AU)', N2_VIB_AMBER),
    ]

    df_margin = compute_margin_to_limit(df_full, ac_col, margin_specs)

    if df_margin.empty:
        st.info("Not enough data to compute margins yet — each engine needs at least 3 dated flights.")
    else:
        st.dataframe(df_margin, use_container_width=True, hide_index=True)

        # Alert: N2 at/above amber (pct >= 100) or ITT within 2% of T/O limit (pct >= 98)
        is_itt = df_margin['Parameter'].str.startswith('ITT')
        alert_mask = (is_itt & (df_margin['% of limit'] >= 98)) | (~is_itt & (df_margin['% of limit'] >= 100))
        flagged = df_margin[alert_mask]

        if not flagged.empty:
            items = "; ".join(
                f"{r['MSN']} — {r['Parameter']} (median {r['Recent median (last 5)']}, "
                f"{r['% of limit']}% of limit)"
                for _, r in flagged.iterrows()
            )
            st.warning(f"Engines at or near an AMM limit: {items}")
        else:
            st.success("All engines have a healthy margin to their AMM limit.")

    st.divider()

    st.subheader(":material/device_thermostat: ITT — Inter-Turbine Temperature (°C)")
    col_l, col_r = st.columns(2)

    for col, side, param in [
        (col_l, "LH Engine", 'max_itt_lh_takeoff'),
        (col_r, "RH Engine", 'max_itt_rh_takeoff'),
    ]:
        with col:
            if param in df.columns and df[param].notna().any():
                fig = px.scatter(
                    df.dropna(subset=['date', param]).sort_values('date'),
                    x='date', y=param,
                    color=ac_col if ac_col else None,
                    title=f"ITT Takeoff Max — {side}",
                    labels={param: "ITT (°C)", 'date': ''},
                )
                fig.add_hline(y=ITT_TAKEOFF_C, line_dash='dash', line_color='red',
                              annotation_text=f"T/O limit {ITT_TAKEOFF_C}°C", annotation_position="top right")
                fig.add_hline(y=ITT_CONTINUOUS_C, line_dash='dot', line_color='orange',
                              annotation_text=f"Continuous {ITT_CONTINUOUS_C}°C", annotation_position="bottom right")
                fig.update_layout(height=300, xaxis=dict(tickformat="%d-%b-%y"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"ITT {side} — no data available.")

    st.subheader(":material/vibration: N2 Vibration (AU)")
    col_l2, col_r2 = st.columns(2)
    for col, side, param in [
        (col_l2, "LH Engine", 'max_n2_vib_lh'),
        (col_r2, "RH Engine", 'max_n2_vib_rh'),
    ]:
        with col:
            if param in df.columns and df[param].notna().any():
                fig = px.scatter(
                    df.dropna(subset=['date', param]).sort_values('date'),
                    x='date', y=param,
                    color=ac_col if ac_col else None,
                    title=f"N2 Vib Max — {side}",
                    labels={param: "Amplitude Units (AU)", 'date': ''},
                )
                fig.add_hline(y=N2_VIB_AMBER,     line_color='orange', line_dash='dash', annotation_text="Amber 4.00 AU")
                fig.add_hline(y=N2_VIB_BORESCOPE, line_color='red',    line_dash='dash', annotation_text="Borescope 4.08 AU")
                fig.add_hline(y=N2_VIB_REMOVE,    line_color='darkred', line_dash='solid', annotation_text="Remove 4.31 AU")
                fig.update_layout(height=300, xaxis=dict(tickformat="%d-%b-%y"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"N2 Vib {side} — no data available.")

    # ── N2 Vibration Degradation Forecast ──────────────────────────────────────
    st.subheader(":material/query_stats: N2 Vibration Degradation Forecast")
    st.caption(
        "Advisory engine-condition trend (ECTM): a linear fit of each engine's N2 "
        "vibration history projected toward the certified **4.00 AU amber limit**. The "
        "shaded band is the **95% confidence interval of the trend fit** (it widens on "
        "extrapolation) — not a daily-drop quantile and not a per-flight alert. Only "
        "engines with **≥6 dated flights** and a coherent upward fit (**R² ≥ 0.30**, "
        "positive slope) are shown; abrupt vibration jumps (FOD / blade events) are "
        "caught by the exceedance flags above, not by this trend."
    )

    forecasts = _n2vib_forecast(df_full, ac_col)

    if not forecasts:
        st.success("No engine shows a significant upward N2 vibration trend — all fits are flat or improving.")
    else:
        forecasts = sorted(forecasts, key=lambda d: d['days_to_amber'])

        # (1) Fleet comparative ranking — most urgent first
        rank_rows = [{
            'Aircraft': display_name(f['ac_sn'], _foqa_prefix_map),
            'Engine': f['side'],
            'Flights': f['n'],
            'Trend (AU/month)': f"{f['slope_per_month']:+.3f}",
            'Fit R²': f"{f['r2']:.2f}",
            'Current N2 vib (AU)': f"{f['current_fit']:.2f}",
            'Expected amber-cross': f"{f['expected_date']:%d-%b-%Y}",
            'Confidence': 'Low' if f['low_conf'] else 'OK',
        } for f in forecasts[:12]]
        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

        # (2) Per-engine forecast chart
        def _fmt_engine(i):
            f = forecasts[i]
            return f"{display_name(f['ac_sn'], _foqa_prefix_map)} — {f['side']}"

        sel_idx = st.selectbox(
            "Engine to forecast",
            list(range(len(forecasts))),
            index=0,  # most urgent (smallest days_to_amber)
            format_func=_fmt_engine,
        )
        f = forecasts[sel_idx]
        param = 'max_n2_vib_lh' if f['side'] == 'LH' else 'max_n2_vib_rh'

        sub = (df_full[df_full[ac_col] == f['ac_sn']]
               .dropna(subset=[param, 'date']).sort_values('date'))
        x0 = f['x_min_date']
        x_days = (sub['date'] - x0).dt.days.astype(float).to_numpy()

        slope, intercept = f['slope'], f['intercept']
        residual_std, n_pts, Sxx, x_mean = f['residual_std'], f['n'], f['Sxx'], f['x_mean']

        # Extend the fit from the first flight to the expected amber-cross (or +180d, sooner)
        last_day = float(x_days.max())
        cross_day = last_day + f['days_to_amber']
        end_day = max(min(cross_day, last_day + 180.0), last_day + 1.0)
        x_line = np.linspace(float(x_days.min()), end_day, 80)
        y_line = slope * x_line + intercept
        band = f['t_mult'] * residual_std * np.sqrt(1.0 / n_pts + (x_line - x_mean) ** 2 / Sxx)
        line_dates = [x0 + pd.Timedelta(days=float(d)) for d in x_line]

        fig_fc = go.Figure()
        # confidence band (upper first, then lower fills up to it)
        fig_fc.add_trace(go.Scatter(
            x=line_dates, y=y_line + band, mode='lines',
            line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))
        fig_fc.add_trace(go.Scatter(
            x=line_dates, y=y_line - band, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(255,191,0,0.18)',
            name='95% CI', hoverinfo='skip',
        ))
        # fitted trend line
        fig_fc.add_trace(go.Scatter(
            x=line_dates, y=y_line, mode='lines',
            line=dict(color='#ff7f0e', width=2), name='Trend fit',
        ))
        # raw points
        fig_fc.add_trace(go.Scatter(
            x=sub['date'], y=sub[param].astype(float), mode='markers',
            marker=dict(color='gray', size=6), name='N2 vib',
        ))
        fig_fc.add_hline(y=N2_VIB_AMBER, line_color='orange', line_dash='dash',
                         annotation_text='4.00 AU amber')
        fig_fc.add_hline(y=N2_VIB_BORESCOPE, line_color='red', line_dash='dash',
                         annotation_text='4.08 AU borescope')
        fig_fc.add_hline(y=N2_VIB_REMOVE, line_color='darkred', line_dash='solid',
                         annotation_text='4.31 AU remove')
        fig_fc.update_layout(
            height=340, xaxis=dict(tickformat='%d-%b-%y'),
            yaxis_title='Amplitude Units (AU)',
            title=f"{_fmt_engine(sel_idx)} — N2 vibration forecast",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Advisory message
        disp = display_name(f['ac_sn'], _foqa_prefix_map)
        earliest_s = f"{f['earliest_date']:%d-%b-%Y}" if f['earliest_date'] is not None else "—"
        latest_s = f"{f['latest_date']:%d-%b-%Y}" if f['latest_date'] is not None else "trend not significant"
        msg = (
            f"MSN {disp} {f['side']} engine N2 vibration trending toward the 4.00 AU amber "
            f"limit — expected crossing {f['expected_date']:%d-%b-%Y} "
            f"(95% CI {earliest_s} to {latest_s}), R²={f['r2']:.2f} over {f['n']} flights. "
            f"Advisory trend; schedule a borescope per AMM before the amber crossing."
        )
        if f['low_conf']:
            msg += " — low-confidence fit (R²<0.50), treat as indicative."
        if f['days_to_amber'] <= 180:
            st.warning(f"{msg}")
        else:
            st.info(msg)

    st.subheader(":material/water_drop: Oil Pressure — Min in Flight (psig)")
    oil_cols = [c for c in ('min_oil_press_lh', 'min_oil_press_rh') if c in df.columns and df[c].notna().any()]
    if oil_cols and 'date' in df.columns:
        fig = px.line(
            df.dropna(subset=['date']).sort_values('date').melt(id_vars=['date', ac_col] if ac_col else ['date'],
                                                                  value_vars=oil_cols,
                                                                  var_name='engine', value_name='oil_press'),
            x='date', y='oil_press', color='engine',
            title="Min Oil Pressure per Flight",
            labels={'oil_press': 'psig', 'date': ''},
        )
        fig.add_hline(y=OIL_PRESS_MIN, line_dash='dash', line_color='red',
                      annotation_text=f"Min idle {OIL_PRESS_MIN} psig")
        fig.update_layout(height=280, xaxis=dict(tickformat="%d-%b-%y"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Oil pressure data not yet available.")


# ── AIRCRAFT EXCEEDANCES ───────────────────────────────────────────────────────
with tab_acft:
    st.subheader(":material/flight_land: Hard Landing Events")

    if 'hard_landing_flag' in df.columns and 'max_normal_accel_landing' in df.columns:
        df_hl = df[df['hard_landing_flag'] == True].copy() if df['hard_landing_flag'].any() else pd.DataFrame()
        if not df_hl.empty:
            cols_show = [c for c in ['date', ac_col, 'max_normal_accel_landing', 'hard_landing_g_limit', 'gross_weight_landing_kg'] if c and c in df_hl.columns]
            st.dataframe(df_hl[cols_show].sort_values('date', ascending=False), use_container_width=True)
        else:
            st.success("No hard landings detected in the selected period.")

        if 'date' in df.columns and df['max_normal_accel_landing'].notna().any():
            fig = px.scatter(
                df.dropna(subset=['date', 'max_normal_accel_landing']).sort_values('date'),
                x='date', y='max_normal_accel_landing',
                color=ac_col if ac_col else None,
                color_discrete_map={} ,
                title="Normal Acceleration at Landing (g)",
                labels={'max_normal_accel_landing': 'g-load', 'date': ''},
            )
            if 'hard_landing_g_limit' in df.columns:
                avg_limit = df['hard_landing_g_limit'].mean()
                fig.add_hline(y=avg_limit, line_dash='dash', line_color='red',
                              annotation_text=f"~{avg_limit:.2f} g limit (weight-dep.)")
            fig.update_layout(height=300, xaxis=dict(tickformat="%d-%b-%y"))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(":material/speed: VMO / MMO Exceedances")
    if 'vmo_exceedance' in df.columns and 'max_cas_kias' in df.columns:
        df_vmo = df[df['vmo_exceedance'] == True] if df['vmo_exceedance'].any() else pd.DataFrame()
        if not df_vmo.empty:
            cols_show = [c for c in ['date', ac_col, 'max_cas_kias', 'max_mach'] if c and c in df_vmo.columns]
            st.dataframe(df_vmo[cols_show].sort_values('date', ascending=False), use_container_width=True)
        else:
            st.success("No VMO/MMO events detected in the selected period.")

    st.subheader(":material/speed: Gear Overspeed")
    if 'vle_exceedance' in df.columns and df['vle_exceedance'].any():
        df_vle = df[df['vle_exceedance'] == True]
        cols_show = [c for c in ['date', ac_col, 'max_cas_gear_down'] if c and c in df_vle.columns]
        st.dataframe(df_vle[cols_show].sort_values('date', ascending=False), use_container_width=True)
    else:
        st.success("No gear overspeed (VLE) events detected.")


# ── APU ────────────────────────────────────────────────────────────────────────
with tab_apu:
    st.subheader(":material/trending_up: APU EGT Trend")
    if 'max_apu_egt' in df.columns and df['max_apu_egt'].notna().any():
        fig = px.scatter(
            df.dropna(subset=['date', 'max_apu_egt']).sort_values('date'),
            x='date', y='max_apu_egt',
            color=ac_col if ac_col else None,
            title="APU EGT Max per Flight",
            labels={'max_apu_egt': '°C', 'date': ''},
        )
        fig.update_layout(height=320, xaxis=dict(tickformat="%d-%b-%y"))
        st.plotly_chart(fig, use_container_width=True)

        apu_exc = df[df.get('apu_egt_exceedance', pd.Series(False)).fillna(False) == True] if 'apu_egt_exceedance' in df.columns else pd.DataFrame()
        if not apu_exc.empty:
            st.warning(f"{len(apu_exc)} APU EGT exceedance event(s) — check AMM task MPP7166_05-50-55")
            cols_show = [c for c in ['date', ac_col, 'max_apu_egt'] if c and c in apu_exc.columns]
            st.dataframe(apu_exc[cols_show], use_container_width=True)
        else:
            st.success("No APU EGT exceedances detected.")
    else:
        st.info("APU EGT data not yet available.")


# ── EVENT LOG ─────────────────────────────────────────────────────────────────
with tab_events:
    st.subheader(":material/warning: Full Exceedance Log")
    if 'any_exceedance' in df.columns:
        df_exc = df[df['any_exceedance'] == True].copy() if df['any_exceedance'].any() else pd.DataFrame()
        if not df_exc.empty:
            if 'exceedance_types' in df_exc.columns:
                df_exc['exceedance_types'] = df_exc['exceedance_types'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
            cols = [c for c in ['date', ac_col, 'exceedance_types', 'max_itt_lh', 'max_n2_vib_lh', 'hard_landing_flag', 'vmo_exceedance'] if c and c in df_exc.columns]
            st.dataframe(df_exc[cols].sort_values('date', ascending=False), use_container_width=True)
            st.download_button("Export CSV", df_exc.to_csv(index=False), f"foqa_exceedances_{date.today()}.csv", "text/csv")
        else:
            st.success("No exceedances found in the selected period.")
    else:
        st.info("Exceedance data not yet available.")
