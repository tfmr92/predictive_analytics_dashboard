"""
Download de arquivos Parquet do Google Drive para o Streamlit.
Credenciais lidas de st.secrets["google_service_account"].
"""

import io

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=_SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=3600, show_spinner="Atualizando dados...")
def load(filename: str) -> pd.DataFrame:
    """Baixa *filename* da pasta Drive configurada e retorna DataFrame.
    Quando há nomes duplicados, usa o modificado mais recentemente."""
    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    service = _get_service()

    results = service.files().list(
        q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute()

    files = results.get("files", [])
    if not files:
        st.warning(f"Arquivo **{filename}** não encontrado no Drive.")
        return pd.DataFrame()

    request = service.files().get_media(fileId=files[0]["id"])
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buf.seek(0)
    try:
        return pd.read_parquet(buf)
    except Exception as exc:
        st.warning(f"Falha ao ler **{filename}**: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ac_master() -> pd.DataFrame:
    """Returns ac_master.parquet with TCRF_SN and AC (registration) columns."""
    return load("ac_master.parquet")


@st.cache_data(ttl=3600, show_spinner=False)
def make_prefix_map() -> dict:
    """Returns {TCRF_SN: registration} e.g. {'20077': 'PR-PXA'}.
    Each serial is registered under every format the report parquets use:
    full TRAX SN ('19020077'), 5-digit TCRF suffix ('20077') and the
    TCRF archive form ('06120077' = '061' + suffix). Empty dict when
    ac_master is unavailable (prefix display degrades gracefully)."""
    df = load_ac_master()
    if df.empty or "TCRF_SN" not in df.columns or "AC" not in df.columns:
        return {}
    prefix_map: dict = {}
    for sn, ac in zip(df["TCRF_SN"].astype(str).str.strip(), df["AC"].astype(str).str.strip()):
        prefix_map[sn] = ac
        if sn.isdigit() and len(sn) >= 5:
            suffix = sn[-5:]
            prefix_map.setdefault(suffix, ac)
            prefix_map.setdefault("061" + suffix, ac)
    return prefix_map


def display_name(msn: str, prefix_map: dict) -> str:
    """'PR-PXA · 20077' if prefix available, else just '20077'."""
    key = str(msn).strip()
    if key.endswith(".0"):
        key = key[:-2]
    prefix = prefix_map.get(key, "")
    return f"{prefix} · {msn}" if prefix else str(msn)


@st.cache_data(ttl=900, show_spinner=False)
def get_file_mtime(filename: str) -> pd.Timestamp | None:
    """Most recent Drive modifiedTime of *filename* as a UTC Timestamp, or None.

    Used to badge how fresh a report parquet is — i.e. whether the producing
    Dagster job is still running. Returns None on any failure or when the file
    is absent, so callers can show an honest 'unavailable' state."""
    try:
        folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
        service = _get_service()
        results = service.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields="files(id,name,modifiedTime)",
            orderBy="modifiedTime desc",
        ).execute()
        files = results.get("files", [])
        if not files:
            return None
        return pd.to_datetime(files[0]["modifiedTime"], utc=True)
    except Exception:
        return None


def render_freshness_badge(
    filename: str,
    *,
    fresh_hours: float = 6,
    stale_hours: float = 48,
    label: str | None = None,
) -> float | None:
    """Render a Streamlit freshness badge for *filename* from its Drive
    modifiedTime and return the report's age in hours (None when unavailable).

    The badge keys off refresh time, not flight dates: the producing job runs on
    a fixed schedule regardless of new flights, so a stale mtime distinguishes
    'job stopped' from 'fleet idle'."""
    mtime = get_file_mtime(filename)
    name = label or filename
    if mtime is None:
        st.caption("Source refresh time unavailable.")
        return None
    age_h = (pd.Timestamp.now(tz="UTC") - mtime) / pd.Timedelta(hours=1)
    if age_h <= fresh_hours:
        st.success(f"{name} refreshed {age_h:.0f}h ago")
    elif age_h <= stale_hours:
        st.caption(f"{name} refreshed {age_h:.0f}h ago")
    else:
        st.warning(
            f"{name} not refreshed in {age_h:.0f}h — the producing job may be "
            "stopped; predictions below may be outdated"
        )
    return age_h


def render_freshest_badge(
    filenames: list,
    *,
    label: str | None = None,
    fresh_hours: float = 6,
    stale_hours: float = 48,
) -> float | None:
    """Render a single freshness badge for the most-stale report among *filenames*.

    A page may be fed by several report parquets (one per fleet/engine). The badge
    should reflect the OLDEST source, so a stale state precedes any prediction even
    if only one feeding job has stopped. Single-source pages pass a 1-element list.

    Delegates to render_freshness_badge for the file with the oldest mtime. When no
    file has a known mtime, badges the first filename (emits the 'unavailable' state)."""
    dated = [(f, get_file_mtime(f)) for f in filenames]
    dated = [(f, m) for f, m in dated if m is not None]
    if not dated:
        return render_freshness_badge(filenames[0], label=label)
    oldest = min(dated, key=lambda pair: pair[1])[0]
    return render_freshness_badge(
        oldest, fresh_hours=fresh_hours, stale_hours=stale_hours, label=label
    )


def clean_df(
    df: pd.DataFrame,
    date_col: str = "date",
    ac_col: str | None = None,
    prefix_map: dict | None = None,
) -> pd.DataFrame:
    """Remove serials absent from ac_master (only when prefix_map is non-empty).
    Date filtering is intentionally omitted: the deployed parquets have a systematic
    day/month format inversion in the Dagster ops that places all dates in December 2026.
    Filtering by date would empty the entire DataFrame. Individual pages use the
    sidebar 'Days of history' slider to control the visible window.
    """
    if df.empty:
        return df
    if ac_col and ac_col in df.columns and prefix_map:
        valid = set(prefix_map.keys())
        # Normalize float-typed serials ('20018.0' → '20018') before matching
        serials = df[ac_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        df = df[serials.isin(valid)]
    return df.reset_index(drop=True)
