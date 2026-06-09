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
    """Baixa *filename* da pasta Drive configurada e retorna DataFrame."""
    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    service = _get_service()

    results = service.files().list(
        q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
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
    return pd.read_parquet(buf)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ac_master() -> pd.DataFrame:
    """Returns ac_master.parquet with TCRF_SN and AC (registration) columns."""
    return load("ac_master.parquet")


@st.cache_data(ttl=3600, show_spinner=False)
def make_prefix_map() -> dict:
    """Returns {TCRF_SN: registration} e.g. {'20077': 'PR-PXA'}.
    Empty dict when ac_master is unavailable (prefix display degrades gracefully)."""
    df = load_ac_master()
    if df.empty or "TCRF_SN" not in df.columns or "AC" not in df.columns:
        return {}
    return dict(zip(df["TCRF_SN"].astype(str).str.strip(), df["AC"].astype(str).str.strip()))


def display_name(msn: str, prefix_map: dict) -> str:
    """'PR-PXA · 20077' if prefix available, else just '20077'."""
    prefix = prefix_map.get(str(msn).strip(), "")
    return f"{prefix} · {msn}" if prefix else str(msn)


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
        df = df[df[ac_col].astype(str).isin(valid)]
    return df.reset_index(drop=True)
