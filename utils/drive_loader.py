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
