"""
Central BigQuery helper – ALWAYS uses the same service-account file.
"""
from google.cloud import bigquery, bigquery_storage_v1
from google.oauth2 import service_account
import pandas as pd, boi.config as cfg
from pathlib import Path

# ── 1. Path to your key (pull from config or hard-code here) ────────────
SERVICE_ACCOUNT_FILE = Path(
    r"C:\Users\Armstrong\boi\mindful-vial-460001-h6-fd8675f1598b.json"
)
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE
)
# ────────────────────────────────────────────────────────────────────────

client   = bigquery.Client(project=cfg.PROJECT, credentials=creds)
bqreader = bigquery_storage_v1.BigQueryReadClient(credentials=creds)

def table_id(name: str) -> str:
    return f"{cfg.PROJECT}.{cfg.DATASET}.{name}"

def read_sql(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe(bqstorage_client=bqreader)

def write_df(df: pd.DataFrame, name: str, mode: str = "append"):
    job_cfg = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if mode == "replace"
            else bigquery.WriteDisposition.WRITE_APPEND
        )
    )
    client.load_table_from_dataframe(df, table_id(name), job_config=job_cfg).result()
