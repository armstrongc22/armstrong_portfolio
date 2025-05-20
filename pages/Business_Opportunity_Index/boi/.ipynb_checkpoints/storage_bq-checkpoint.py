# boi/storage_bq.py
from google.cloud import bigquery, bigquery_storage_v1
import pandas as pd
import boi.config as cfg
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file(
    r"C:\Users\Armstrong\boi\mindful-vial-460001-h6-fd8675f1598b.json"
)
client = bigquery.Client(project=cfg.PROJECT, credentials=creds)

client   = bigquery.Client(project=cfg.PROJECT)
bqreader = bigquery_storage_v1.BigQueryReadClient()

def table_id(name: str) -> str:
    """Return fully-qualified BigQuery table id."""
    return f"{cfg.PROJECT}.{cfg.DATASET}.{name}"

def read_sql(sql: str) -> pd.DataFrame:
    """Run SQL and return dataframe (fast BQ Storage API)."""
    return client.query(sql).to_dataframe(bqstorage_client=bqreader)

def write_df(df: pd.DataFrame, name: str, mode: str = "append") -> None:
    """Upload dataframe to BigQuery table.

    mode = "append"  → WRITE_APPEND  
    mode = "replace" → WRITE_TRUNCATE
    """
    job_cfg = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if mode == "replace"
            else bigquery.WriteDisposition.WRITE_APPEND
        )
    )
    client.load_table_from_dataframe(df, table_id(name), job_config=job_cfg).result()
