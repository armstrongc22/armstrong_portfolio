import os
import re
import json
import random
import io
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import prince
import plotly.express as px
from confluent_kafka import Consumer, TopicPartition
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.cluster import KMeans
import streamlit as st

# ── Directories for sampled and chunked data ─────────────────────────────────
BASELINE_DIR = Path.cwd() / "baseline_samples"
CHUNKS_DIR = Path.cwd() / "data_chunks"
BASELINE_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)

# ── Kafka configurations ────────────────────────────────────────────────────
SAMPLE_KAFKA_CONF = {
    'bootstrap.servers':  'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'security.protocol':  'SASL_SSL',
    'sasl.mechanisms':    'PLAIN',
    'sasl.username':      'NYHPZWSGVV6RI3WW',  # Sample API Key
    'sasl.password':      'Su6PTf+ty5ixCWaj6QIHnKMwEKhUPJL8Pcxmdp1A2KfQNRrb9XV1Fp+KxF3MBAA2',  # Sample API Secret
    'group.id':           'streamlit-sample-group',
    'auto.offset.reset':  'earliest',
    'enable.auto.commit': False
}

LIVE_KAFKA_CONF = {
    'bootstrap.servers':  'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'security.protocol':  'SASL_SSL',
    'sasl.mechanisms':    'PLAIN',
    'sasl.username':      'H7C5SHD4EUVIGHTZ',  # Live API Key
    'sasl.password':      'oyA7H99XPrK6c6I/aA3yB5fGhAlcp055Hr9ZdPrcqm5qlPRdsshfzS/Ku4xCHD8z',  # Live API Secret
    'group.id':           'streamlit-live-group',
    'auto.offset.reset':  'earliest',
    'enable.auto.commit': False
}

# Live Kafka consumer for streaming data
live_consumer = Consumer(LIVE_KAFKA_CONF)
live_consumer.subscribe(["watch_live_topic"])

# ── BigQuery client setup ─────────────────────────────────────────────────
info = st.secrets["bigquery"]["SERVICE_ACCOUNT_FILE"]
credentials = service_account.Credentials.from_service_account_info(info)
PROJECT = info["project_id"]
DATASET = "euphoria"
client = bigquery.Client(project=PROJECT, credentials=credentials)

# ── Sampling helper: sample Kafka topic to ~target_mb CSV ──────────────────
@st.cache_data
def sample_topic_to_size(topic: str, target_mb: int = 99, max_partitions: int = 5) -> Path:
    target_bytes = target_mb * 1024**2
    consumer = Consumer(SAMPLE_KAFKA_CONF)

    md = consumer.list_topics(topic, timeout=10.0)
    partitions = list(md.topics[topic].partitions.keys())[:max_partitions]
    random.shuffle(partitions)

    records = []
    bytes_accum = 0
    for partition in partitions:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, partition))
        if high <= low:
            continue
        start = random.randint(low, high - 1)
        tp = TopicPartition(topic, partition, start)
        consumer.assign([tp])
        while bytes_accum < target_bytes:
            msg = consumer.poll(timeout=1.0)
            if not msg or msg.error():
                break
            records.append(json.loads(msg.value()))
            bytes_accum += len(msg.value())
        if bytes_accum >= target_bytes:
            break
    consumer.close()

    df = pd.DataFrame(records)
    out_path = BASELINE_DIR / f"{topic}_baseline.csv"
    df.to_csv(out_path, index=False)
    return out_path

# ── Chunking helper: split DataFrame into ~chunk_mb CSVs ────────────────
@st.cache_data
def chunk_df_to_size(df: pd.DataFrame, prefix: str, chunk_mb: int = 40) -> list[Path]:
    sample = df.sample(n=min(len(df), 1000))
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    avg_row = len(buf.getvalue().encode()) / len(sample)

    rows_per_chunk = max(1, int((chunk_mb * 1024**2) / avg_row))
    paths = []
    for i in range(0, len(df), rows_per_chunk):
        subset = df.iloc[i : i + rows_per_chunk]
        path = CHUNKS_DIR / f"{prefix}_part{i // rows_per_chunk + 1}.csv"
        subset.to_csv(path, index=False)
        paths.append(path)
    return paths

# ── Cached BigQuery loaders and computation ─────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    KPI_SQL = f"""
    WITH
      viewed AS (...)
      -- [SQL as before]
    """
    return client.query(KPI_SQL).to_dataframe()

@st.cache_data
def compute_trophy_segments(sample_limit=50000, k=4):
    # [Function body as before]
    return coords, df, summary, km.cluster_centers_

@st.cache_data
def update_year(y: int):
    # [Function body as before]
    return fig

# ── Live update function (non-cached) ─────────────────────────────────────
def update_live():
    for tp in live_consumer.assignment():
        live_consumer.seek(TopicPartition(tp.topic, tp.partition, 0))
    msgs = live_consumer.consume(num_messages=200, timeout=1.0)
    # [Function body as before]
    return fig

# ── Streamlit app ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Euphoria Analytical Dashboard", layout="wide")
    st.title("Euphoria Analytical Dashboard")

    # Preload data
    df_kpi = load_kpis()
    coords_seg, df_seg, df_seg_summary, seg_centers = compute_trophy_segments()

    tabs = st.tabs([
        "SQL Runner", "Live Watch (5m)", "KPI Dashboard",
        "Yearly Watch Rank", "Trophy Segments"
    ])

    # SQL Runner
    with tabs[0]:
        st.header("BigQuery SQL Runner")
        # [UI code as before]

    # Live Watch
    with tabs[1]:
        st.header("Live Watch Hours (last 5 min)")
        if st.button("Refresh Live Data"): pass
        st.plotly_chart(update_live(), use_container_width=True)

    # KPI Dashboard
    with tabs[2]:
        st.header("Euphoria KPIs")
        # [UI code as before]

    # Yearly Watch Rank
    with tabs[3]:
        st.header("Yearly Relative Watch Hours")
        # [UI code as before]

    # Trophy Segments
    with tabs[4]:
        st.header("Trophy Buyer Segments (MCA + KMeans)")
        # [UI code as before]

if __name__ == "__main__":
    main()
