import os
import json
import random
import io
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import prince
import plotly.express as px
from confluent_kafka import Consumer, TopicPartition
from sklearn.cluster import KMeans
import streamlit as st

# ── Directory setup ─────────────────────────────────────────────────────────
BASELINE_DIR = Path.cwd() / "baseline_samples"
CHUNKS_DIR = Path.cwd() / "data_chunks"
DATA_DIR = Path.cwd() / "data_csvs"
for d in (BASELINE_DIR, CHUNKS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ── Kafka config loader ───────────────────────────────────────────────────────
def get_kafka_conf(group_id: str):
    kc = st.secrets.get("kafka", {})
    return {
        'bootstrap.servers':  kc.get('servers', ''),
        'security.protocol':  'SASL_SSL',
        'sasl.mechanisms':    'PLAIN',
        'sasl.username':      kc.get('username', ''),
        'sasl.password':      kc.get('password', ''),
        'group.id':           group_id,
        'auto.offset.reset':  'earliest',
        'enable.auto.commit': False
    }

# ── 1) Sample Kafka topic to CSVs ────────────────────────────────────────────
@st.cache_data
def sample_topic_to_size(topic: str, target_mb: int = 99, max_partitions: int = 5) -> dict:
    """
    Pull ~target_mb MB of random messages from Kafka topic,
    save to both baseline_samples/ and data_csvs/ as <topic>.csv.
    Returns dict with paths.
    """
    conf = get_kafka_conf('streamlit-sample-group')
    consumer = Consumer(conf)
    md = consumer.list_topics(topic, timeout=10.0)
    parts = list(md.topics.get(topic, {}).partitions.keys())[:max_partitions]
    random.shuffle(parts)

    records, bytes_accum = [], 0
    target_bytes = target_mb * 1024**2
    for p in parts:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, p))
        if high <= low:
            continue
        start = random.randint(low, high - 1)
        consumer.assign([TopicPartition(topic, p, start)])
        while bytes_accum < target_bytes:
            msg = consumer.poll(1.0)
            if not msg or msg.error():
                break
            rec = json.loads(msg.value())
            records.append(rec)
            bytes_accum += len(msg.value())
        if bytes_accum >= target_bytes:
            break
    consumer.close()

    df = pd.DataFrame(records)
    baseline_path = BASELINE_DIR / f"{topic}_baseline.csv"
    data_path = DATA_DIR / f"{topic}.csv"
    df.to_csv(baseline_path, index=False)
    df.to_csv(data_path, index=False)
    return {'baseline': baseline_path, 'data_csv': data_path}

# ── 2) Chunk DataFrame into ~chunk_mb CSV parts ─────────────────────────────
@st.cache_data
def chunk_df_to_size(df: pd.DataFrame, prefix: str, chunk_mb: int = 40) -> list[Path]:
    sample = df.sample(min(len(df), 1000))
    buf = io.StringIO(); sample.to_csv(buf, index=False)
    avg = len(buf.getvalue().encode()) / len(sample)
    rows = max(1, int((chunk_mb * 1024**2) / avg))

    paths = []
    for i in range(0, len(df), rows):
        part = df.iloc[i:i+rows]
        p = CHUNKS_DIR / f"{prefix}_part{i//rows+1}.csv"
        part.to_csv(p, index=False)
        paths.append(p)
    return paths
@st.cache_data
def load_topic_csv(topic: str) -> pd.DataFrame:
    """
    Look first for chunked CSV parts in data_chunks/<topic>_part*.csv;
    if none exist, fall back to the single data_csvs/<topic>.csv.
    """
    # 1) try chunked parts
    parts = sorted(CHUNKS_DIR.glob(f"{topic}_part*.csv"))
    if parts:
        return pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    # 2) try single CSV
    single = DATA_DIR / f"{topic}.csv"
    if single.exists():
        return pd.read_csv(single)
    raise FileNotFoundError(f"No CSV found for topic `{topic}`")

@st.cache_data
def load_full_topic_to_csv(topic: str) -> Path:
    """
    Consume every message from `topic` and write it to data_csvs/<topic>.csv
    """
    conf = get_kafka_conf('streamlit-full-group')
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    records = []
    while True:
        msg = consumer.poll(1.0)
        if msg is None:  # no more messages
            break
        if msg.error():
            continue
        records.append(json.loads(msg.value()))

    consumer.close()
    df = pd.DataFrame(records)
    path = DATA_DIR / f"{topic}.csv"
    df.to_csv(path, index=False)
    return path

# ── 3) CSV-based analytics ─────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    # The list of topics we need for KPIs:
    topics = [
        "watch_topic",
        "purchase_events_topic",
        "streams_topic",
        "partners_topic",
        "games_topic",
    ]

    # Check that at least one CSV (or chunk) exists for each topic:
    missing = []
    for t in topics:
        single = DATA_DIR / f"{t}.csv"
        parts  = list(CHUNKS_DIR.glob(f"{t}_part*.csv"))
        if not single.exists() and not parts:
            missing.append(t)
    if missing:
        raise FileNotFoundError(f"No CSV found for topic(s): {missing}")

    # Load each DataFrame:
    watch    = load_topic_csv("watch_topic")
    purchase = load_topic_csv("purchase_events_topic")
    streams  = load_topic_csv("streams_topic")
    partners = load_topic_csv("partners_topic")
    games    = load_topic_csv("games_topic")

    # Top 10 viewed countries
    viewed = (
        watch.groupby("country")["length"]
        .sum()
        .nlargest(10)
        .reset_index()
        .rename(columns={"length": "value", "country": "label"})
    )
    viewed["kpi"] = "Top 10 Viewed Countries"

    # Top 8 purchased products
    purchased = (
        purchase.groupby("product_name")
        .size()
        .nlargest(8)
        .reset_index(name="value")
        .rename(columns={"product_name": "label"})
    )
    purchased["kpi"] = "Top 8 Purchased Products"

    # Top 10 streamer performance
    sp = streams.merge(partners, on="partner_id")
    sp["score"] = (sp.viewers_total / sp.length.replace(0, 1)) * sp.comments_total
    streamer = (
        sp.groupby("screen_name")["score"]
        .sum()
        .nlargest(10)
        .reset_index()
        .rename(columns={"screen_name": "label", "score": "value"})
    )
    streamer["kpi"] = "Top 10 Streamer Performance"
    streamer["value"] = streamer["value"].round(2).astype(str)

    # Top 2 best-selling games
    best_games = (
        purchase[purchase.category == "game"]
        .groupby("product_name")
        .size()
        .nlargest(2)
        .reset_index(name="value")
        .rename(columns={"product_name": "label"})
    )
    best_games["kpi"] = "Top 2 Best-Selling Games"

    # Top 2 most-streamed games
    sg = (
        streams.merge(games, on="game_id")
        .groupby("title")
        .size()
        .nlargest(2)
        .reset_index(name="value")
        .rename(columns={"title": "label"})
    )
    sg["kpi"] = "Top 2 Most-Streamed Games"

    # Combine into one DataFrame
    return pd.concat(
        [
            viewed[["kpi", "label", "value"]],
            purchased[["kpi", "label", "value"]],
            streamer[["kpi", "label", "value"]],
            best_games[["kpi", "label", "value"]],
            sg[["kpi", "label", "value"]],
        ],
        ignore_index=True,
    )

@st.cache_data
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    """
    Load purchase_events_topic and customers_topic via CSVs or chunks,
    filter for "Authentic Mahiman Trophy", perform MCA then KMeans.
    Returns (coords_df, full_df, summary_df, cluster_centers)
    """
    # 1) Load data
    try:
        purchase = load_topic_csv("purchase_events_topic")
        cust = load_topic_csv("customers_topic")
    except FileNotFoundError as e:
        # no data yet
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # 2) Filter trophy buyers
    df = purchase.merge(cust, on="customer_id", how="inner")
    df = df[df.product_name == "Authentic Mahiman Trophy"].copy()
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    # 3) Compute age
    df['age'] = ((pd.to_datetime('today') - pd.to_datetime(df['birthday'], errors='coerce')).dt.days // 365)
    df = df[['customer_id','age','gender','region']].head(sample_limit)

    # 4) Drop rows with missing values
    df = df.dropna(subset=['age','gender','region'])
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    # 5) Binning age
    df['age_bin'] = pd.cut(
        df['age'], bins=range(10, 81, 5),
        labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False
    )
    df = df.dropna(subset=['age_bin'])
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    # 6) MCA on categorical columns
    df_mca = df[['age_bin','gender','region']].astype(str)
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df_mca)
    coords = pd.DataFrame(coords_arr, columns=['Dim1','Dim2'], index=df.index)

    # 7) KMeans clustering
    coords_clean = coords.dropna()
    if coords_clean.empty:
        return coords, df, pd.DataFrame(), []
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(coords_clean)
    coords_clean = coords_clean.assign(cluster=labels)

    # 8) Map clusters back to full coords and df
    coords = coords.join(coords_clean['cluster']).fillna(-1)
    df['cluster'] = coords['cluster'].astype(int)

    # 9) Summary statistics
    summary = coords_clean.groupby('cluster').agg(
        size=('cluster','count'),
        avg_dim1=('Dim1','mean')
    ).reset_index()

    return coords, df, summary, km.cluster_centers_


@st.cache_data
# Generate choropleth of watch hours by country for a given year
def update_year(y: int):
    data_file = DATA_DIR / "watch_topic.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing watch CSV: {data_file}")

    watch = pd.read_csv(data_file)
    # Parse dates, coerce errors to NaT, then drop
    watch['date'] = pd.to_datetime(watch['date'], utc=True, errors='coerce')
    watch = watch.dropna(subset=['date'])

    # Filter by year
    df_year = watch[watch['date'].dt.year == y]
    if df_year.empty:
        # return empty figure
        empty = pd.DataFrame({'country':[], 'watch_hours':[]})
        fig = px.choropleth(empty, locations='country', color='watch_hours')
        fig.update_layout(title=f"Yearly Watch Rank: {y} (no data)")
        return fig

    grouped = df_year.groupby('country', as_index=False)['length'].sum()
    grouped['watch_hours'] = grouped['length'] / 3600.0
    grouped['pct_rank'] = grouped['watch_hours'].rank(pct=True)

    fig = px.choropleth(
        grouped,
        locations='country',
        locationmode='country names',
        color='pct_rank',
        hover_data={'watch_hours':':.1f', 'pct_rank':':.2f'},
        color_continuous_scale='Viridis',
        range_color=(0,1)
    )
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig

def update_live():
    conf = get_kafka_conf('streamlit-live-group')
    consumer = Consumer(conf)
    consumer.subscribe(["watch_live_topic"])
    for tp in consumer.assignment():
        consumer.seek(TopicPartition(tp.topic, tp.partition, 0))
    msgs = consumer.consume(num_messages=200, timeout=1.0)
    consumer.close()

    recs = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    for m in msgs or []:
        if m and not m.error():
            d = json.loads(m.value().decode())
            ts = pd.to_datetime(d['date'], utc=True, infer_datetime_format=True)
            if ts >= cutoff:
                recs.append(d)
    if not recs:
        fig = px.choropleth(pd.DataFrame(columns=['country','watch_hours']), locations='country', color='watch_hours')
        fig.update_layout(title="No live data")
        return fig
    df_live = pd.DataFrame(recs).groupby('country')['length'].sum().reset_index()
    df_live['watch_hours'] = df_live.length/3600
    fig = px.choropleth(df_live, locations='country', color='watch_hours', hover_name='country', color_continuous_scale='Viridis')
    fig.update_layout(title="Live Watch Hours (last 5 min)")
    return fig

# ── Streamlit app UI ─────────────────────────────────────────────────────────
def main():
    st.title("Euphoria Analytical Dashboard")

    # Load KPI data (CSV)
    try:
        df_kpi = load_kpis()
    except FileNotFoundError as e:
        df_kpi = pd.DataFrame()
        st.warning(f"KPI CSV missing: run sampling first ({e})")

    # Load segment data
    try:
        coords_seg, df_seg, df_seg_summary, seg_centers = compute_trophy_segments()
    except FileNotFoundError as e:
        coords_seg = pd.DataFrame()
        df_seg_summary = pd.DataFrame()
        seg_centers = []
        st.warning(f"Segment CSV missing: run sampling first ({e})")

    tabs = st.tabs(["Data Sampling", "Live Watch", "KPIs", "Yearly Rank", "Segments"])

    with tabs[0]:
        st.header("Kafka → CSV Sampling & CSV Management")

        # 1) Sample ~99MB slice per topic
        if st.button("Sample Kafka Topics to CSVs (~99MB each)  "):
            topics = [
                'watch_topic','purchase_events_topic','streams_topic',
                'partners_topic','games_topic','customers_topic'
            ]
            results = {}
            from confluent_kafka import KafkaException
            for t in topics:
                try:
                    paths = sample_topic_to_size(t)
                    results[t] = {k: str(v) for k, v in paths.items()}
                except KafkaException as ke:
                    results[t] = {'error': f'Kafka error: {ke}'}
                except Exception as e:
                    results[t] = {'error': str(e)}
            st.json(results)

        # 2) Pull full purchase & customer topics for segments
        if st.button("Pull Full Purchase & Customer CSVs (complete)"):
            full_results = {}
            for t in ['purchase_events_topic','customers_topic']:
                try:
                    path = load_full_topic_to_csv(t)
                    full_results[t] = str(path)
                except Exception as e:
                    full_results[t] = f"Error: {e}"
            st.json(full_results)

        # 3) Chunk existing CSVs into ~40MB parts for GitHub
        if st.button("Chunk All CSVs to ~40MB"):
            chunk_results = {}
            for p in DATA_DIR.glob("*_topic.csv"):
                topic = p.stem
                try:
                    df = load_topic_csv(topic)
                    parts = chunk_df_to_size(df, topic)
                    chunk_results[topic] = [str(x) for x in parts]
                except Exception as e:
                    chunk_results[topic] = f"Error: {e}"
            st.json(chunk_results)

        st.markdown("---")
        st.write("**Current CSV files in data_csvs/**")
        st.write([x.name for x in DATA_DIR.iterdir()])

    with tabs[1]:


        st.header("Live Watch (last 5 min)")
        if st.button("Refresh Live"):
            pass
        st.plotly_chart(update_live(), use_container_width=True)

    with tabs[2]:
        st.header("Euphoria KPIs")
        if df_kpi.empty:
            st.write("No KPI data. Sample Kafka topics first.")
        else:
            kpi = st.selectbox("Select KPI", df_kpi.kpi.unique())
            st.dataframe(df_kpi[df_kpi.kpi == kpi])

    with tabs[3]:
        st.header("Yearly Watch Rank")
        year = st.selectbox("Year", list(range(datetime.now().year, datetime.now().year-10, -1)))
        if df_kpi.empty:
            st.write("No watch data. Sample Kafka topics first.")
        else:
            st.plotly_chart(update_year(year), use_container_width=True)

    with tabs[4]:
        st.header("Buyer Segments")
        if df_seg_summary.empty:
            st.write("No segment data. Sample Kafka topics first.")
        else:
            st.dataframe(df_seg_summary)
            fig = px.scatter(coords_seg, x='Dim1', y='Dim2', color='cluster')
            if len(seg_centers) > 0:
                fig.add_scatter(x=seg_centers[:,0], y=seg_centers[:,1], mode='markers', marker=dict(symbol='x', size=12))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

