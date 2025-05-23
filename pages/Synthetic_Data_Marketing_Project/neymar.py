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

# ── Directories ─────────────────────────────────────────────────────────────
BASELINE_DIR = Path.cwd() / "baseline_samples"
CHUNKS_DIR = Path.cwd() / "data_chunks"
DATA_DIR = Path.cwd() / "data_csvs"
for d in (BASELINE_DIR, CHUNKS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ── Kafka configurations ────────────────────────────────────────────────────
SAMPLE_KAFKA_CONF = {
    'bootstrap.servers':  os.getenv('CONFLUENT_BOOTSTRAP', ''),
    'security.protocol':  'SASL_SSL',
    'sasl.mechanisms':    'PLAIN',
    'sasl.username':      os.getenv('CONFLUENT_API_KEY_SAMPLE', ''),
    'sasl.password':      os.getenv('CONFLUENT_API_SECRET_SAMPLE', ''),
    'group.id':           'streamlit-sample-group',
    'auto.offset.reset':  'earliest',
    'enable.auto.commit': False
}
LIVE_KAFKA_CONF = SAMPLE_KAFKA_CONF.copy()
LIVE_KAFKA_CONF['group.id'] = 'streamlit-live-group'

live_consumer = Consumer(LIVE_KAFKA_CONF)
live_consumer.subscribe(["watch_live_topic"])

# ── Sampling helper ──────────────────────────────────────────────────────────
@st.cache_data
def sample_topic_to_size(topic: str, target_mb: int = 99, max_partitions: int = 5) -> Path:
    target_bytes = target_mb * 1024**2
    consumer = Consumer(SAMPLE_KAFKA_CONF)
    md = consumer.list_topics(topic, timeout=10.0)
    parts = list(md.topics[topic].partitions.keys())[:max_partitions]
    random.shuffle(parts)
    records, bytes_accum = [], 0
    for p in parts:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, p))
        if high <= low: continue
        start = random.randint(low, high - 1)
        consumer.assign([TopicPartition(topic, p, start)])
        while bytes_accum < target_bytes:
            msg = consumer.poll(1.0)
            if not msg or msg.error(): break
            records.append(json.loads(msg.value()))
            bytes_accum += len(msg.value())
        if bytes_accum >= target_bytes: break
    consumer.close()
    df = pd.DataFrame(records)
    out = BASELINE_DIR / f"{topic}_baseline.csv"
    df.to_csv(out, index=False)
    return out

# ── Chunking helper ──────────────────────────────────────────────────────────
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

# ── CSV-based analytics ─────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    # read CSVs
    watch = pd.read_csv(DATA_DIR / "watch_topic.csv")
    purchase = pd.read_csv(DATA_DIR / "purchase_events_topic.csv")
    streams = pd.read_csv(DATA_DIR / "streams_topic.csv")
    partners = pd.read_csv(DATA_DIR / "partners_topic.csv")
    games = pd.read_csv(DATA_DIR / "games_topic.csv")
    # compute
    viewed = watch.groupby('country')['length'].sum().nlargest(10).reset_index()
    viewed['kpi'], viewed['value'] = 'Top 10 Viewed Countries', viewed['length'].astype(str)
    purchased = purchase.groupby('product_name').size().nlargest(8).reset_index(name='count')
    purchased['kpi'], purchased['label'], purchased['value'] = 'Top 8 Purchased Products', purchased['product_name'], purchased['count'].astype(str)
    sp = streams.merge(partners, on='partner_id')
    sp['score'] = (sp.viewers_total/sp.length.replace(0,1))*sp.comments_total
    streamer = sp.groupby('screen_name')['score'].sum().nlargest(10).reset_index()
    streamer['kpi'], streamer['value'] = 'Top 10 Streamer Performance', streamer['score'].round(2).astype(str)
    best_games = purchase[purchase.category=='game'].groupby('product_name').size().nlargest(2).reset_index(name='count')
    best_games['kpi'], best_games['value'] = 'Top 2 Best-Selling Games', best_games['count'].astype(str)
    sg = streams.merge(games, on='game_id').groupby('title').size().nlargest(2).reset_index(name='count')
    sg['kpi'], sg['label'], sg['value'] = 'Top 2 Most-Streamed Games', sg['title'], sg['count'].astype(str)
    dfs = [viewed[['kpi','country','value']].rename(columns={'country':'label'}),
           purchased[['kpi','label','value']],
           streamer[['kpi','screen_name','value']].rename(columns={'screen_name':'label'}),
           best_games[['kpi','product_name','value']].rename(columns={'product_name':'label'}),
           sg[['kpi','label','value']]]
    df = pd.concat(dfs, ignore_index=True)
    return df[['kpi','label','value']]

@st.cache_data
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    purchase = pd.read_csv(DATA_DIR / "purchase_events_topic.csv")
    cust = pd.read_csv(DATA_DIR / "customers_topic.csv")
    df = purchase.merge(cust, on='customer_id')
    df = df[df.product_name=='Authentic Mahiman Trophy'].copy()
    df['age'] = (pd.to_datetime('today') - pd.to_datetime(df.birthday)).dt.days // 365
    df = df[['customer_id','age','gender','region']].head(sample_limit)
    df['age_bin'] = pd.cut(df.age, bins=range(10,81,5), labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False)
    mca = prince.MCA(n_components=2, random_state=42).fit(df[['age_bin','gender','region']].astype(str))
    coords = pd.DataFrame(mca.transform(df[['age_bin','gender','region']]), columns=['Dim1','Dim2'])
    km = KMeans(n_clusters=k, random_state=42).fit(coords)
    coords['cluster'] = km.labels_
    df['cluster'] = coords['cluster']
    summary = coords.groupby('cluster').agg(size=('cluster','count'), avg_age=('Dim1','mean')).reset_index()
    return coords, df, summary, km.cluster_centers_

@st.cache_data
def update_year(y: int):
    watch = pd.read_csv(DATA_DIR / "watch_topic.csv")
    watch['date'] = pd.to_datetime(watch.date)
    df = watch[watch.date.dt.year==y].groupby('country')['length'].sum().reset_index()
    df['watch_hours'] = df.length/3600
    df['pct_rank'] = df.watch_hours.rank(pct=True)
    fig = px.choropleth(df, locations='country', locationmode='country names', color='pct_rank',
                        hover_data={'watch_hours':':.1f','pct_rank':':.2f'}, color_continuous_scale='Viridis')
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig

def update_live():
    for tp in live_consumer.assignment():
        live_consumer.seek(TopicPartition(tp.topic, tp.partition, 0))
    msgs = live_consumer.consume(num_messages=200, timeout=1.0)
    recs = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    for m in msgs or []:
        if m and not m.error():
            d = json.loads(m.value().decode())
            ts = pd.to_datetime(d['date'])
            if ts >= cutoff: recs.append(d)
    if not recs:
        fig = px.choropleth(pd.DataFrame(columns=['country','watch_hours']), locations='country', color='watch_hours')
        fig.update_layout(title="No live data")
        return fig
    df_live = pd.DataFrame(recs).groupby('country')['length'].sum().reset_index()
    df_live['watch_hours'] = df_live.length/3600
    fig = px.choropleth(df_live, locations='country', color='watch_hours', hover_name='country', color_continuous_scale='Viridis')
    fig.update_layout(title="Live Watch Hours (last 5 min)")
    return fig

def main():
    st.set_page_config(page_title="Euphoria Analytical Dashboard", layout="wide")
    st.title("Euphoria Analytical Dashboard")
    df_kpi = load_kpis()
    coords_seg, df_seg, df_seg_summary, seg_centers = compute_trophy_segments()
    tabs = st.tabs(["Data Sampling", "Live Watch (5m)", "KPI Dashboard", "Yearly Rank", "Segments"])
    with tabs[0]:
        st.header("Data Sampling")
        if st.button("Generate Baselines (~99MB)"):
            topics = [f.stem for f in (DATA_DIR).glob("*_topic.csv")]
            res = {t: sample_topic_to_size(t) for t in topics}
            st.write(res)
        topic = st.selectbox("Topic to chunk", [f.stem for f in DATA_DIR.glob("*_topic.csv")])
        if st.button("Chunk to 40MB parts"):
            df = pd.read_csv(DATA_DIR/f"{topic}.csv")
            parts = chunk_df_to_size(df, topic)
            st.write(parts)
    with tabs[1]:
        st.header("Live Watch (last 5 min)")
        if st.button("Refresh Live"):
            pass
        st.plotly_chart(update_live(), use_container_width=True)
    with tabs[2]:
        st.header("Euphoria KPIs")
        kpi = st.selectbox("Select KPI", df_kpi.kpi.unique())
        st.dataframe(df_kpi[df_kpi.kpi==kpi])
    with tabs[3]:
        st.header("Yearly Watch Rank")
        year = st.selectbox("Year", list(range(datetime.now().year, datetime.now().year-10, -1)))
        st.plotly_chart(update_year(year), use_container_width=True)
    with tabs[4]:
        st.header("Buyer Segments")
        st.dataframe(df_seg_summary)
        fig = px.scatter(coords_seg, x='Dim1', y='Dim2', color='cluster')
        fig.add_scatter(x=seg_centers[:,0], y=seg_centers[:,1], mode='markers', marker=dict(symbol='x', size=12))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
