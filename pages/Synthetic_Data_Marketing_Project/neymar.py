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

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria Analytical Dashboard", layout="wide")

# ── Directories ───────────────────────────────────────────────────────────
BASELINE_DIR = Path.cwd() / "baseline_samples"
CHUNKS_DIR = Path.cwd() / "data_chunks"
DATA_DIR = Path.cwd() / "data_csvs"
for d in (BASELINE_DIR, CHUNKS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ── Kafka config loader ─────────────────────────────────────────────────────
def get_kafka_conf(group_id: str) -> dict:
    kc = st.secrets.get("kafka", {})
    return {
        'bootstrap.servers': kc.get('servers', ''),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': kc.get('username', ''),
        'sasl.password': kc.get('password', ''),
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    }

# ── 1) Sample Kafka to CSV ─────────────────────────────────────────────────
@st.cache_data
def sample_topic_to_size(topic: str, target_mb: int = 99, max_partitions: int = 5) -> dict:
    """
    Pull ~target_mb MB of random messages from Kafka topic,
    save to baseline_samples/ and data_csvs/.
    """
    consumer = Consumer(get_kafka_conf('streamlit-sample-group'))
    md = consumer.list_topics(topic, timeout=10.0)
    partitions = list(md.topics.get(topic, {}).partitions.keys())[:max_partitions]
    random.shuffle(partitions)

    records = []
    bytes_accum = 0
    target_bytes = target_mb * 1024**2
    for part in partitions:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, part))
        if high <= low:
            continue
        start = random.randint(low, high - 1)
        consumer.assign([TopicPartition(topic, part, start)])
        while bytes_accum < target_bytes:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
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

# ── 2) Chunk DataFrame ─────────────────────────────────────────────────────
@st.cache_data
def chunk_df_to_size(df: pd.DataFrame, prefix: str, chunk_mb: int = 40) -> list[Path]:
    sample = df.sample(n=min(len(df), 1000))
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    avg_bytes = len(buf.getvalue().encode()) / len(sample)
    rows_per_chunk = max(1, int((chunk_mb * 1024**2) / avg_bytes))

    parts = []
    for start in range(0, len(df), rows_per_chunk):
        subset = df.iloc[start:start + rows_per_chunk]
        path = CHUNKS_DIR / f"{prefix}_part{start//rows_per_chunk+1}.csv"
        subset.to_csv(path, index=False)
        parts.append(path)
    return parts

# ── Helpers to load CSVs ────────────────────────────────────────────────────
@st.cache_data
def load_full_topic_to_csv(topic: str) -> Path:
    consumer = Consumer(get_kafka_conf('streamlit-full-group'))
    consumer.subscribe([topic])
    records = []
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            break
        if msg.error():
            continue
        records.append(json.loads(msg.value()))
    consumer.close()
    df = pd.DataFrame(records)
    path = DATA_DIR / f"{topic}.csv"
    df.to_csv(path, index=False)
    return path

@st.cache_data
def load_topic_csv(topic: str) -> pd.DataFrame:
    parts = sorted(CHUNKS_DIR.glob(f"{topic}_part*.csv"))
    if parts:
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    single = DATA_DIR / f"{topic}.csv"
    if single.exists():
        return pd.read_csv(single)
    raise FileNotFoundError(f"No CSV for topic '{topic}'")

# ── 3) Load KPIs ───────────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    required = ["watch_topic.csv", "purchase_events_topic.csv", "streams_topic.csv", "partners_topic.csv", "games_topic.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists() or (DATA_DIR / f).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty CSVs: {missing}")

    watch = pd.read_csv(DATA_DIR / "watch_topic.csv")
    purchase = pd.read_csv(DATA_DIR / "purchase_events_topic.csv")
    streams = pd.read_csv(DATA_DIR / "streams_topic.csv")
    partners = pd.read_csv(DATA_DIR / "partners_topic.csv")
    games = pd.read_csv(DATA_DIR / "games_topic.csv")

    # Top 10 viewed
    viewed = watch.groupby('country')['length'].sum().nlargest(10).reset_index()
    viewed = viewed.rename(columns={'country':'label','length':'value'})
    viewed['kpi'] = 'Top 10 Viewed Countries'

    # Top 8 purchased
    purchased = purchase.groupby('product_name').size().nlargest(8).reset_index(name='value')
    purchased = purchased.rename(columns={'product_name':'label'})
    purchased['kpi'] = 'Top 8 Purchased Products'

    # Top 10 streamer performance
    sp = streams.merge(partners, on='partner_id')
    sp['score'] = (sp.viewers_total / sp.length.replace(0,1)) * sp.comments_total
    streamer = sp.groupby('screen_name')['score'].sum().nlargest(10).reset_index()
    streamer = streamer.rename(columns={'screen_name':'label','score':'value'})
    streamer['value'] = streamer['value'].round(2)
    streamer['kpi'] = 'Top 10 Streamer Performance'

    # Top 2 best-selling games
    best_games = purchase[purchase.category=='game'].groupby('product_name').size().nlargest(2).reset_index(name='value')
    best_games = best_games.rename(columns={'product_name':'label'})
    best_games['kpi'] = 'Top 2 Best-Selling Games'

    # Top 2 most-streamed games
    sg = streams.merge(games, on='game_id').groupby('title').size().nlargest(2).reset_index(name='value')
    sg = sg.rename(columns={'title':'label'})
    sg['kpi'] = 'Top 2 Most-Streamed Games'

    return pd.concat([viewed[['kpi','label','value']],
                      purchased[['kpi','label','value']],
                      streamer[['kpi','label','value']],
                      best_games[['kpi','label','value']],
                      sg[['kpi','label','value']]], ignore_index=True)

# ── 4) Compute trophy segments ──────────────────────────────────────────────
@st.cache_data
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    try:
        purchase = load_topic_csv('purchase_events_topic')
        cust = load_topic_csv('customers_topic')
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    df = purchase.merge(cust, on='customer_id')
    df = df[df.product_name == 'Authentic Mahiman Trophy'].copy()
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    df['age'] = ((pd.to_datetime('today') - pd.to_datetime(df['birthday'], errors='coerce')).dt.days // 365)
    df = df[['customer_id','age','gender','region']].head(sample_limit).dropna()
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    df['age_bin'] = pd.cut(df['age'], bins=range(10,81,5), labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False)
    df = df.dropna(subset=['age_bin'])
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    df_mca = df[['age_bin','gender','region']].astype(str)
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df_mca)
    coords = pd.DataFrame(coords_arr, columns=['Dim1','Dim2'], index=df.index)

    coords_clean = coords.dropna()
    if coords_clean.empty:
        return coords, df, pd.DataFrame(), []
    km = KMeans(n_clusters=k, random_state=42)
    coords_clean['cluster'] = km.fit_predict(coords_clean[['Dim1','Dim2']])

    coords['cluster'] = coords_clean['cluster']
    df['cluster'] = coords['cluster'].astype(int)

    summary = coords_clean.groupby('cluster').agg(size=('cluster','count'), avg_dim1=('Dim1','mean')).reset_index()
    centers = km.cluster_centers_
    return coords, df, summary, centers

# ── 5) Yearly watch rank ───────────────────────────────────────────────────
@st.cache_data
def update_year(y: int):
    path = DATA_DIR / 'watch_topic.csv'
    if not path.exists():
        raise FileNotFoundError

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df_year = df[df['date'].dt.year == y]
    if df_year.empty:
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
        hover_data={'watch_hours':':.1f','pct_rank':':.2f'},
        color_continuous_scale='Viridis',
        range_color=(0,1)
    )
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig

# ── 6) Live watch ──────────────────────────────────────────────────────────
def update_live():
    consumer = Consumer(get_kafka_conf('streamlit-live-group'))
    consumer.subscribe(['watch_live_topic'])
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
        empty = pd.DataFrame({'country':[], 'watch_hours':[]})
        fig = px.choropleth(empty, locations='country', color='watch_hours')
        fig.update_layout(title="No live data")
        return fig

    df_live = pd.DataFrame(recs).groupby('country', as_index=False)['length'].sum()
    df_live['watch_hours'] = df_live['length'] / 3600.0

    fig = px.choropleth(
        df_live,
        locations='country',
        color='watch_hours',
        hover_name='country',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(title="Live Watch Hours (last 5 min)")
    return fig

# ── Streamlit UI ───────────────────────────────────────────────────────────
def main():
    st.title("Euphoria Analytical Dashboard")

    # Load KPIs
    try:
        df_kpi = load_kpis()
    except FileNotFoundError:
        df_kpi = pd.DataFrame()
        st.warning("Missing KPI CSVs. Sample data first.")

    tabs = st.tabs(["Data Sampling","Live Watch","KPIs","Yearly Rank","Segments"])

    with tabs[0]:
        st.header("Data Sampling & CSV Management")
        if st.button("Sample Kafka (99MB each)"):
            res = {t: sample_topic_to_size(t) for t in [
                'watch_topic','purchase_events_topic','streams_topic',
                'partners_topic','games_topic','customers_topic']}
            st.json(res)

        if st.button("Pull Full CSVs (purchase/customer)"):
            res = {t: str(load_full_topic_to_csv(t)) for t in ['purchase_events_topic','customers_topic']}
            st.json(res)

        if st.button("Chunk all CSVs (~40MB)"):
            out = {}
            for p in DATA_DIR.glob('*_topic.csv'):
                topic = p.stem
                try:
                    df = load_topic_csv(topic)
                    parts = chunk_df_to_size(df, topic)
                    out[topic] = [str(x) for x in parts]
                except FileNotFoundError:
                    out[topic] = []
            st.json(out)

        st.markdown("---")
        st.write("data_csvs:", [x.name for x in DATA_DIR.iterdir()])

    with tabs[1]:
        st.header("Live Watch (last 5 min)")
        if st.button("Refresh Live"): pass
        st.plotly_chart(update_live(), use_container_width=True)

    with tabs[2]:
        st.header("Euphoria KPIs")
        if df_kpi.empty:
            st.write("No KPI data. Sample data first.")
        else:
            choice = st.selectbox("Select KPI", df_kpi.kpi.unique())
            st.dataframe(df_kpi[df_kpi.kpi == choice])

    with tabs[3]:
        st.header("Yearly Watch Rank")
        years = list(range(datetime.now().year, datetime.now().year - 10, -1))
        yr = st.selectbox("Year", years)
        try:
            fig = update_year(yr)
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning(
                """
                Missing 'watch_topic.csv'. Sample data first.
                """
            )

    with tabs[4]:
        st.header("Buyer Segments")
        if 'seg' not in st.session_state:
            st.session_state.seg = {'coords':pd.DataFrame(), 'df':pd.DataFrame(), 'summary':pd.DataFrame(), 'centers':[]}
        if st.button("Compute Segments"):
            coords, df_seg, summary, centers = compute_trophy_segments()
            st.session_state.seg = {'coords':coords, 'df':df_seg, 'summary':summary, 'centers':centers}

        res = st.session_state.seg
        if res['df'].empty:
            st.warning(
                """
                Missing trophy-purchase data. Pull full CSVs, then click 'Compute Segments'.
                """
            )
        else:
            st.dataframe(res['summary'])
            fig = px.scatter(res['coords'], x='Dim1', y='Dim2', color='cluster')
            if len(res['centers']):
                fig.add_scatter(
                    x=res['centers'][:,0], y=res['centers'][:,1], mode='markers', marker=dict(symbol='x', size=12)
                )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()