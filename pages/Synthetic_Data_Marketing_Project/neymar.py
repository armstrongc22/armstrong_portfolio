import os
import re
import json
import pandas as pd
import prince
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
from confluent_kafka import Consumer, TopicPartition
from datetime import datetime, timezone, timedelta
import streamlit as st
from sklearn.cluster import KMeans

# ── 1) Streamlit configuration ───────────────────────────────────────────
st.set_page_config(page_title="Euphoria Analytical Dashboard", layout="wide")

# ── 2) Kafka consumer setup ───────────────────────────────────────────────
KAFKA_CONF = {
    'bootstrap.servers':  'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'security.protocol':  'SASL_SSL',
    'sasl.mechanisms':    'PLAIN',
    'sasl.username':      'H7C5SHD4EUVIGHTZ',
    'sasl.password':      'oyA7H99XPrK6c6I/aA3yB5fGhAlcp055Hr9ZdPrcqm5qlPRdsshfzS/Ku4xCHD8z',
    'group.id':           'streamlit-live-group',
    'auto.offset.reset':  'earliest',
    'enable.auto.commit': False
}

live_consumer = Consumer(KAFKA_CONF)
live_consumer.subscribe(['watch_live_topic'])

# ── 3) BigQuery client setup ──────────────────────────────────────────────
KEY_PATH = "mindful-vial-460001-h6-4d83b36dd3e9.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
CRED = service_account.Credentials.from_service_account_file(KEY_PATH)
PROJECT = "mindful-vial-460001-h6"
DATASET = "euphoria"
client = bigquery.Client(project=PROJECT, credentials=CRED)

# ── 4) Cached data loaders ─────────────────────────────────────────────────
@st.cache_data
def load_kpis():
    KPI_SQL = f"""
    WITH
      viewed AS (
        SELECT country, SUM(length) AS total_watch_seconds
        FROM `{PROJECT}.{DATASET}.watch_topic`
        GROUP BY country
        ORDER BY total_watch_seconds DESC
        LIMIT 10
      ),
      purchased AS (
        SELECT product_name, COUNT(*) AS purchase_count
        FROM `{PROJECT}.{DATASET}.purchase_events_topic`
        GROUP BY product_name
        ORDER BY purchase_count DESC
        LIMIT 8
      ),
      streamer_perf AS (
        SELECT
          p.screen_name,
          SUM((s.viewers_total/NULLIF(s.length,0)) * s.comments_total) AS performance_score
        FROM `{PROJECT}.{DATASET}.streams_topic`   AS s
        JOIN `{PROJECT}.{DATASET}.partners_topic`  AS p
          ON s.partner_id = p.partner_id
        GROUP BY p.screen_name
        ORDER BY performance_score DESC
        LIMIT 10
      ),
      best_games AS (
        SELECT product_name, COUNT(*) AS purchase_count
        FROM `{PROJECT}.{DATASET}.purchase_events_topic`
        WHERE category = 'game'
        GROUP BY product_name
        ORDER BY purchase_count DESC
        LIMIT 2
      ),
      streamed_games AS (
        SELECT
          g.title      AS game_title,
          COUNT(*)     AS stream_count
        FROM `{PROJECT}.{DATASET}.streams_topic` AS s
        JOIN `{PROJECT}.{DATASET}.games_topic`   AS g
          ON s.game_id = g.game_id
        GROUP BY g.title
        ORDER BY stream_count DESC
        LIMIT 2
      ),
      target_union AS (
        SELECT 'Top 10 Viewed Countries' AS kpi, country AS label, CAST(total_watch_seconds AS STRING) AS value FROM viewed
        UNION ALL
        SELECT 'Top 8 Purchased Products', product_name, CAST(purchase_count AS STRING) FROM purchased
        UNION ALL
        SELECT 'Top 10 Streamer Performance', screen_name, CAST(ROUND(performance_score,2) AS STRING) FROM streamer_perf
        UNION ALL
        SELECT 'Top 2 Best-Selling Games', product_name, CAST(purchase_count AS STRING) FROM best_games
        UNION ALL
        SELECT 'Top 2 Most-Streamed Games', game_title, CAST(stream_count AS STRING) FROM streamed_games
      )
    SELECT * FROM target_union;
    """
    return client.query(KPI_SQL).to_dataframe()

@st.cache_data
def compute_trophy_segments(sample_limit=50000, k=4):
    sql = f"""
      WITH trophy_profiles AS (
        SELECT
          c.customer_id,
          DATE_DIFF(CURRENT_DATE(), DATE(c.birthday), YEAR) AS age,
          c.gender,
          c.region
        FROM `{PROJECT}.{DATASET}.purchase_events_topic` p
        JOIN `{PROJECT}.{DATASET}.customers_topic` c USING(customer_id)
        WHERE p.category='merch' AND p.product_name='Authentic Mahiman Trophy'
      )
      SELECT * FROM trophy_profiles LIMIT {sample_limit}
    """
    df = client.query(sql).to_dataframe()
    df['age_bin'] = pd.cut(
      df['age'], bins=range(10,81,5), labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False
    )
    df_mca = df[['age_bin','gender','region']].astype(str)
    mca = prince.MCA(n_components=2, engine='sklearn', random_state=42).fit(df_mca)
    coords = mca.transform(df_mca)
    coords.columns = ['Dim1','Dim2']
    km = KMeans(n_clusters=k, random_state=42)
    coords['cluster'] = km.fit_predict(coords[['Dim1','Dim2']])
    df['cluster'] = coords['cluster']
    summary = pd.DataFrame([
      {'cluster': i, 'size': int((df.cluster==i).sum()), 'avg_age': df[df.cluster==i].age.mean()}
      for i in range(k)
    ])
    return coords, df, summary, km.cluster_centers_

@st.cache_data
def update_year(y):
    sql = f"""
      WITH country_totals AS (
        SELECT country, SUM(length)/3600.0 AS watch_hours
        FROM `{PROJECT}.{DATASET}.watch_topic`
        WHERE EXTRACT(YEAR FROM DATE(date)) = {y}
        GROUP BY country
      )
      SELECT country, watch_hours,
        PERCENT_RANK() OVER (ORDER BY watch_hours) AS pct_rank
      FROM country_totals
    """
    df = client.query(sql).to_dataframe()
    fig = px.choropleth(
        df, locations='country', locationmode='country names',
        color='pct_rank', hover_name='country', hover_data={'watch_hours':':.1f','pct_rank':':.2f'},
        color_continuous_scale='Viridis', range_color=(0,1)
    )
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig


def update_live():
    for tp in live_consumer.assignment():
        tp = TopicPartition(tp.topic, tp.partition, 0)
        live_consumer.seek(tp)
    msgs = live_consumer.consume(num_messages=200, timeout=1.0)
    records = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    for msg in msgs or []:
        if msg is None or msg.error(): continue
        rec = json.loads(msg.value().decode('utf-8'))
        ts = datetime.fromisoformat(rec['date'].replace('Z','+00:00'))
        if ts >= cutoff: records.append(rec)
    if not records:
        empty = pd.DataFrame({'country':[], 'watch_hours':[]})
        fig = px.choropleth(empty, locations='country', locationmode='country names', color='watch_hours')
        fig.update_layout(title="No live data")
        return fig
    df_live = pd.DataFrame(records)
    agg = df_live.groupby('country', as_index=False)['length'].sum()
    agg['watch_hours'] = agg['length']/3600.0
    fig = px.choropleth(
        agg, locations='country', locationmode='country names', color='watch_hours',
        hover_name='country', color_continuous_scale='Viridis'
    )
    fig.update_layout(title="Live Watch Hours (last 5 min)")
    return fig


def main():
    st.title("Euphoria Analytical Dashboard")

    # Load and display KPIs
    df_kpi = load_kpis()
    # Compute segments
    coords_seg, df_seg, df_seg_summary, seg_centers = compute_trophy_segments()

    tabs = st.tabs(["SQL Runner", "Live Watch (5m)", "KPI Dashboard", "Yearly Watch Rank", "Trophy Segments"])

    # SQL Runner
    with tabs[0]:
        st.header("BigQuery SQL Runner")
        default_query = f"SELECT * FROM `{PROJECT}.{DATASET}.streams_topic` LIMIT 5;"
        query = st.text_area("SQL Query", value=default_query, height=120)
        if st.button("Run Query"):
            q = re.sub(r"\bPROJECT\b", PROJECT, query)
            q = re.sub(r"\bDATASET\b", DATASET, q)
            df = client.query(q).to_dataframe()
            st.dataframe(df)

    # Live Watch
    with tabs[1]:
        st.header("Live Watch Hours (last 5 min)")
        if st.button("Refresh Live Data"):
            pass
        fig_live = update_live()
        st.plotly_chart(fig_live, use_container_width=True)

    # KPI Dashboard
    with tabs[2]:
        st.header("Euphoria KPIs")
        kpi = st.selectbox("Select KPI", options=df_kpi.kpi.unique())
        st.dataframe(df_kpi[df_kpi.kpi == kpi])

    # Yearly Watch Rank
    with tabs[3]:
        st.header("Yearly Relative Watch Hours")
        current_year = datetime.now().year
        year = st.selectbox("Year", options=list(range(current_year, current_year-10, -1)))
        fig_year = update_year(year)
        st.plotly_chart(fig_year, use_container_width=True)

    # Trophy Segments
    with tabs[4]:
        st.header("Trophy Buyer Segments (MCA + KMeans)")
        st.dataframe(df_seg_summary)
        fig_seg = px.scatter(coords_seg, x='Dim1', y='Dim2', color='cluster', title='Segments')
        fig_seg.add_scatter(x=seg_centers[:,0], y=seg_centers[:,1], mode='markers', marker=dict(symbol='x', size=12, color='black'))
        st.plotly_chart(fig_seg, use_container_width=True)

if __name__ == "__main__":
    main()