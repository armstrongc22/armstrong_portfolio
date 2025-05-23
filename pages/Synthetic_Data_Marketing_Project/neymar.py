import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria Dashboard (CSV)", layout="wide")
DATA_DIR = Path("data_csvs")

# ── Utility: load all versioned CSVs for a topic ────────────────────────────
def load_topic_csvs(topic: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Reads files named <topic><N>.csv in data_dir, sorts by N, concatenates.
    Raises FileNotFoundError if none or all are empty.
    """
    pattern = f"{topic}" + "*" + ".csv"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs found for topic '{topic}' in {data_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            continue
    if not dfs:
        raise FileNotFoundError(f"All CSVs for '{topic}' are empty")
    return pd.concat(dfs, ignore_index=True)

# ── KPI Computation ────────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    # load raw tables
    watch = load_topic_csvs("watch_topic")
    purchase = load_topic_csvs("purchase_events_topic")
    streams = load_topic_csvs("streams_topic")
    partners = load_topic_csvs("partners_topic")
    games = load_topic_csvs("games_topic")

    # Top 10 viewed countries
    viewed = (
        watch.groupby('country')['length']
             .sum()
             .nlargest(10)
             .reset_index()
             .rename(columns={'country':'label','length':'value'})
    )
    viewed['kpi'] = 'Top 10 Viewed Countries'

    # Top 8 purchased products
    purchased = (
        purchase.groupby('product_name')
                .size()
                .nlargest(8)
                .reset_index(name='value')
                .rename(columns={'product_name':'label'})
    )
    purchased['kpi'] = 'Top 8 Purchased Products'

    # Top 10 streamer performance
    sp = streams.merge(partners, on='partner_id')
    sp['score'] = (sp.viewers_total / sp.length.replace(0,1)) * sp.comments_total
    streamer = (
        sp.groupby('screen_name')['score']
          .sum()
          .nlargest(10)
          .reset_index()
          .rename(columns={'screen_name':'label','score':'value'})
    )
    streamer['value'] = streamer['value'].round(2)
    streamer['kpi'] = 'Top 10 Streamer Performance'

    # Top 2 best-selling games
    best_games = (
        purchase[purchase.category=='game']
               .groupby('product_name')
               .size()
               .nlargest(2)
               .reset_index(name='value')
               .rename(columns={'product_name':'label'})
    )
    best_games['kpi'] = 'Top 2 Best-Selling Games'

    # Top 2 most-streamed games
    sg = (
        streams.merge(games, on='game_id')
               .groupby('title')
               .size()
               .nlargest(2)
               .reset_index(name='value')
               .rename(columns={'title':'label'})
    )
    sg['kpi'] = 'Top 2 Most-Streamed Games'

    # combine all
    kpis = pd.concat([
        viewed[['kpi','label','value']],
        purchased[['kpi','label','value']],
        streamer[['kpi','label','value']],
        best_games[['kpi','label','value']],
        sg[['kpi','label','value']]
    ], ignore_index=True)
    return kpis

# ── Yearly Watch Rank ──────────────────────────────────────────────────────
@st.cache_data
def update_year(y: int):
    watch = load_topic_csvs("watch_topic")
    watch['date'] = pd.to_datetime(watch['date'], errors='coerce')
    df_year = watch[watch['date'].dt.year == y]
    if df_year.empty:
        st.warning(f"No data for year {y}")
        return px.choropleth(pd.DataFrame(columns=['country','watch_hours']), locations='country', color='watch_hours')

    grouped = (
        df_year.groupby('country')['length']
               .sum()
               .reset_index()
               .rename(columns={'length':'watch_hours'})
    )
    grouped['pct_rank'] = grouped['watch_hours'].rank(pct=True)

    fig = px.choropleth(
        grouped,
        locations='country',
        locationmode='country names',
        color='pct_rank',
        hover_data=['watch_hours','pct_rank'],
        color_continuous_scale='Viridis',
        range_color=(0,1)
    )
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig

# ── Buyer Segments ─────────────────────────────────────────────────────────
@st.cache_data
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    purchase = load_topic_csvs("purchase_events_topic")
    customers = load_topic_csvs("customers_topic")
    df = purchase.merge(customers, on='customer_id')
    df = df[df.product_name=='Authentic Mahiman Trophy'].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    df['age'] = ((pd.to_datetime('today') - pd.to_datetime(df['birthday'], errors='coerce')).dt.days // 365)
    df = df[['customer_id','age','gender','region']].dropna().head(sample_limit)
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    df['age_bin'] = pd.cut(df['age'], bins=range(10,81,5), labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False)
    df = df.dropna(subset=['age_bin'])
    coords = prince.MCA(n_components=2, random_state=42).fit_transform(df[['age_bin','gender','region']])
    coords_df = pd.DataFrame(coords, columns=['Dim1','Dim2'], index=df.index)

    km = KMeans(n_clusters=k, random_state=42)
    df_clean = coords_df.dropna()
    labels = km.fit_predict(df_clean)
    df_clean['cluster'] = labels

    coords_df['cluster'] = df_clean['cluster']
    df['cluster'] = coords_df['cluster'].astype(int)

    summary = df_clean.groupby('cluster').agg(size=('cluster','count'), avg_dim1=('Dim1','mean')).reset_index()
    centers = km.cluster_centers_
    return coords_df, df, summary, centers

# ── Streamlit App ──────────────────────────────────────────────────────────
def main():
    st.title("Euphoria Analytical Dashboard (CSV)")
    tabs = st.tabs(["KPIs","Yearly Rank","Buyer Segments","Data Files"])

            with tabs[0]:
        st.header(\"Key Performance Indicators\")
        try:
            df_kpi = load_kpis()
        except FileNotFoundError as e:
            st.warning(str(e))
            df_kpi = pd.DataFrame()

        if df_kpi.empty:
            st.write(\"No KPI data available. Please ensure your CSVs are loaded.\")
        else:
            choice = st.selectbox(\"Select KPI\", df_kpi['kpi'].unique())
            filtered = df_kpi[df_kpi['kpi'] == choice]
            st.dataframe(filtered)

    with tabs[1]:」
        st.header("Yearly Watch Map")
        year = st.selectbox("Year", list(range(datetime.now().year, datetime.now().year-10, -1)))
        fig = update_year(year)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("Buyer Segments (Trophy Purchasers)")
        if st.button("Compute Segments"):
            coords, full, summary, centers = compute_trophy_segments()
            if full.empty:
                st.warning("No trophy-purchase data; ensure CSVs are present.")
            else:
                st.dataframe(summary)
                fig = px.scatter(coords, x='Dim1', y='Dim2', color='cluster')
                if len(centers): fig.add_scatter(x=centers[:,0], y=centers[:,1], mode='markers', marker=dict(symbol='x', size=12))
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.header("Data CSV Files")
        st.write(list(DATA_DIR.glob("*.csv")))

if __name__ == "__main__":
    main()
