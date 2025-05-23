import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria Analytical Dashboard (CSV)", layout="wide")

# ── Data directory ─────────────────────────────────────────────────────────
DATA_DIR = Path("data_csvs")

# ── Utility: load all versioned CSVs for a topic ────────────────────────────
# ── Utility: load all versioned CSVs for a topic ────────────────────────────
def load_topic_csvs(topic: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Reads files matching either
      <topic>*.csv
    or
      <topic>_topic*.csv
    in data_dir, sorts them, concatenates, and returns a DataFrame.
    Raises FileNotFoundError if none or all are empty.
    """
    # build two candidate patterns
    pats = [f"{topic}*.csv"]
    if not topic.endswith("_topic"):
        pats.append(f"{topic}_topic*.csv")
    else:
        base = topic[:-6]  # strip “_topic”
        pats.append(f"{base}*.csv")

    # collect & dedupe
    files = []
    for pat in pats:
        files.extend(data_dir.glob(pat))
    # unique & sorted
    files = sorted({f for f in files})
    if not files:
        raise FileNotFoundError(f"No CSVs found for topic '{topic}' in {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"All CSVs for '{topic}' are empty")
    return pd.concat(dfs, ignore_index=True)

# ── KPI loader ─────────────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    """
    Load and compute KPI table from CSVs.
    """
    # Load source tables
    watch = load_topic_csvs("watch")
    purchase = load_topic_csvs("purchase_events")
    streams = load_topic_csvs("streams")
    partners = load_topic_csvs("partners")
    games = load_topic_csvs("games")

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
        purchase[purchase['category']=='game']
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

    # Combine all KPI tables
    return pd.concat([
        viewed[['kpi','label','value']],
        purchased[['kpi','label','value']],
        streamer[['kpi','label','value']],
        best_games[['kpi','label','value']],
        sg[['kpi','label','value']]
    ], ignore_index=True)

# ── Yearly watch rank ──────────────────────────────────────────────────────
@st.cache_data
def update_year(y: int):
    # note: use "watch_topic" not just "watch"
    watch = load_topic_csvs("watch")
    watch['date'] = pd.to_datetime(watch['date'], errors='coerce')
    df_year = watch[watch['date'].dt.year == y]
    if df_year.empty:
        st.warning(f"No data for year {y}")
        return px.choropleth(
            pd.DataFrame({'country':[], 'watch_hours':[]}),
            locations='country', color='watch_hours'
        )

    grouped = (
        df_year.groupby('country')['length']
               .sum()
               .reset_index()
               .rename(columns={'length':'watch_hours'})
    )
    grouped['pct_rank'] = grouped['watch_hours'].rank(pct=True)

    fig = px.choropleth(
        grouped,
        locations='country', locationmode='country names',
        color='pct_rank', hover_data={'watch_hours':':.1f','pct_rank':':.2f'},
        color_continuous_scale='Viridis', range_color=(0,1)
    )
    fig.update_layout(title=f"Yearly Watch Rank: {y}")
    return fig

# ── Buyer segments ─────────────────────────────────────────────────────────
@st.cache_data
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    """
    Perform MCA + KMeans on trophy purchasers.
    Returns coords_df, full_df, summary_df, centers.
    """
    purchase = load_topic_csvs("purchase_events")
    customers = load_topic_csvs("customers")
    df = purchase.merge(customers, on='customer_id')
    df = df[df['product_name']=='Authentic Mahiman Trophy'].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # Compute age and drop missing
    df['age'] = ((pd.to_datetime('today') - pd.to_datetime(df['birthday'], errors='coerce'))
                 .dt.days // 365)
    df = df[['customer_id','age','gender','region']].dropna().head(sample_limit)
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), []

    # Age bins
    df['age_bin'] = pd.cut(
        df['age'], bins=range(10,81,5),
        labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False
    )
    df = df.dropna(subset=['age_bin'])

    # MCA transformation
    coords_arr = prince.MCA(n_components=2, random_state=42).fit_transform(df[['age_bin','gender','region']].astype(str))
    coords_df = pd.DataFrame(coords_arr, columns=['Dim1','Dim2'], index=df.index)

    # KMeans clustering
    km = KMeans(n_clusters=k, random_state=42)
    df_clean = coords_df.dropna()
    if df_clean.empty:
        return coords_df, df, pd.DataFrame(), []
    df_clean['cluster'] = km.fit_predict(df_clean[['Dim1','Dim2']])

    coords_df['cluster'] = df_clean['cluster']
    df['cluster'] = coords_df['cluster'].astype(int)

    summary = df_clean.groupby('cluster').agg(
        size=('cluster','count'), avg_dim1=('Dim1','mean')
    ).reset_index()
    centers = km.cluster_centers_
    return coords_df, df, summary, centers

# ── Main app UI ────────────────────────────────────────────────────────────
def main():
    st.title("Euphoria Analytical Dashboard (CSV)")
    tabs = st.tabs(["KPIs","Yearly Rank","Buyer Segments","Data Files"])

    # KPIs
    with tabs[0]:
        st.header("Key Performance Indicators")
        try:
            df_kpi = load_kpis()
        except FileNotFoundError as e:
            st.warning(str(e))
            df_kpi = pd.DataFrame()
        if df_kpi.empty:
            st.info("No KPI data. Populate `data_csvs/` and restart.")
        else:
            choice = st.selectbox("Select KPI", df_kpi['kpi'].unique())
            st.dataframe(df_kpi[df_kpi['kpi']==choice])

    # Yearly Rank
    with tabs[1]:
        st.header("Yearly Watch Rank")
        years = list(range(datetime.now().year, datetime.now().year-10, -1))
        year = st.selectbox("Select Year", years)
        fig = update_year(year)
        st.plotly_chart(fig, use_container_width=True)

    # Buyer Segments
    with tabs[2]:
        st.header("Buyer Segments (Authentic Mahiman Trophy)")
        if st.button("Compute Segments"):
            with st.spinner("Running MCA + KMeans..."):
                coords, full, summary, centers = compute_trophy_segments()
            if full.empty:
                st.warning("No trophy-purchase data found. Ensure CSVs loaded.")
            else:
                st.dataframe(summary)
                fig = px.scatter(coords, x='Dim1', y='Dim2', color='cluster', title="Trophy Buyer Segments")
                if centers:
                    fig.add_scatter(x=centers[:,0], y=centers[:,1], mode='markers', marker=dict(symbol='x', size=12))
                st.plotly_chart(fig, use_container_width=True)

    # Data Files
    with tabs[3]:
        st.header("Available CSV Files")
        files = sorted([p.name for p in DATA_DIR.glob("*.csv")])
        if files:
            st.write(files)
        else:
            st.info("No CSV files found in `data_csvs/`.")

if __name__ == "__main__":
    main()
