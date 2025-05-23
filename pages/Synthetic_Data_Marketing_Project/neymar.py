import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria CSV Dashboard", layout="wide")
DATA_DIR = Path(__file__).parent / "data_csvs"

# ── UTILITY: load any CSVs matching base name ─────────────────────────────────
def load_topic_csvs(topic: str) -> pd.DataFrame:
    """
    Finds all files data_csvs/<base>*\.csv where base = topic or topic[:-6]
    (to handle topics named like 'watch_topic' vs files named 'watch.csv').
    Returns the concatenated DataFrame (or empty DF if none/non-empty).
    """
    # strip trailing "_topic" if present:
    base = topic[:-6] if topic.endswith("_topic") else topic
    files = sorted(DATA_DIR.glob(f"{base}*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# ── 1) KPI loader ─────────────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    watch    = load_topic_csvs("watch_topic")
    purchase = load_topic_csvs("purchase_events_topic")
    streams  = load_topic_csvs("streams_topic")
    partners = load_topic_csvs("partners")
    games    = load_topic_csvs("games_topic")

    if watch.empty or purchase.empty or streams.empty or partners.empty or games.empty:
        return pd.DataFrame()

    # Top 10 viewed countries
    viewed = (
        watch.groupby("country")["length"]
             .sum()
             .nlargest(10)
             .reset_index()
             .rename(columns={"country":"label","length":"value"})
    )
    viewed["kpi"] = "Top 10 Viewed Countries"

    # Top 8 purchased products
    purchased = (
        purchase.groupby("product_name")
                .size()
                .nlargest(8)
                .reset_index(name="value")
                .rename(columns={"product_name":"label"})
    )
    purchased["kpi"] = "Top 8 Purchased Products"

    # Top 10 streamer performance
    sp = streams.merge(partners, on="partner_id", how="left")
    sp["score"] = (sp.viewers_total/(sp.length.replace(0,1))) * sp.comments_total
    streamer = (
        sp.groupby("screen_name")["score"]
          .sum()
          .nlargest(10)
          .reset_index()
          .rename(columns={"screen_name":"label","score":"value"})
    )
    streamer["value"] = streamer["value"].round(2)
    streamer["kpi"] = "Top 10 Streamer Performance"

    # Top 2 best‐selling games
    best_games = (
        purchase[purchase.category=="game"]
        .groupby("product_name")
        .size()
        .nlargest(2)
        .reset_index(name="value")
        .rename(columns={"product_name":"label"})
    )
    best_games["kpi"] = "Top 2 Best-Selling Games"

    # Top 2 most‐streamed games
    sg = (
        streams.merge(games, on="game_id", how="left")
               .groupby("title")
               .size()
               .nlargest(2)
               .reset_index(name="value")
               .rename(columns={"title":"label"})
    )
    sg["kpi"] = "Top 2 Most-Streamed Games"

    return pd.concat(
        [viewed, purchased, streamer, best_games, sg],
        ignore_index=True
    )

# ── 2) Yearly Watch Rank ──────────────────────────────────────────────────────
@st.cache_data
def update_year(year: int):
    watch = load_topic_csvs("watch")
    if watch.empty:
        return px.choropleth(pd.DataFrame(columns=["country","watch_hours"]),
                             locations="country", color="watch_hours")

    watch["date"] = pd.to_datetime(watch["date"], errors="coerce")
    df_year = watch[watch["date"].dt.year==year]
    if df_year.empty:
        return px.choropleth(pd.DataFrame(columns=["country","watch_hours"]),
                             locations="country", color="watch_hours")

    grouped = (
        df_year.groupby("country")["length"]
               .sum()
               .reset_index()
               .rename(columns={"length":"watch_hours"})
    )
    grouped["pct_rank"] = grouped["watch_hours"].rank(pct=True)

    fig = px.choropleth(
        grouped,
        locations="country",
        locationmode="country names",
        color="pct_rank",
        hover_data=["watch_hours","pct_rank"],
        color_continuous_scale="Viridis",
        range_color=(0,1),
    )
    fig.update_layout(title=f"Yearly Watch Rank: {year}")
    return fig

# ── 3) MCA + KMeans on Trophy Purchasers ──────────────────────────────────────
@st.cache_data
def compute_trophy_segments(sample_limit: int=50000, k: int=4):
    purchase  = load_topic_csvs("purchase_events")
    customers = load_topic_csvs("customers")

    if purchase.empty or customers.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    df = purchase.merge(customers, on="customer_id", how="inner")
    df = df[df.product_name=="Authentic Mahiman Trophy"].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # compute age & drop NA
    df["age"] = ((pd.to_datetime("today") -
                  pd.to_datetime(df["birthday"], errors="coerce"))
                 .dt.days // 365)
    df = df[["customer_id","age","gender","region"]].dropna().head(sample_limit)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # age bins
    df["age_bin"] = pd.cut(df["age"],
                          bins=range(10,81,5),
                          labels=[f"{i}-{i+4}" for i in range(10,80,5)],
                          right=False)
    df = df.dropna(subset=["age_bin"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # MCA
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df[["age_bin","gender","region"]].astype(str))
    coords = pd.DataFrame(coords_arr, columns=["Dim1","Dim2"], index=df.index)

    # KMeans
    clean = coords.dropna()
    if clean.empty:
        return coords, df, pd.DataFrame(), []
    km = KMeans(n_clusters=k, random_state=42).fit(clean)
    clean["cluster"] = km.labels_

    coords["cluster"] = clean["cluster"]
    df["cluster"]     = coords["cluster"].astype(int)

    summary = (
        clean.groupby("cluster")
             .agg(size=("cluster","count"), avg_dim1=("Dim1","mean"))
             .reset_index()
    )
    return coords, df, summary, km.cluster_centers_

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.title("Euphoria CSV Analytics")

    tabs = st.tabs(["KPIs","Yearly Rank","Segments","Data Files"])

    with tabs[0]:
        st.header("Key Performance Indicators")
        kpis = load_kpis()
        if kpis.empty:
            st.warning("No KPI data: please drop your CSVs under data_csvs/")
        else:
            choice = st.selectbox("Choose KPI", kpis["kpi"].unique())
            st.dataframe(kpis[kpis["kpi"]==choice])

    with tabs[1]:
        st.header("Yearly Watch Rank")
        year = st.selectbox("Year", list(range(datetime.now().year, datetime.now().year-10, -1)))
        fig = update_year(year)
        st.plotly_chart(fig, use_container_width=True)
    # Segments
    with tabs[2]:
        st.header("Buyer Segments (Authentic Mahiman Trophy)")
        if st.button("Compute Segments"):
            coords, full, summary, centers = compute_trophy_segments()
            if full.empty:
                st.warning("No trophy purchasers found in your CSVs.")
            else:
                st.dataframe(summary)
                # only plot if coords has the right columns
                if {"Dim1","Dim2","cluster"}.issubset(coords.columns) and not coords.empty:
                    dfp = coords.copy()
                    dfp["cluster"] = dfp["cluster"].astype(str)
                    fig = px.scatter(dfp, x="Dim1", y="Dim2", color="cluster")
                    if len(centers):
                        fig.add_scatter(
                            x=centers[:,0], y=centers[:,1],
                            mode="markers",
                            marker=dict(symbol="x", size=12)
                        )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Segment coords incomplete; check your CSVs.")

    with tabs[3]:
        st.header("Detected CSV files")
        st.write([p.name for p in DATA_DIR.glob("*.csv")])

if __name__=="__main__":
    main()
