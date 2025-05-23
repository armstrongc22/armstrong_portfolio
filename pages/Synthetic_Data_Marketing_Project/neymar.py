import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria CSV Dashboard", layout="wide")
HERE     = Path(__file__).parent
DATA_DIR = HERE / "data_csvs"
TROPHY_DIR = DATA_DIR / "trophy"
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

@st.cache_data
def load_trophy_customers() -> pd.DataFrame:
    parts = sorted(TROPHY_DIR.glob("trophy_customers_part*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"No trophy CSVs in {TROPHY_DIR}; expected trophy_customers_part1.csv, part2.csv"
        )
    dfs = []
    for p in parts:
        df = pd.read_csv(p)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("All trophy CSVs are empty")
    return pd.concat(dfs, ignore_index=True)

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
import numpy as np


@st.cache_data
def compute_trophy_segments(n_clusters: int = 4):
    df = load_trophy_customers()
    # compute age from birthday
    df["age"] = (
            pd.to_datetime("today")
            .tz_localize(None)
            .sub(pd.to_datetime(df["birthday"], errors="coerce"))
            .dt.days
            // 365
    )
    df = df.dropna(subset=["age", "gender", "region"])
    # bin age
    df["age_bin"] = pd.cut(
        df["age"],
        bins=range(10, 81, 5),
        labels=[f"{i}-{i + 4}" for i in range(10, 80, 5)],
        right=False
    )
    df = df.dropna(subset=["age_bin"])

    # MCA to 2 dims
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df[["age_bin", "gender", "region"]].astype(str))
    coords = pd.DataFrame(coords_arr, columns=["Dim1", "Dim2"], index=df.index)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(coords)
    coords["cluster"] = labels
    df["cluster"] = labels

    # summary
    summary = (
        coords.groupby("cluster")
        .agg(size=("cluster", "count"), avg_dim1=("Dim1", "mean"))
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
    # ── Buyer Segments tab ────────────────────────────────────────────────────
    # ── Tab 2: Buyer Segments
    # … after your KPI and Yearly‐rank tabs …
    with tabs[2]:
        st.header("Buyer Segments (Authentic Mahiman Trophy)")
        if st.button("Compute Segments"):
            try:
                coords, full_df, summary, centers = compute_trophy_segments()
            except FileNotFoundError as e:
                st.error(str(e))
            else:
                if full_df.empty:
                    st.warning("No trophy‐customer records found.")
                else:
                    st.dataframe(summary)
                    fig = px.scatter(
                        coords,
                        x="Dim1",
                        y="Dim2",
                        color="cluster",
                        title="Trophy Buyer Segments"
                    )
                    fig.add_scatter(
                        x=centers[:, 0],
                        y=centers[:, 1],
                        mode="markers",
                        marker=dict(symbol="x", size=12, color="black"),
                        name="Centroids"
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__=="__main__":
    main()
