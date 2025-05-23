import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Euphoria CSV Analytical Dashboard", layout="wide")

# ── Where your CSVs live ───────────────────────────────────────────────────────
DATA_DIR = Path("data_csvs")
TROPHY_DIR = DATA_DIR / "trophy"

# ── Helper to glob & concat versioned files ────────────────────────────────────
def load_csv(name: str, folder: Path = DATA_DIR) -> pd.DataFrame:
    """
    Find all files matching name*.csv in folder, concat, drop empty.
    Raise FileNotFoundError if none or all empty.
    """
    files = sorted(folder.glob(f"{name}*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found for '{name}' in {folder}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            pass
    if not dfs:
        raise FileNotFoundError(f"All {name} CSVs are empty in {folder}")
    return pd.concat(dfs, ignore_index=True)

# ── KPI loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_kpis() -> pd.DataFrame:
    # load each topic dump
    watch    = load_csv("watch")
    streams  = load_csv("streams")
    purchase = load_csv("purchase_events")
    partners = load_csv("partners")
    games    = load_csv("games")
    merch    = load_csv("merch")

    # Top 10 Viewed Countries
    viewed = (
        watch.groupby("country")["length"]
             .sum()
             .nlargest(10)
             .reset_index(name="value")
             .rename(columns={"country":"label"})
    )
    viewed["kpi"] = "Top 10 Viewed Countries"

    # Top 8 Purchased Products
    purchased = (
        purchase.groupby("product_name")
                .size()
                .nlargest(8)
                .reset_index(name="value")
                .rename(columns={"product_name":"label"})
    )
    purchased["kpi"] = "Top 8 Purchased Products"

    # Top 10 Streamer Performance
    sp = streams.merge(partners, on="partner_id")
    sp["score"] = (sp.viewers_total / sp.length.replace(0,1)) * sp.comments_total
    streamer = (
        sp.groupby("screen_name")["score"]
          .sum()
          .nlargest(10)
          .reset_index(name="value")
          .rename(columns={"screen_name":"label"})
    )
    streamer["kpi"] = "Top 10 Streamer Performance"

    # Top 2 Best‐Selling Games
    best_games = (
        purchase[purchase.category=="game"]
               .groupby("product_name")
               .size()
               .nlargest(2)
               .reset_index(name="value")
               .rename(columns={"product_name":"label"})
    )
    best_games["kpi"] = "Top 2 Best-Selling Games"

    # Top 2 Most-Streamed Games
    sg = (
        streams.merge(games, on="game_id")
               .groupby("title")
               .size()
               .nlargest(2)
               .reset_index(name="value")
               .rename(columns={"title":"label"})
    )
    sg["kpi"] = "Top 2 Most-Streamed Games"

    # combine
    return pd.concat(
        [viewed, purchased, streamer, best_games, sg],
        ignore_index=True
    )

# ── Yearly Watch Map ──────────────────────────────────────────────────────────
@st.cache_data
def update_year(year: int):
    watch = load_csv("watch")
    watch["date"] = pd.to_datetime(watch["date"], errors="coerce")
    df_year = watch[watch["date"].dt.year == year]
    if df_year.empty:
        st.warning(f"No watch data for {year}")
        return px.choropleth(pd.DataFrame(columns=["country","watch_hours"]),
                             locations="country", color="watch_hours")

    grouped = (
        df_year.groupby("country")["length"]
               .sum()
               .reset_index(name="watch_hours")
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

# ── MCA + KMeans on Trophy Customers ──────────────────────────────────────────
@st.cache_data
def compute_trophy_segments(n_clusters: int = 4):
    # read your two halves of trophy_customers
    parts = sorted(TROPHY_DIR.glob("trophy_customers_part*.csv"))
    if not parts:
        raise FileNotFoundError("Place trophy_customers_part1.csv & part2.csv under data_csvs/trophy/")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if df.empty:
        # nothing to do
        return pd.DataFrame(), df, pd.DataFrame(), []

    # compute age
    df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
    df["age"] = (pd.Timestamp.now() - df["birthday"]).dt.days // 365
    df = df.dropna(subset=["age","gender","region"])

    # bin
    df["age_bin"] = pd.cut(
        df["age"],
        bins=range(10,81,5),
        labels=[f"{i}-{i+4}" for i in range(10,80,5)],
        right=False
    ).astype(str)
    df = df[df["age_bin"] != "nan"]

    # MCA
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df[["age_bin","gender","region"]])
    coords = pd.DataFrame(coords_arr, columns=["Dim1","Dim2"], index=df.index)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42)
    coords["cluster"] = km.fit_predict(coords)

    # attach back
    df["cluster"] = coords["cluster"].astype(int)

    # summary
    summary = (
        coords.groupby("cluster")
              .agg(size=("cluster","count"), avg_dim1=("Dim1","mean"))
              .reset_index()
    )

    return coords, df, summary, km.cluster_centers_

# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    st.title("Euphoria CSV Analytical Dashboard")

    tabs = st.tabs(["KPIs","Yearly Rank","Segments","Data Files"])

    # KPIs
    with tabs[0]:
        st.header("Key Performance Indicators")
        try:
            kpis = load_kpis()
        except FileNotFoundError as e:
            st.warning(str(e))
            return
        choice = st.selectbox("Choose KPI", kpis["kpi"].unique())
        st.dataframe(kpis[kpis["kpi"]==choice].reset_index(drop=True))

    # Yearly Map
    with tabs[1]:
        st.header("Yearly Watch Rank")
        year = st.selectbox("Year",
            list(range(datetime.now().year, datetime.now().year-10, -1))
        )
        fig = update_year(year)
        st.plotly_chart(fig, use_container_width=True)

    # Segments
    with tabs[2]:
        st.header("Buyer Segments (Authentic Mahiman Trophy)")
        if st.button("Compute Segments"):
            try:
                coords, df_seg, summary, centers = compute_trophy_segments()
            except FileNotFoundError as e:
                st.error(str(e))
                return
            if df_seg.empty:
                st.warning("No trophy customers found in your CSVs.")
                return

            st.dataframe(summary)
            fig = px.scatter(coords, x="Dim1", y="Dim2", color="cluster",
                             title="MCA + KMeans on Trophy Buyers")
            fig.add_scatter(x=centers[:,0], y=centers[:,1],
                            mode="markers", marker=dict(symbol="x",size=12),
                            name="Centroids")
            st.plotly_chart(fig, use_container_width=True)

    # list files
    with tabs[3]:
        st.header("Data Files")
        st.write("`data_csvs/`:", [p.name for p in DATA_DIR.iterdir()])
        st.write("`data_csvs/trophy/`:", [p.name for p in TROPHY_DIR.iterdir()])

if __name__=="__main__":
    main()
