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

def load_trophy_customers() -> pd.DataFrame:
    """
    Load all files trophy_customers_part*.csv under data_csvs/trophy/,
    concat them, and return the full DataFrame of customers.
    """
    parts = sorted(TROPHY_DIR.glob("trophy_customers_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"No trophy CSV parts in {TROPHY_DIR}")
    dfs = []
    for f in parts:
        df = pd.read_csv(f)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("All trophy customer parts are empty")
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
def compute_trophy_segments(sample_limit: int = 50000, k: int = 4):
    """
    Loads the trophy‐customer CSVs that you split into two halves under
    data_csvs/trophy/, merges with the full purchase_events, runs MCA + KMeans.
    Returns coords_df, full_df, summary_df, centers (ndarray or empty).
    """
    # 1) Load the two halves of your trophy customer list
    trophy_dir = DATA_DIR / "trophy"
    parts = sorted(trophy_dir.glob("trophy_customers_part*.csv"))
    if not parts:
        # no trophy CSVs found at all
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), np.empty((0,2))

    cust = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if cust.empty:
        return pd.DataFrame(), cust, pd.DataFrame(), np.empty((0,2))

    # 2) Load the full purchases (you sampled) and filter for “Authentic Mahiman Trophy”
    purchase = pd.read_csv(DATA_DIR / "purchase_events.csv")
    trophy_purch = purchase[purchase.product_name == "Authentic Mahiman Trophy"]
    if trophy_purch.empty:
        return pd.DataFrame(), trophy_purch, pd.DataFrame(), np.empty((0,2))

    # 3) Join them back to get demographics
    df = trophy_purch.merge(cust, on="customer_id", how="inner")
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), np.empty((0,2))

    # 4) Derive age bins
    df["age"] = ((pd.to_datetime("today") - pd.to_datetime(df["birthday"], errors="coerce"))
                  .dt.days // 365)
    df = df.dropna(subset=["age","gender","region"]).head(sample_limit)
    df["age_bin"] = pd.cut(df["age"], bins=range(10,81,5),
                          labels=[f"{i}-{i+4}" for i in range(10,80,5)], right=False)
    df = df.dropna(subset=["age_bin"])
    if df.empty:
        return pd.DataFrame(), df, pd.DataFrame(), np.empty((0,2))

    # 5) MCA on age_bin, gender, region
    df_mca = df[["age_bin","gender","region"]].astype(str)
    mca = prince.MCA(n_components=2, random_state=42)
    coords_arr = mca.fit_transform(df_mca)
    coords = pd.DataFrame(coords_arr, columns=["Dim1","Dim2"], index=df.index)

    # 6) Guard against zero rows
    coords_clean = coords.dropna()
    if coords_clean.shape[0] == 0:
        return coords, df, pd.DataFrame(), np.empty((0,2))

    # 7) KMeans
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(coords_clean)
    coords_clean["cluster"] = km.labels_

    # 8) Map clusters back
    coords["cluster"] = coords_clean["cluster"]
    df["cluster"]    = coords["cluster"].fillna(-1).astype(int)

    # 9) Summary
    summary = coords_clean.groupby("cluster").agg(
        size=("cluster","count"),
        avg_dim1=("Dim1","mean"),
    ).reset_index()

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
    with tabs[2]:
        st.header("Buyer Segments (Authentic Mahiman Trophy)")

        if st.button("Compute Segments"):
            coords, full, summary, centers = compute_trophy_segments()

            if full.empty or summary.empty:
                st.warning(
                    "No trophy purchasers found – check that you split and placed your CSVs in data_csvs/trophy/")
            else:
                st.dataframe(summary)
                fig = px.scatter(coords, x="Dim1", y="Dim2", color="cluster")
                if centers.size:
                    fig.add_scatter(
                        x=centers[:, 0], y=centers[:, 1],
                        mode="markers",
                        marker=dict(symbol="x", size=12)
                    )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.header("Detected CSV files")
        st.write([p.name for p in DATA_DIR.glob("*.csv")])

if __name__=="__main__":
    main()
