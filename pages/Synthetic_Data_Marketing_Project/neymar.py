import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Build correct folders relative to this script ───────────────────────────
HERE       = Path(__file__).resolve().parent
DATA_DIR   = HERE / "data_csvs"
TROPHY_DIR = DATA_DIR / "trophy"

# ── Helpers ──────────────────────────────────────────────────────────────────
def try_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"❌ missing file: {path.relative_to(HERE)}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ failed to read {path.name}: {e}")
        return pd.DataFrame()
    return df

def list_files(folder: Path, pattern: str):
    return sorted(folder.glob(pattern))

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Euphoria CSV Dashboard (fixed paths)")

    tabs = st.tabs(["1️⃣ Disk & Heads","2️⃣ KPIs","3️⃣ Yearly Map","4️⃣ Segments"])

    # Tab 1: show exactly what files we see
    with tabs[0]:
        st.header("Files under `data_csvs/`")
        for p in list_files(DATA_DIR, "*.csv"):
            st.write(f"- {p.name} ({p.stat().st_size//1024} KB)")
        st.write("Files under `data_csvs/trophy/`")
        for p in list_files(TROPHY_DIR, "*.csv"):
            st.write(f"- {p.name} ({p.stat().st_size//1024} KB)")

        st.markdown("---")
        st.header("Preview first 3 rows of each")
        for fn in ["watch","streams","purchase_events","partners","games","merch"]:
            p = DATA_DIR / f"{fn}.csv"
            st.subheader(p.name)
            st.dataframe(try_read(p).head(3))

        st.subheader("trophy/*.csv")
        for p in list_files(TROPHY_DIR, "trophy_customers_part*.csv"):
            st.write(p.name)
            st.dataframe(try_read(p).head(3))

    # Tab 2: KPIs
    with tabs[1]:
        st.header("Key Performance Indicators")
        # load each
        watch    = try_read(DATA_DIR/"watch.csv")
        streams  = try_read(DATA_DIR/"streams.csv")
        purchase = try_read(DATA_DIR/"purchase_events.csv")
        partners = try_read(DATA_DIR/"partners.csv")
        games    = try_read(DATA_DIR/"games.csv")
        merch    = try_read(DATA_DIR/"merch.csv")

        if any(df.empty for df in (watch,streams,purchase,partners,games,merch)):
            st.warning("One or more source tables is empty — check Disk & Heads.")
        else:
            # Top 10 viewed
            st.subheader("Top 10 Viewed Countries")
            tv = watch.groupby("country")["length"].sum().nlargest(10).reset_index()
            st.dataframe(tv)

            # Top 8 purchased products
            st.subheader("Top 8 Purchased Products")
            tp = (purchase.groupby("product_name")
                           .size()
                           .nlargest(8)
                           .reset_index(name="count"))
            st.dataframe(tp)

            # Top 10 streamer performance
            st.subheader("Top 10 Streamer Performance")
            sp = streams.merge(partners, on="partner_id", how="left")
            sp["score"] = (sp.viewers_total / sp.length.replace(0,1)) * sp.comments_total
            top_str = (sp.groupby("screen_name")["score"]
                         .sum()
                         .nlargest(10)
                         .reset_index())
            top_str["score"] = top_str["score"].round(2)
            st.dataframe(top_str)

            # Top 2 best‐selling games
            st.subheader("Top 2 Best‐Selling Games")
            bg = (purchase[purchase.category=="game"]
                  .groupby("product_name")
                  .size()
                  .nlargest(2)
                  .reset_index(name="count"))
            st.dataframe(bg)

            # Top 2 most‐streamed games
            st.subheader("Top 2 Most‐Streamed Games")
            ms = (streams.merge(games, on="game_id", how="left")
                   .groupby("title").size()
                   .nlargest(2)
                   .reset_index(name="count"))
            st.dataframe(ms)

    # Tab 3: Yearly Map
    with tabs[2]:
        st.header("Yearly Watch Choropleth")
        watch = try_read(DATA_DIR/"watch.csv")
        if watch.empty:
            st.warning("watch.csv is empty.")
        else:
            watch["date"] = pd.to_datetime(watch["date"], errors="coerce")
            year = st.selectbox("Select Year",
                                list(range(datetime.now().year, datetime.now().year - 10, -1)))
            dfy = watch[watch["date"].dt.year == year]
            if dfy.empty:
                st.warning(f"No data for {year}")
            else:
                grp = (dfy.groupby("country")["length"]
                       .sum()
                       .reset_index(name="watch_hours"))
                grp["pct_rank"] = grp["watch_hours"].rank(pct=True)
                fig = px.choropleth(grp,
                                    locations="country",
                                    color="pct_rank",
                                    locationmode="country names",
                                    hover_data=["watch_hours","pct_rank"],
                                    range_color=(0,1))
                fig.update_layout(title=f"Yearly Watch Rank: {year}")
                st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Trophy Segments
    with tabs[3]:
        st.header("Buyer Segments: Authentic Mahiman Trophy")
        parts = list_files(TROPHY_DIR, "trophy_customers_part*.csv")
        if not parts:
            st.error("No trophy CSVs under data_csvs/trophy/")
        else:
            df = pd.concat([try_read(p) for p in parts], ignore_index=True)
            st.write(f"Loaded {len(df)} total trophy rows.")
            if st.button("Compute Segments") and not df.empty:
                # MCA + KMeans
                df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
                df["age"] = (pd.Timestamp.now() - df["birthday"]).dt.days // 365
                clean = df.dropna(subset=["age","gender","region"])
                clean["age_bin"] = pd.cut(clean["age"],
                                          bins=range(10,81,5),
                                          labels=[f"{i}-{i+4}" for i in range(10,80,5)],
                                          right=False).astype(str)
                coords = prince.MCA(n_components=2, random_state=42)\
                             .fit_transform(clean[["age_bin","gender","region"]])
                coords = pd.DataFrame(coords, columns=["Dim1","Dim2"], index=clean.index)
                coords["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(coords)
                summary = (coords.groupby("cluster")
                              .agg(size=("cluster","count"),
                                   avg_dim1=("Dim1","mean"))
                              .reset_index())
                st.subheader("Segment summary")
                st.dataframe(summary)
                fig = px.scatter(coords, x="Dim1", y="Dim2", color="cluster")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
