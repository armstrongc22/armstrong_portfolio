import streamlit as st
import pandas as pd
import plotly.express as px
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data_csvs")
TROPHY_DIR = DATA_DIR / "trophy"

# ── Helpers ──────────────────────────────────────────────────────────────────
def try_read(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()
    return df

def glob_read(pattern: str, folder: Path) -> list[Path]:
    return sorted(folder.glob(pattern))

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Euphoria CSV Dashboard — Debug Edition")
    tabs = st.tabs(["Disk & Heads","KPIs","Yearly Map","Segments"])

    # --- Tab 1: list & head
    with tabs[0]:
        st.header("1) Files on disk")
        st.write("**data_csvs/**")
        for p in DATA_DIR.glob("*.csv"):
            st.write(f"- {p.name}  ({p.stat().st_size//1024} KB)")
        st.write("**data_csvs/trophy/**")
        for p in TROPHY_DIR.glob("*.csv"):
            st.write(f"- {p.name}  ({p.stat().st_size//1024} KB)")

        st.markdown("---")
        st.write("2) Preview heads")
        for fname in ["watch","streams","purchase_events","partners","games","merch"]:
            path = DATA_DIR / f"{fname}.csv"
            st.subheader(path.name)
            df = try_read(path)
            st.dataframe(df.head(3))

        st.subheader("trophy/*.csv")
        for p in TROPHY_DIR.glob("trophy_customers_part*.csv"):
            st.write(p.name)
            st.dataframe(try_read(p).head(3))

    # --- Tab 2: KPIs
    with tabs[1]:
        st.header("Key Performance Indicators")
        try:
            watch    = try_read(DATA_DIR/"watch.csv")
            streams  = try_read(DATA_DIR/"streams.csv")
            purchase = try_read(DATA_DIR/"purchase_events.csv")
            partners = try_read(DATA_DIR/"partners.csv")
            games    = try_read(DATA_DIR/"games.csv")
            merch    = try_read(DATA_DIR/"merch.csv")
            # ensure non-empty
            if any(df.empty for df in [watch,streams,purchase,partners,games,merch]):
                st.warning("One or more source tables is empty—see Disk & Heads.")
            else:
                # Top viewed
                viewed = watch.groupby("country")["length"].sum().nlargest(10)
                st.write("**Top 10 Viewed Countries**")
                st.dataframe(viewed.reset_index())

                # Top purchased
                purchased = purchase.groupby("product_name").size().nlargest(8)
                st.write("**Top 8 Purchased Products**")
                st.dataframe(purchased.reset_index(name="count"))

                # And so on...
        except Exception as e:
            st.error(f"KPI error: {e}")

    # --- Tab 3: Yearly Watch Map
    with tabs[2]:
        st.header("Yearly Watch Map")
        try:
            watch = try_read(DATA_DIR/"watch.csv")
            if watch.empty:
                st.warning("watch.csv is empty.")
            else:
                watch["date"] = pd.to_datetime(watch["date"], errors="coerce")
                year = st.selectbox("Year", list(range(datetime.now().year, datetime.now().year-10,-1)))
                dfy = watch[watch["date"].dt.year==year]
                if dfy.empty:
                    st.warning(f"No watch rows for {year}.")
                else:
                    grp = dfy.groupby("country")["length"].sum().reset_index(name="watch_hours")
                    grp["pct_rank"] = grp["watch_hours"].rank(pct=True)
                    fig = px.choropleth(grp, locations="country", color="pct_rank",
                                        locationmode="country names", range_color=(0,1))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Yearly error: {e}")

    # --- Tab 4: Trophy Segments
    with tabs[3]:
        st.header("MCA + KMeans on Trophy Buyers")
        parts = glob_read("trophy_customers_part*.csv", TROPHY_DIR)
        if not parts:
            st.error("No trophy CSVs found under data_csvs/trophy/")
            return
        dfs = [try_read(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        st.write(f"Loaded {len(df)} total trophy rows.")
        if st.button("Run Segments") and not df.empty:
            try:
                df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
                df["age"] = (pd.Timestamp.now() - df["birthday"]).dt.days // 365
                clean = df.dropna(subset=["age","gender","region"])
                clean["age_bin"] = pd.cut(clean["age"], bins=range(10,81,5),
                                          labels=[f"{i}-{i+4}" for i in range(10,80,5)],
                                          right=False).astype(str)
                x = prince.MCA(n_components=2, random_state=42)\
                        .fit_transform(clean[["age_bin","gender","region"]])
                coords = pd.DataFrame(x, columns=["Dim1","Dim2"], index=clean.index)
                coords["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(coords)
                summary = coords.groupby("cluster")\
                                .agg(size=("cluster","count"), avg_dim1=("Dim1","mean"))\
                                .reset_index()
                st.dataframe(summary)
                fig = px.scatter(coords, x="Dim1", y="Dim2", color="cluster")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Segmentation error: {e}")

if __name__=="__main__":
    main()
