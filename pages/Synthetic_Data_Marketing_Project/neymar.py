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

@st.cache_data
def load_trophy_customers():
    parts = sorted(TROPHY_DIR.glob("trophy_customers_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"No files matching trophy_customers_part*.csv in {TROPHY_DIR}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df['birthday'] = pd.to_datetime(df['birthday'], errors='coerce')
    df = df.dropna(subset=['birthday']).copy()
    df['age'] = ((pd.Timestamp.now() - df['birthday']).dt.days // 365).astype(int)
    # 5-year bins
    bins   = list(range(10,81,5))
    labels = [f"{i}-{i+4}" for i in range(10,80,5)]
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    # top-10 regions + Other
    top10 = df['region'].value_counts().nlargest(10).index
    df['region2'] = df['region'].where(df['region'].isin(top10), other='Other')
    return df

@st.cache_data
def compute_mca_and_segments(df: pd.DataFrame, col1: str, col2: str, n_clusters: int = 4):
    X = df[[col1, col2]].dropna().astype(str)
    mca = prince.MCA(n_components=2, random_state=42).fit(X)
    coords = mca.row_coordinates(X)
    coords.columns = ['Dim1','Dim2']
    coords.index   = X.index
    # K-Means segments
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    coords['segment'] = km.labels_
    # inertia percentages
    eigs = mca.eigenvalues_
    total = eigs.sum()
    inertia1 = eigs[0]/total*100
    inertia2 = eigs[1]/total*100
    return coords, inertia1, inertia2

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
        st.title("🏆 Euphoria Trophy Purchasers: MCA + Segments")
        df = load_trophy_customers()
        st.write(f"Loaded **{len(df):,}** trophy customers")
        st.markdown("---")

        pairs = [
            ("age_bin", "gender"),
            ("age_bin", "region2"),
            ("gender", "region2"),
        ]
        for col1, col2 in pairs:
            st.header(f"MCA: **{col1}** + **{col2}**")
            coords, i1, i2 = compute_mca_and_segments(df, col1, col2)
            fig = px.scatter(
                coords,
                x='Dim1', y='Dim2',
                color='segment',
                title=f"MCA: {col1} + {col2}",
                labels={
                    'Dim1': f"Dim1 ({i1:.1f}% inertia)",
                    'Dim2': f"Dim2 ({i2:.1f}% inertia)",
                },
                render_mode='webgl'
            )
            fig.update_traces(marker={'size': 3, 'opacity': 0.3})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
