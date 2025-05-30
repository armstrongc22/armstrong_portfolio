# Show insights only if we have them

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prince
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ── Build correct folders relative to this script ───────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_csvs"
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

    # Load only first part if multiple files to reduce memory
    if len(parts) > 1:
        st.info(f"Loading sample data from {parts[0].name} for performance (found {len(parts)} files)")
        df = pd.read_csv(parts[0])
    else:
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    # Sample data if too large
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        st.info(f"Sampled 10,000 rows for performance")

    df['birthday'] = pd.to_datetime(df['birthday'], errors='coerce')
    df = df.dropna(subset=['birthday']).copy()
    df['age'] = ((pd.Timestamp.now() - df['birthday']).dt.days // 365).astype(int)
    # 5-year bins
    bins = list(range(10, 81, 5))
    labels = [f"{i}-{i + 4}" for i in range(10, 80, 5)]
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    # top-10 regions + Other
    top10 = df['region'].value_counts().nlargest(10).index
    df['region2'] = df['region'].where(df['region'].isin(top10), other='Other')
    return df


@st.cache_data
def compute_mca_and_segments(df: pd.DataFrame, col1: str, col2: str, n_clusters: int = 3):
    try:
        # Further sample if still too large
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df.copy()

        X = df_sample[[col1, col2]].dropna().astype(str)

        if len(X) < 100:
            st.warning(f"Not enough data for MCA analysis ({len(X)} rows)")
            return pd.DataFrame(), 0, 0

        mca = prince.MCA(n_components=2, random_state=42).fit(X)
        coords = mca.row_coordinates(X)
        coords.columns = ['Dim1', 'Dim2']
        coords.index = X.index

        # K-Means segments with fewer clusters
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(coords)
        coords['segment'] = km.labels_

        # inertia percentages
        eigs = mca.eigenvalues_
        total = eigs.sum()
        inertia1 = eigs[0] / total * 100
        inertia2 = eigs[1] / total * 100

        # Add back original data for segment analysis
        for col in [col1, col2]:
            coords[col] = df_sample.loc[coords.index, col].values

        return coords, inertia1, inertia2
    except Exception as e:
        st.error(f"MCA analysis failed: {str(e)}")
        return pd.DataFrame(), 0, 0


def create_professional_charts():
    """Create professional KPI visualizations"""
    # Color palette for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    return colors


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Euphoria CSV Dashboard (Desktop Only)")

    tabs = st.tabs(["Home", "KPIs", "Yearly Map", "Segments"])


    # Tab 2: Enhanced KPIs with Professional Charts
    with tabs[0]:
        st.markdown("""
        **Euphoria** is a massive fictional fantasy IP created as the backbone for a marketing project using synthetic data.
        The company produces physical copies of their manga, video games, and a variety of merchandise. The video games have created an online community of streamers and customers who religiously tune in. 
        This project analyzes the KPI's for the company, visualizes the countries that watch the Euphoria streams, and identifies customer segments ripe for optimization for the company's most expensive product.
        The data for all of this was produced synthetically in Python. A pipeline was created from the local server to Confluent Kafka, and then BigQuery where it was stored and structured into a relational database.
        That database was queried using SQL syntax and the results displayed using Dash and Streamlit applications. 
        """"")
    with tabs[1]:
        st.header("📈 Key Performance Indicators")
        colors = create_professional_charts()

        # load each
        watch = try_read(DATA_DIR / "watch.csv")
        streams = try_read(DATA_DIR / "streams.csv")
        purchase = try_read(DATA_DIR / "purchase_events.csv")
        partners = try_read(DATA_DIR / "partners.csv")
        games = try_read(DATA_DIR / "games.csv")
        merch = try_read(DATA_DIR / "merch.csv")

        if any(df.empty for df in (watch, streams, purchase, partners, games, merch)):
            st.warning("One or more source tables is empty — check Disk & Heads.")
        else:
            # Create 2x2 grid for main KPIs
            col1, col2 = st.columns(2)

            with col1:
                # Top 10 viewed countries - Horizontal Bar Chart
                st.subheader("🌍 Top Viewing Countries")
                tv = watch.groupby("country")["length"].sum().nlargest(10).reset_index()
                tv['length_hours'] = tv['length'] / 60  # Convert to hours

                fig_countries = px.bar(
                    tv,
                    x='length_hours',
                    y='country',
                    orientation='h',
                    title="Total Watch Hours by Country",
                    labels={'length_hours': 'Watch Hours', 'country': 'Country'},
                    color='length_hours',
                    color_continuous_scale='Blues'
                )
                fig_countries.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_countries, use_container_width=True)

            with col2:
                # Top 8 purchased products - Pie Chart
                st.subheader("🛒 Product Purchase Distribution")
                tp = (purchase.groupby("product_name")
                      .size()
                      .nlargest(8)
                      .reset_index(name="count"))

                fig_products = px.pie(
                    tp,
                    values='count',
                    names='product_name',
                    title="Top 8 Products by Purchase Volume"
                )
                fig_products.update_traces(textposition='inside', textinfo='percent+label')
                fig_products.update_layout(height=400)
                st.plotly_chart(fig_products, use_container_width=True)

            # Streamer Performance - Enhanced Bar Chart
            st.subheader("🎮 Top Streamer Performance")
            sp = streams.merge(partners, on="partner_id", how="left")
            sp["score"] = (sp.viewers_total / sp.length.replace(0, 1)) * sp.comments_total
            top_str = (sp.groupby("screen_name")["score"]
                       .sum()
                       .nlargest(10)
                       .reset_index())
            top_str["score"] = top_str["score"].round(2)

            fig_streamers = px.bar(
                top_str,
                x='screen_name',
                y='score',
                title="Streamer Performance Score (Engagement × Views/Time)",
                labels={'score': 'Performance Score', 'screen_name': 'Streamer'},
                color='score',
                color_continuous_scale='Viridis'
            )
            fig_streamers.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_streamers, use_container_width=True)

            # Gaming Analytics - Side by side comparison
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("🏆 Best-Selling Games")
                bg = (purchase[purchase.category == "game"]
                      .groupby("product_name")
                      .size()
                      .nlargest(5)  # Show top 5 instead of 2
                      .reset_index(name="sales"))

                fig_sales = px.bar(
                    bg,
                    x='product_name',
                    y='sales',
                    title="Game Sales Volume",
                    color='sales',
                    color_continuous_scale='Reds'
                )
                fig_sales.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_sales, use_container_width=True)

            with col4:
                st.subheader("📺 Most-Streamed Games")
                ms = (streams.merge(games, on="game_id", how="left")
                      .groupby("title").size()
                      .nlargest(5)  # Show top 5 instead of 2
                      .reset_index(name="streams"))

                fig_streams = px.bar(
                    ms,
                    x='title',
                    y='streams',
                    title="Game Streaming Volume",
                    color='streams',
                    color_continuous_scale='Greens'
                )
                fig_streams.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_streams, use_container_width=True)

            # Summary metrics
            st.markdown("---")
            st.subheader("📊 Summary Metrics")

            total_watch_hours = watch['length'].sum() / 60
            total_purchases = len(purchase)
            avg_stream_viewers = streams['viewers_total'].mean()
            unique_countries = watch['country'].nunique()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Total Watch Hours", f"{total_watch_hours:,.0f}")
            with metric_col2:
                st.metric("Total Purchases", f"{total_purchases:,}")
            with metric_col3:
                st.metric("Avg Stream Viewers", f"{avg_stream_viewers:,.0f}")
            with metric_col4:
                st.metric("Countries Reached", f"{unique_countries}")

    # Tab 3: Yearly Map
    with tabs[2]:
        st.header("🗺️ Yearly Watch Choropleth")
        watch = try_read(DATA_DIR / "watch.csv")
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
                                    hover_data=["watch_hours", "pct_rank"],
                                    range_color=(0, 1))
                fig.update_layout(title=f"Yearly Watch Rank: {year}")
                st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Enhanced Trophy Segments with Insights
    with tabs[3]:
        st.title("🏆 Euphoria Trophy Purchasers: MCA + Segments")
        df = load_trophy_customers()
        st.write(f"Loaded **{len(df):,}** trophy customers")
        st.markdown("---")

        # Simplified pairs - do one at a time to reduce memory
        pairs = [
            ("age_bin", "gender"),
        ]

        # Add option to run more analyses
        if st.checkbox("Run additional MCA analyses (may be slower)", value=False):
            pairs.extend([
                ("age_bin", "region"),
                ("gender", "region"),
            ])

        all_insights = []

        for col1, col2 in pairs:
            st.header(f"MCA: **{col1}** + **{col2}**")

            with st.spinner(f"Computing MCA for {col1} + {col2}..."):
                coords, i1, i2 = compute_mca_and_segments(df, col1, col2)

            if coords.empty:
                st.warning("Skipping this analysis due to insufficient data")
                continue

            my_palette = ["#93329E", "#F4D03F", "#EA7600"]

            # Simplified scatter plot
            fig = px.scatter(
                coords.sample(n=min(2000, len(coords)), random_state=42),  # Sample points for display
                x='Dim1', y='Dim2',
                color='segment',
                title=f"MCA: {col1} + {col2}",
                labels={
                    'Dim1': f"Dim1 ({i1:.1f}% inertia)",
                    'Dim2': f"Dim2 ({i2:.1f}% inertia)",
                },
                color_discrete_sequence=my_palette,
                template="plotly_white",  # Lighter template for performance
                width=800, height=500
            )

            fig.update_traces(
                marker=dict(size=3, opacity=0.7)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Generate insights for this MCA (simplified)
            if not coords.empty:
                try:
                    segment_profiles = coords.groupby('segment')[[col1, col2]].agg(
                        lambda s: s.value_counts().index[0] if len(s.value_counts()) > 0 else 'Unknown'
                    )
                    segment_sizes = coords['segment'].value_counts().sort_index()

                    insight_data = {
                        'analysis': f"{col1} + {col2}",
                        'profiles': segment_profiles,
                        'sizes': segment_sizes,
                        'total_variance_explained': i1 + i2
                    }
                    all_insights.append(insight_data)
                except Exception as e:
                    st.warning(f"Could not generate insights for {col1} + {col2}: {str(e)}")

            st.markdown("---")

        # Comprehensive Insights Section
        st.header("🎯 Strategic Insights & Targeting Recommendations")

        st.subheader("Customer Segment Profiles")

        for i, insight in enumerate(all_insights):
            with st.expander(f"📊 Analysis: {insight['analysis']}", expanded=True):
                st.write(f"**Variance Explained:** {insight['total_variance_explained']:.1f}%")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.write("**Segment Characteristics:**")
                    for seg_id, row in insight['profiles'].iterrows():
                        size = insight['sizes'][seg_id]
                        pct = (size / insight['sizes'].sum()) * 100
                        st.write(f"**Segment {seg_id}** ({size:,} customers, {pct:.1f}%)")
                        for col_name, dominant_value in row.items():
                            st.write(f"  • {col_name}: {dominant_value}")

                with col_b:
                    # Create segment size chart
                    fig_segments = px.pie(
                        values=insight['sizes'].values,
                        names=[f"Segment {i}" for i in insight['sizes'].index],
                        title=f"Segment Distribution: {insight['analysis']}"
                    )
                    fig_segments.update_layout(height=300)
                    st.plotly_chart(fig_segments, use_container_width=True)

        # Strategic Recommendations
        st.subheader("🚀 Marketing & Targeting Strategy")

        # Check if we have insights before proceeding
        if all_insights:
            # Analyze the most informative segmentation
            best_analysis = max(all_insights, key=lambda x: x['total_variance_explained'])

            st.success(f"**Primary Segmentation Recommendation: {best_analysis['analysis']}**")
            st.write(
                f"This analysis explains {best_analysis['total_variance_explained']:.1f}% of customer variance and provides the clearest segmentation.")

            st.write("**Recommended Targeting Priorities:**")

            # Sort segments by size for targeting recommendations
            sorted_segments = best_analysis['sizes'].sort_values(ascending=False)

            for rank, (seg_id, size) in enumerate(sorted_segments.items(), 1):
                profile = best_analysis['profiles'].loc[seg_id]
                pct = (size / sorted_segments.sum()) * 100

                if rank == 1:
                    priority = "🥇 **HIGH PRIORITY**"
                    recommendation = "Primary target - largest segment with clear characteristics"
                elif rank == 2:
                    priority = "🥈 **MEDIUM PRIORITY**"
                    recommendation = "Secondary target - significant size, distinct profile"
                else:
                    priority = "🥉 **LOW PRIORITY**"
                    recommendation = "Niche segment - specialized targeting approach"

                st.markdown(f"""
                **{priority} - Segment {seg_id}**
                - **Size:** {size:,} customers ({pct:.1f}% of trophy buyers)
                - **Profile:** {', '.join([f"{k}: {v}" for k, v in profile.items()])}
                - **Strategy:** {recommendation}
                """)

            # Additional strategic insights
            st.subheader("💡 Key Business Insights")

            total_customers = len(df)

            # Age insights
            age_dist = df['age_bin'].value_counts()
            dominant_age = age_dist.index[0]

            # Gender insights
            gender_dist = df['gender'].value_counts()
            dominant_gender = gender_dist.index[0]
            gender_pct = (gender_dist.iloc[0] / len(df)) * 100

            # Region insights
            region_dist = df['region'].value_counts()
            top_region = region_dist.index[0]
            region_pct = (region_dist.iloc[0] / len(df)) * 100

            insights_text = f"""
            **Customer Base Overview:**
            - **Dominant Age Group:** {dominant_age} years ({(age_dist.iloc[0] / total_customers) * 100:.1f}% of customers)
            - **Gender Distribution:** {dominant_gender} represents {gender_pct:.1f}% of trophy buyers
            - **Geographic Concentration:** {top_region} accounts for {region_pct:.1f}% of purchases

            **Strategic Recommendations:**
            1. **Focus marketing campaigns** on the {dominant_age} age group and {dominant_gender} demographic
            2. **Expand regional presence** beyond {top_region} to capture underserved markets  
            3. **Develop targeted content** for each identified customer segment
            4. **Optimize trophy product offerings** based on segment preferences
            5. **Consider cross-segment promotions** to increase customer lifetime value
            """

            st.markdown(insights_text)
        else:
            st.warning("No insights available. Please check if the trophy customer data is loading correctly.")


if __name__ == "__main__":
    main()