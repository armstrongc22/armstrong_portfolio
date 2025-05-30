import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Sample data based on what I can see from your screenshot
reactor_data = {
    'Country': ['Argentina', 'Armenia', 'Belgium', 'Brazil', 'Bulgaria', 'Canada', 'China', 'Czech Republic', 'Finland',
                'France', 'Germany', 'Hungary', 'India', 'Iran', 'Japan', 'Mexico', 'Netherlands', 'Pakistan',
                'Romania', 'Russia', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sweden',
                'Switzerland', 'Ukraine', 'United Kingdom', 'United States'],
    'Reactor_Count': [3, 1, 7, 2, 2, 19, 55, 6, 4, 56, 3, 4, 23, 1, 33, 2, 1, 6, 2, 38, 4, 1, 2, 25, 7, 6, 4, 15, 9,
                      93],
    'iso_alpha': ['ARG', 'ARM', 'BEL', 'BRA', 'BGR', 'CAN', 'CHN', 'CZE', 'FIN', 'FRA', 'DEU', 'HUN', 'IND', 'IRN',
                  'JPN', 'MEX', 'NLD', 'PAK', 'ROU', 'RUS', 'SVK', 'SVN', 'ZAF', 'KOR', 'ESP', 'SWE', 'CHE', 'UKR',
                  'GBR', 'USA']
}


def create_nuclear_choropleth():
    """Create a choropleth map showing nuclear reactor counts by country"""

    # Create DataFrame
    df = pd.DataFrame(reactor_data)

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['iso_alpha'],
        z=df['Reactor_Count'],
        locationmode='ISO-3',
        colorscale=[
            [0, '#FFFFFF'],  # White for 0 reactors
            [0.1, '#E6FFE6'],  # Very light green
            [0.3, '#B3FFB3'],  # Light green
            [0.5, '#66FF66'],  # Medium green
            [0.7, '#33FF33'],  # Bright green
            [1.0, '#00FF00']  # Neon green for highest counts
        ],
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar=dict(
            title="Number of Nuclear Reactors",
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white', size=12),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            x=1.02
        ),
        hovertemplate='<b>%{text}</b><br>Nuclear Reactors: %{z}<extra></extra>',
        text=df['Country'],
        showscale=True
    ))

    # Update layout for black background theme
    fig.update_layout(
        title=dict(
            text='Global Nuclear Reactor Distribution',
            font=dict(color='white', size=20),
            x=0.5
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='gray',
            projection_type='natural earth',
            bgcolor='black',
            landcolor='#1a1a1a',
            oceancolor='black',
            showlakes=True,
            lakecolor='black',
            showrivers=False
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        width=1200,
        height=700
    )

    return fig


def create_reactor_stats_table(df):
    """Create a summary statistics table"""

    # Top 10 countries by reactor count
    top_countries = df.nlargest(10, 'Reactor_Count')

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Country', 'Number of Reactors'],
            fill_color='#00FF00',
            font=dict(color='black', size=14),
            align='center'
        ),
        cells=dict(
            values=[
                list(range(1, len(top_countries) + 1)),
                top_countries['Country'].tolist(),
                top_countries['Reactor_Count'].tolist()
            ],
            fill_color='rgba(0, 255, 0, 0.1)',
            font=dict(color='white', size=12),
            align='center'
        )
    )])

    fig.update_layout(
        title=dict(
            text='Top 10 Countries by Nuclear Reactor Count',
            font=dict(color='white', size=16),
            x=0.5
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=400
    )

    return fig


def main():
    # Set page config with white background
    st.set_page_config(
        page_title="Nuclear Reactor Global Distribution",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS to ensure white background and proper styling
    st.markdown("""
    <style>
    .main {
        background-color: white;
    }
    .stApp {
        background-color: white;
    }
    .block-container {
        background-color: white;
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: black !important;
    }
    .stMarkdown {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("⚛️ Global Nuclear Reactor Distribution")
    st.markdown("**Interactive choropleth map showing the distribution of nuclear reactors worldwide**")

    # Create DataFrame
    df = pd.DataFrame(reactor_data)

    # Sidebar with statistics
    st.sidebar.header("📊 Global Statistics")
    st.sidebar.metric("Total Countries with Nuclear Power", len(df))
    st.sidebar.metric("Total Nuclear Reactors", df['Reactor_Count'].sum())
    st.sidebar.metric("Average Reactors per Country", f"{df['Reactor_Count'].mean():.1f}")
    st.sidebar.metric("Median Reactors per Country", df['Reactor_Count'].median())

    # Top country
    top_country = df.loc[df['Reactor_Count'].idxmax()]
    st.sidebar.metric("Country with Most Reactors", f"{top_country['Country']} ({top_country['Reactor_Count']})")

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Create and display the choropleth map
        fig_map = create_nuclear_choropleth()
        st.plotly_chart(fig_map, use_container_width=True)

        # Display the data table
        st.subheader("📋 Reactor Data by Country")
        st.dataframe(
            df.sort_values('Reactor_Count', ascending=False),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        # Key insights
        st.subheader("🔑 Key Insights")
        st.markdown(f"""
        - **{df['Reactor_Count'].sum()}** total nuclear reactors worldwide
        - **{len(df)}** countries operate nuclear power plants
        - The **{top_country['Country']}** leads with **{top_country['Reactor_Count']}** reactors
        - Countries with 20+ reactors: **{len(df[df['Reactor_Count'] >= 20])}**
        - Average reactors per country: **{df['Reactor_Count'].mean():.1f}**
        """)

        # Distribution chart
        st.subheader("📈 Distribution Analysis")

        # Histogram of reactor counts
        fig_hist = px.histogram(
            df,
            x='Reactor_Count',
            nbins=10,
            title='Distribution of Reactor Counts',
            color_discrete_sequence=['#00FF00']
        )
        fig_hist.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Bottom section with top countries table
    st.subheader("🏆 Top Countries by Nuclear Reactor Count")
    fig_table = create_reactor_stats_table(df)
    st.plotly_chart(fig_table, use_container_width=True)

    # Additional information
    st.info("""
    **About this visualization:**
    This choropleth map displays the global distribution of nuclear reactors using a neon green to white color scale. 
    Countries with more reactors appear in brighter green, while countries with fewer reactors appear in lighter shades.
    The data includes operational nuclear power reactors as of 2024.
    """)


if __name__ == "__main__":
    main()