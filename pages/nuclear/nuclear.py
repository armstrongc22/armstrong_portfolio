# nuclear/nuclear.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NUCLEAR = DATA_DIR / "wn_all_countries_reactors.csv"

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Nuclear Energy Dashboard",
    layout="wide",
)

# Import scripts AFTER set_page_config
from scripts import growing, performance, pipeline, capacity


def create_nuclear_choropleth(df_reactors):
    """
    Create a simple choropleth map showing nuclear reactor counts by country.
    Args:
        df_reactors: DataFrame with 'Country' and reactor count columns
    Returns:
        plotly.graph_objects.Figure: Choropleth map
    """
    # Count reactors by country
    reactor_counts = df_reactors.groupby('Country').size().reset_index(name='Reactor_Count')

    # Country to ISO code mapping (add more as needed)
    country_iso_map = {
        'Argentina': 'ARG',
        'Armenia': 'ARM',
        'Belgium': 'BEL',
        'Brazil': 'BRA',
        'Bulgaria': 'BGR',
        'Canada': 'CAN',
        'China': 'CHN',
        'Czech Republic': 'CZE',
        'Finland': 'FIN',
        'France': 'FRA',
        'Germany': 'DEU',
        'Hungary': 'HUN',
        'India': 'IND',
        'Iran': 'IRN',
        'Japan': 'JPN',
        'Mexico': 'MEX',
        'Netherlands': 'NLD',
        'Pakistan': 'PAK',
        'Romania': 'ROU',
        'Russia': 'RUS',
        'Slovakia': 'SVK',
        'Slovenia': 'SVN',
        'South Africa': 'ZAF',
        'South Korea': 'KOR',
        'Spain': 'ESP',
        'Sweden': 'SWE',
        'Switzerland': 'CHE',
        'Ukraine': 'UKR',
        'United Kingdom': 'GBR',
        'United States': 'USA',
        'USA': 'USA'
    }

    # Add ISO codes
    reactor_counts['iso_alpha'] = reactor_counts['Country'].map(country_iso_map)

    # Remove countries without ISO codes
    reactor_counts = reactor_counts.dropna(subset=['iso_alpha'])

    # Create choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=reactor_counts['iso_alpha'],
        z=reactor_counts['Reactor_Count'],
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
            title="Nuclear Reactors",
            titlefont=dict(color='white', size=12),
            tickfont=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1,
            len=0.7,
            thickness=15
        ),
        hovertemplate='<b>%{text}</b><br>Reactors: %{z}<extra></extra>',
        text=reactor_counts['Country'],
        showscale=True
    ))

    # Update layout for black map background
    fig.update_layout(
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
        paper_bgcolor='white',  # Keep page background white
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )

    return fig


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Home", "Growth Atlas", "Supply-Chain Risk", "Performance Benchmark/Opportunity", "Deal Pipeline")
)

if page == "Home":
    st.title("🏠 Nuclear Energy Insights")
    st.markdown(
        """
        Welcome to the **Nuclear Energy Dashboard**.  
        Use the menu on the left to explore:
        1. **Growth Atlas** – Identify the fastest-growing nuclear markets.  
        2. **Supply-Chain Risk** – Uranium feed production vs reactor demand. 
        3. **Performance Benchmark/Opportunity** - Where increase capacity leads to largest service opportunities. 
        4. **Deal Pipeline** – Live reactor financings with sovereign & ECA overlays.  
        """
    )

    # Try to load reactor data and create choropleth map
    try:
        # You'll need to adjust this path to match your actual data file
        # This assumes you have a CSV file with reactor data including a 'Country' column
        df_reactors = pd.read_csv(NUCLEAR)  # Adjust path as needed

        st.subheader("Global Nuclear Reactor Distribution")
        fig = create_nuclear_choropleth(df_reactors)
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("Reactor data file not found. Please ensure 'data/reactors.csv' exists with a 'Country' column.")
        # Fallback to the original image
        st.image(
            "https://www.world-nuclear.org/getmedia/0a212cba-1a5f-4de6-9d7a-5efb9728f691/World-Map-of-Nuclear-Power-Reactors.png",
            caption="Global Nuclear Reactor Map", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading reactor data: {e}")
        # Fallback to the original image
        st.image(
            "https://www.world-nuclear.org/getmedia/0a212cba-1a5f-4de6-9d7a-5efb9728f691/World-Map-of-Nuclear-Power-Reactors.png",
            caption="Global Nuclear Reactor Map", use_column_width=True)

elif page == "Growth Atlas":
    growing.main()

elif page == "Supply-Chain Risk":
    capacity.main()

elif page == "Performance Benchmark/Opportunity":
    performance.main()

elif page == "Deal Pipeline":
    pipeline.main()