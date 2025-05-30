import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path
# Define the path to your CSV file
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NUCLEAR = DATA_DIR / "wn_all_countries_reactors.csv"


# Country coordinates for mapping (you can expand this based on your data)
COUNTRY_COORDS = {
    'United States': {'lat': 39.8283, 'lon': -98.5795},
    'France': {'lat': 46.2276, 'lon': 2.2137},
    'China': {'lat': 35.8617, 'lon': 104.1954},
    'Japan': {'lat': 36.2048, 'lon': 138.2529},
    'Russia': {'lat': 61.5240, 'lon': 105.3188},
    'South Korea': {'lat': 35.9078, 'lon': 127.7669},
    'Ukraine': {'lat': 48.3794, 'lon': 31.1656},
    'Canada': {'lat': 56.1304, 'lon': -106.3468},
    'United Kingdom': {'lat': 55.3781, 'lon': -3.4360},
    'Spain': {'lat': 40.4637, 'lon': -3.7492},
    'Sweden': {'lat': 60.1282, 'lon': 18.6435},
    'India': {'lat': 20.5937, 'lon': 78.9629},
    'Belgium': {'lat': 50.5039, 'lon': 4.4699},
    'Czech Republic': {'lat': 49.8175, 'lon': 15.4730},
    'Finland': {'lat': 61.9241, 'lon': 25.7482},
    'Switzerland': {'lat': 46.8182, 'lon': 8.2275},
    'Bulgaria': {'lat': 42.7339, 'lon': 25.4858},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253},
    'Argentina': {'lat': -38.4161, 'lon': -63.6167},
    'South Africa': {'lat': -30.5595, 'lon': 22.9375},
    'Hungary': {'lat': 47.1625, 'lon': 19.5033},
    'Slovenia': {'lat': 46.1512, 'lon': 14.9955},
    'Netherlands': {'lat': 52.1326, 'lon': 5.2913},
    'Romania': {'lat': 45.9432, 'lon': 24.9668},
    'Slovakia': {'lat': 48.6690, 'lon': 19.6990},
    'Armenia': {'lat': 40.0691, 'lon': 45.0382},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528},
    'Pakistan': {'lat': 30.3753, 'lon': 69.3451},
    'Iran': {'lat': 32.4279, 'lon': 53.6880},
    'UAE': {'lat': 23.4241, 'lon': 53.8478},
    'Taiwan': {'lat': 23.6978, 'lon': 120.9605},
    'Belarus': {'lat': 53.7098, 'lon': 27.9534},
    'Turkey': {'lat': 38.9637, 'lon': 35.2433}
}


def process_reactor_data(df):
    """Process the reactor DataFrame to get country-level statistics"""
    # Clean and process the data
    df = df.copy()
    df['Capacity (MWe)'] = pd.to_numeric(df['Capacity (MWe)'], errors='coerce')
    df['Net Capacity (MWe)'] = pd.to_numeric(df['Net Capacity (MWe)'], errors='coerce')

    # Clean the Load Factor column - convert to numeric
    df['Load Factor (2023) (%)'] = pd.to_numeric(df['Load Factor (2023) (%)'], errors='coerce')

    # Clean the Electricity Generated column - convert to numeric
    df['Electricity Generated (2023) (GWh)'] = pd.to_numeric(df['Electricity Generated (2023) (GWh)'], errors='coerce')

    # Use Net Capacity if available, otherwise use Capacity
    df['Total_Capacity'] = df['Net Capacity (MWe)'].fillna(df['Capacity (MWe)'])

    # Group by country with proper error handling
    try:
        country_stats = df.groupby('Country').agg({
            'Total_Capacity': ['sum', 'count'],
            'Load Factor (2023) (%)': 'mean',
            'Electricity Generated (2023) (GWh)': 'sum'
        }).round(2)
    except Exception as e:
        # If there's still an error, create a simpler aggregation
        country_stats = df.groupby('Country').agg({
            'Total_Capacity': ['sum', 'count']
        }).round(2)
        # Add empty columns for the problematic ones
        country_stats[('Load Factor (2023) (%)', 'mean')] = 0
        country_stats[('Electricity Generated (2023) (GWh)', 'sum')] = 0

    # Flatten column names
    country_stats.columns = ['Total_Capacity_MW', 'Reactor_Count', 'Avg_Load_Factor', 'Total_Generation_GWh']
    country_stats = country_stats.reset_index()

    # Add coordinates
    country_stats['lat'] = country_stats['Country'].map(lambda x: COUNTRY_COORDS.get(x, {}).get('lat'))
    country_stats['lon'] = country_stats['Country'].map(lambda x: COUNTRY_COORDS.get(x, {}).get('lon'))

    # Remove countries without coordinates
    country_stats = country_stats.dropna(subset=['lat', 'lon'])

    return country_stats


def create_nuclear_heatmap(df):
    """Create the main nuclear reactor heat map"""

    # Process data
    country_data = process_reactor_data(df)

    # Create the heat map using plotly
    fig = go.Figure()

    # Add the world map base
    fig.add_trace(go.Scattergeo(
        lon=country_data['lon'],
        lat=country_data['lat'],
        text=country_data.apply(lambda row:
                                f"<b>{row['Country']}</b><br>" +
                                f"Reactors: {int(row['Reactor_Count'])}<br>" +
                                f"Total Capacity: {row['Total_Capacity_MW']:,.0f} MW<br>" +
                                f"Avg Load Factor: {row['Avg_Load_Factor']:.1f}%<br>" +
                                f"Generation: {row['Total_Generation_GWh']:,.0f} GWh", axis=1),
        mode='markers',
        marker=dict(
            size=np.sqrt(country_data['Total_Capacity_MW']) / 30,  # Scale marker size
            color=country_data['Total_Capacity_MW'],
            colorscale=[
                [0, '#000000'],  # Black
                [0.3, '#003300'],  # Dark green
                [0.5, '#00ff41'],  # Neon green
                [0.8, '#66ff66'],  # Light green
                [1.0, '#ffffff']  # White
            ],
            colorbar=dict(
                title="<b>Capacity (MW)</b>",
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='#00ff41',
                borderwidth=1
            ),
            line=dict(color='#00ff41', width=2),
            sizemode='diameter',
            sizemin=8,
            sizemax=50,
            opacity=0.9
        ),
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))

    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text="<b>GLOBAL NUCLEAR REACTOR HEAT MAP</b>",
            font=dict(size=24, color='white'),
            x=0.5,
            y=0.95
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(20, 20, 20)',
            showocean=True,
            oceancolor='rgb(0, 0, 0)',
            showlakes=True,
            lakecolor='rgb(0, 0, 0)',
            showrivers=True,
            rivercolor='rgb(0, 0, 0)',
            coastlinecolor='rgb(0, 255, 65)',
            coastlinewidth=1,
            countrycolor='rgb(0, 100, 0)',
            countrywidth=0.5,
            showframe=False,
            showcoastlines=True,
            bgcolor='black'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def create_capacity_distribution_chart(df):
    """Create a capacity distribution chart"""
    country_data = process_reactor_data(df)

    # Sort by capacity for better visualization
    country_data = country_data.sort_values('Total_Capacity_MW', ascending=True).tail(15)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=country_data['Country'],
        x=country_data['Total_Capacity_MW'],
        orientation='h',
        marker=dict(
            color=country_data['Total_Capacity_MW'],
            colorscale=[
                [0, '#000000'],
                [0.3, '#003300'],
                [0.5, '#00ff41'],
                [0.8, '#66ff66'],
                [1.0, '#ffffff']
            ],
            line=dict(color='#00ff41', width=1)
        ),
        text=country_data['Total_Capacity_MW'].apply(lambda x: f'{x:,.0f} MW'),
        textposition='outside',
        textfont=dict(color='white')
    ))

    fig.update_layout(
        title=dict(
            text="<b>Top 15 Countries by Nuclear Capacity</b>",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title='Total Capacity (MW)',
            color='white',
            gridcolor='rgba(0, 255, 65, 0.2)'
        ),
        yaxis=dict(
            color='white'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=500,
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def create_reactor_timeline(df):
    """Create a timeline of reactor construction"""
    df_clean = df.copy()
    df_clean['Construction Start'] = pd.to_datetime(df_clean['Construction Start'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Construction Start'])
    df_clean['Year'] = df_clean['Construction Start'].dt.year

    # Count reactors by year
    timeline_data = df_clean.groupby('Year').size().reset_index(name='Reactors_Started')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeline_data['Year'],
        y=timeline_data['Reactors_Started'],
        mode='lines+markers',
        line=dict(color='#00ff41', width=3),
        marker=dict(
            color='#00ff41',
            size=8,
            line=dict(color='white', width=1)
        ),
        fill='tonexty',
        fillcolor='rgba(0, 255, 65, 0.1)'
    ))

    fig.update_layout(
        title=dict(
            text="<b>Nuclear Reactor Construction Timeline</b>",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title='Year',
            color='white',
            gridcolor='rgba(0, 255, 65, 0.2)'
        ),
        yaxis=dict(
            title='Reactors Started',
            color='white',
            gridcolor='rgba(0, 255, 65, 0.2)'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=400
    )

    return fig


def main():
    """Main function to create the nuclear heatmap dashboard"""

    st.set_page_config(
        page_title="Nuclear Energy Atlas",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .main-header {
        text-align: center;
        color: #00ff41;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px #00ff41;
    }
    .metric-container {
        background: linear-gradient(135deg, rgba(0,255,65,0.1) 0%, rgba(0,0,0,0.8) 100%);
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="main-header">🌍 GLOBAL NUCLEAR ATLAS</h1>', unsafe_allow_html=True)

    # Load your actual CSV data
    try:
        df = pd.read_csv(NUCLEAR)

        # Debug: Display column names and first few rows
        st.write("Debug: CSV Columns found:")
        st.write(df.columns.tolist())
        st.write("Debug: First 5 rows:")
        st.write(df.head())
        st.write("Debug: Data types:")
        st.write(df.dtypes)

    except FileNotFoundError:
        st.error(f"Data file not found at: {NUCLEAR}")
        st.info("Please ensure the CSV file exists in the correct location.")
        return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Main heat map
    st.plotly_chart(create_nuclear_heatmap(df), use_container_width=True, theme=None)

    # Statistics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_reactors = len(df)
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ff41; text-align: center;">{total_reactors}</h3>
            <p style="color: white; text-align: center;">Total Reactors</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_capacity = pd.to_numeric(df['Capacity (MWe)'], errors='coerce').sum() / 1000
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ff41; text-align: center;">{total_capacity:.1f} GW</h3>
            <p style="color: white; text-align: center;">Total Capacity</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        countries = df['Country'].nunique()
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ff41; text-align: center;">{countries}</h3>
            <p style="color: white; text-align: center;">Countries</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_load_factor = pd.to_numeric(df['Load Factor (2023) (%)'], errors='coerce').mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ff41; text-align: center;">{avg_load_factor:.1f}%</h3>
            <p style="color: white; text-align: center;">Avg Load Factor</p>
        </div>
        """, unsafe_allow_html=True)

    # Additional charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_capacity_distribution_chart(df), use_container_width=True, theme=None)

    with col2:
        st.plotly_chart(create_reactor_timeline(df), use_container_width=True, theme=None)


if __name__ == "__main__":
    main()