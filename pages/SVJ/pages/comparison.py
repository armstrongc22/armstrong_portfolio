import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import os

# File path configuration - all CSV files are in the same directory as the script
BASE_PATH = Path(__file__).resolve().parent


def debug_file_loading():
    """Debug function to check what files exist"""
    st.write("### Debug Info: Files in directory")
    files = list(BASE_PATH.glob("*.csv"))
    st.write(f"Base path: {BASE_PATH}")
    st.write(f"CSV files found: {[f.name for f in files]}")
    return files


def clean_sheet(path: Path) -> pd.DataFrame:
    """
    Read and clean CSV files with proper data type handling.
    """
    try:
        st.write(f"Attempting to load: {path}")

        # Check if file exists
        if not path.exists():
            st.error(f"File does not exist: {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)
        st.write(f"Original shape: {df.shape}")
        st.write(f"Original columns: {list(df.columns)}")

        # Show first few rows to debug
        st.write("First 3 rows:")
        st.dataframe(df.head(3))

        # Handle different CSV structures
        if len(df.columns) > 7 and 'Player' not in df.columns:
            st.write("Handling misaligned headers...")
            # Handle misaligned headers - assume 2nd column is player name, 4th is GP
            df['Player'] = df.iloc[:, 1].astype(str)
            if len(df.columns) > 3:
                df['GP'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
            # Drop original misaligned columns
            df = df.drop(columns=df.columns[0:7])

        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)

        # Remove empty or duplicate columns
        df = df.loc[:, df.columns != ""]
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure Player column exists
        if 'Player' not in df.columns and len(df.columns) > 0:
            st.write(f"Renaming first column '{df.columns[0]}' to 'Player'")
            df = df.rename(columns={df.columns[0]: 'Player'})

        # Clean player names
        if 'Player' in df.columns:
            st.write("Before cleaning players:")
            st.write(f"Unique players (first 10): {df['Player'].unique()[:10]}")

            df['Player'] = df['Player'].astype(str).str.strip()
            # Remove rows where Player is NaN, empty, or just numbers
            df = df[df['Player'].notna()]
            df = df[df['Player'] != '']
            df = df[~df['Player'].str.match(r'^\d+$', na=False)]

            st.write("After cleaning players:")
            st.write(f"Shape: {df.shape}")

            # Check for our target players with fuzzy matching
            target_players = ["Jalen Green", "Alperen Sengun"]
            for target in target_players:
                matches = df[df['Player'].str.contains(target.split()[0], case=False, na=False)]
                st.write(f"Players containing '{target.split()[0]}': {matches['Player'].tolist()}")

        # Convert numeric columns properly
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col != 'Player':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle GP column specifically
        if 'GP' in df.columns:
            df['GP'] = pd.to_numeric(df['GP'], errors='coerce')
            st.write(f"GP column stats: min={df['GP'].min()}, max={df['GP'].max()}, null_count={df['GP'].isna().sum()}")
            before_gp_filter = len(df)
            df = df[df['GP'].notna()]  # Remove rows with invalid GP
            st.write(f"Removed {before_gp_filter - len(df)} rows with invalid GP")

        st.write(f"Final shape: {df.shape}")
        st.write(f"Final columns: {list(df.columns)}")

        return df

    except Exception as e:
        st.error(f"Error reading {path}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()


# Data Loading Functions - Updated with debug info
@st.cache_data
def load_guard_basic():
    return clean_sheet(BASE_PATH / "guard_basic.csv")


@st.cache_data
def load_guard_advanced():
    return clean_sheet(BASE_PATH / "guard_advanced.csv")


@st.cache_data
def load_isolation():
    return clean_sheet(BASE_PATH / "isolation.csv")


@st.cache_data
def load_pnr_handler():
    return clean_sheet(BASE_PATH / "pnr_handler.csv")


@st.cache_data
def load_centers_basics():
    return clean_sheet(BASE_PATH / "centers_basics.csv")


@st.cache_data
def load_centers_advanced():
    return clean_sheet(BASE_PATH / "centers_advanced.csv")


@st.cache_data
def load_post_ups():
    return clean_sheet(BASE_PATH / "post_ups.csv")


@st.cache_data
def load_pnr_big():
    return clean_sheet(BASE_PATH / "pnr_big.csv")


# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Player Segmentation & Ranking Explorer - DEBUG MODE")

    # Add debug section
    if st.checkbox("Show Debug Info"):
        debug_file_loading()
        st.write("---")

    # Player selection
    player = st.selectbox("Choose player", ["Jalen Green", "Alperen Sengun"])

    # Segment-to-function mapping
    func_map = {
        "Guard Basic": load_guard_basic,
        "Guard Advanced": load_guard_advanced,
        "Isolation": load_isolation,
        "Pick & Roll Handler": load_pnr_handler,
        "Center Basic": load_centers_basics,
        "Center Advanced": load_centers_advanced,
        "Post Ups": load_post_ups,
        "Pick & Roll Big": load_pnr_big
    }

    # Available segments per player
    segs = {
        "Jalen Green": ["Guard Basic", "Guard Advanced", "Isolation", "Pick & Roll Handler"],
        "Alperen Sengun": ["Center Basic", "Center Advanced", "Post Ups", "Isolation", "Pick & Roll Big"]
    }

    # Segment selection
    choices = segs[player]
    segment = st.selectbox("Choose segment", choices)

    # Load the chosen dataset
    with st.spinner(f"Loading {segment} data..."):
        df = func_map[segment]()

    if df.empty:
        st.error(f"Could not load data for {segment}. Please check if the file exists.")
        return

    # Display basic info about the dataset
    st.write(f"Dataset shape: {df.shape}")
    if 'Player' in df.columns:
        st.write(f"Number of players: {len(df)}")

    # Show all players that might match our target
    if 'Player' in df.columns:
        st.write("### All players in dataset (first 20):")
        st.write(df['Player'].head(20).tolist())

        # Try fuzzy matching
        target_name = player.split()[0]  # First name
        possible_matches = df[df['Player'].str.contains(target_name, case=False, na=False)]
        if not possible_matches.empty:
            st.write(f"### Possible matches for '{player}':")
            st.write(possible_matches['Player'].tolist())

    # Filter: drop GP<50 except chosen player (if GP column exists)
    if 'GP' in df.columns:
        original_size = len(df)
        df = df[(df['GP'] >= 50) | (df['Player'] == player)]
        filtered_size = len(df)
        if original_size != filtered_size:
            st.info(f"Filtered to {filtered_size} players (GP ≥ 50 or selected player)")

    # Check if our chosen player is in the dataset
    if player not in df['Player'].values:
        st.warning(f"{player} not found in this dataset")

        # Try alternative matching
        if 'Player' in df.columns:
            # Try first name only
            first_name = player.split()[0]
            matches = df[df['Player'].str.contains(first_name, case=False, na=False)]
            if not matches.empty:
                st.write(f"Found players with first name '{first_name}':")
                for idx, row in matches.iterrows():
                    st.write(f"- {row['Player']}")

                # Use the first match
                actual_name = matches.iloc[0]['Player']
                st.info(f"Using '{actual_name}' instead of '{player}'")
                player = actual_name
            else:
                return

    # Choose statistic (exclude meta columns)
    exclude = {'Player', 'GP', 'Min', 'W', 'L', 'Age', 'Team'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = [c for c in numeric_cols if c not in exclude]

    if not stats:
        st.error("No numeric statistics found in this dataset")
        st.write("Available columns:", df.columns.tolist())
        return

    stat = st.selectbox("Choose statistic to visualize", stats)

    # Display some info about the selected statistic
    if stat in df.columns:
        player_row = df[df['Player'] == player]
        if len(player_row) > 0:
            player_value = player_row[stat].iloc[0]
            rank = (df[stat] > player_value).sum() + 1
            total = len(df)
            st.write(f"{player}'s {stat}: **{player_value:.3f}** (Rank: {rank}/{total})")

    # Create the scatterplot
    try:
        chart = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X(f"{stat}:Q", title=stat),
                y=alt.Y("Player:N",
                        sort=alt.EncodingSortField(stat, order="descending"),
                        axis=alt.Axis(labelLimit=200)),
                color=alt.condition(
                    alt.datum.Player == player,
                    alt.value("red"),
                    alt.value("steelblue")
                ),
                size=alt.condition(
                    alt.datum.Player == player,
                    alt.value(300),
                    alt.value(60)
                ),
                tooltip=["Player:N", f"{stat}:Q"]
            )
            .properties(height=min(600, len(df) * 20 + 100), width=800)
            .resolve_scale(y='independent')
        )

        st.altair_chart(chart, use_container_width=True)

        # Show summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean", f"{df[stat].mean():.3f}")
        with col2:
            st.metric("Median", f"{df[stat].median():.3f}")
        with col3:
            st.metric("Std Dev", f"{df[stat].std():.3f}")

        # Show top and bottom 5 players
        st.subheader(f"Rankings by {stat}")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top 5:**")
            top_5 = df.nlargest(5, stat)[['Player', stat]]
            st.dataframe(top_5, hide_index=True)

        with col2:
            st.write("**Bottom 5:**")
            bottom_5 = df.nsmallest(5, stat)[['Player', stat]]
            st.dataframe(bottom_5, hide_index=True)

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.write("Data preview:")
        st.dataframe(df.head())


if __name__ == "__main__":
    main()