import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import os

# File path configuration - all CSV files are in the same directory as the script
BASE_PATH = Path(__file__).resolve().parent


def get_available_files():
    """Get all CSV files and create a mapping"""
    files = list(BASE_PATH.glob("*.csv"))
    file_mapping = {}

    for file in files:
        name = file.stem.lower()  # Get filename without extension, lowercase
        file_mapping[name] = file

    return file_mapping


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

        # Clean column names first
        df.columns = df.columns.str.strip()

        # Handle different possible column names for Player
        possible_player_cols = ['Player', 'PLAYER', 'player', 'Name', 'NAME', 'name']
        player_col = None

        for col in possible_player_cols:
            if col in df.columns:
                player_col = col
                break

        if player_col and player_col != 'Player':
            df = df.rename(columns={player_col: 'Player'})
            st.write(f"Renamed '{player_col}' to 'Player'")
        elif not player_col and len(df.columns) > 0:
            # Assume first column is player name
            df = df.rename(columns={df.columns[0]: 'Player'})
            st.write(f"Assumed first column '{df.columns[0]}' is Player")

        # Clean player names
        if 'Player' in df.columns:
            st.write("Before cleaning players:")
            st.write(f"Unique players (first 10): {df['Player'].unique()[:10]}")

            df['Player'] = df['Player'].astype(str).str.strip()
            # Remove rows where Player is NaN, empty, or just numbers
            df = df[df['Player'].notna()]
            df = df[df['Player'] != '']
            df = df[df['Player'] != 'nan']
            df = df[~df['Player'].str.match(r'^\d+\.?\d*$', na=False)]  # Remove pure numbers

            st.write("After cleaning players:")
            st.write(f"Shape: {df.shape}")
            st.write(f"Sample players: {df['Player'].head(10).tolist()}")

        # Handle GP column specifically
        possible_gp_cols = ['GP', 'G', 'Games', 'GAMES']
        gp_col = None

        for col in possible_gp_cols:
            if col in df.columns:
                gp_col = col
                break

        if gp_col and gp_col != 'GP':
            df = df.rename(columns={gp_col: 'GP'})
            st.write(f"Renamed '{gp_col}' to 'GP'")

        # Convert numeric columns properly
        for col in df.columns:
            if col != 'Player':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle GP column validation
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


def find_player_in_dataframe(df, target_player):
    """Find a player in the dataframe using fuzzy matching"""
    if 'Player' not in df.columns:
        return None, None

    # Exact match first
    exact_match = df[df['Player'] == target_player]
    if not exact_match.empty:
        return target_player, exact_match

    # Try case-insensitive match
    case_match = df[df['Player'].str.lower() == target_player.lower()]
    if not case_match.empty:
        return case_match.iloc[0]['Player'], case_match

    # Try partial matches
    target_parts = target_player.lower().split()
    for part in target_parts:
        if len(part) > 2:  # Only use meaningful parts
            partial_matches = df[df['Player'].str.lower().str.contains(part, na=False)]
            if not partial_matches.empty:
                return partial_matches.iloc[0]['Player'], partial_matches

    return None, None


# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Player Segmentation & Ranking Explorer - FIXED VERSION")

    # Get available files
    file_mapping = get_available_files()

    # Debug section
    if st.checkbox("Show Debug Info"):
        st.write("### Debug Info: Available CSV Files")
        st.write(f"Base path: {BASE_PATH}")
        st.write(f"Available files: {list(file_mapping.keys())}")
        st.write("---")

    # Player selection
    player = st.selectbox("Choose player", ["Jalen Green", "Alperen Sengun"])

    # Define segment mappings with flexible file name matching
    segment_file_mapping = {
        "Guard Basic": ["guard_basic", "guard_basics", "guards_basic"],
        "Guard Advanced": ["guard_advanced", "guards_advanced"],
        "Isolation": ["isolation", "iso"],
        "Pick & Roll Handler": ["pnr_handler", "pick_roll_handler", "pickroll_handler"],
        "Center Basic": ["centers_basics", "center_basic", "centers_basic"],
        "Center Advanced": ["centers_advanced", "center_advanced"],
        "Post Ups": ["post_ups", "postups", "post_up"],
        "Pick & Roll Big": ["pnr_big", "pick_roll_big", "pickroll_big"]
    }

    # Available segments per player
    segs = {
        "Jalen Green": ["Guard Basic", "Guard Advanced", "Isolation", "Pick & Roll Handler"],
        "Alperen Sengun": ["Center Basic", "Center Advanced", "Post Ups", "Isolation", "Pick & Roll Big"]
    }

    # Segment selection
    choices = segs[player]
    segment = st.selectbox("Choose segment", choices)

    # Find the correct file for the selected segment
    segment_file = None
    for possible_name in segment_file_mapping[segment]:
        if possible_name in file_mapping:
            segment_file = file_mapping[possible_name]
            break

    if not segment_file:
        st.error(f"Could not find file for {segment}. Available files: {list(file_mapping.keys())}")
        return

    # Load the chosen dataset
    with st.spinner(f"Loading {segment} data from {segment_file.name}..."):
        df = clean_sheet(segment_file)

    if df.empty:
        st.error(f"Could not load data for {segment}. The file exists but contains no valid data.")
        return

    # Display basic info about the dataset
    st.write(f"Dataset shape: {df.shape}")
    if 'Player' in df.columns:
        st.write(f"Number of players: {len(df)}")

    # Show all players that might match our target
    if 'Player' in df.columns:
        st.write("### All players in dataset (first 20):")
        st.write(df['Player'].head(20).tolist())

    # Find the player in the dataset
    found_player, player_data = find_player_in_dataframe(df, player)

    if found_player is None:
        st.error(f"Could not find {player} in this dataset")
        st.write("All available players:")
        st.write(df['Player'].tolist())
        return
    elif found_player != player:
        st.info(f"Using '{found_player}' (found as match for '{player}')")
        player = found_player

    # Filter: drop GP<50 except chosen player (if GP column exists)
    if 'GP' in df.columns:
        original_size = len(df)
        df = df[(df['GP'] >= 50) | (df['Player'] == player)]
        filtered_size = len(df)
        if original_size != filtered_size:
            st.info(f"Filtered to {filtered_size} players (GP ≥ 50 or selected player)")

    # Choose statistic (exclude meta columns)
    exclude = {'Player', 'GP', 'Min', 'W', 'L', 'Age', 'Team', 'G', 'Games'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = [c for c in numeric_cols if c not in exclude and not df[c].isna().all()]

    if not stats:
        st.error("No numeric statistics found in this dataset")
        st.write("Available columns:", df.columns.tolist())
        st.write("Numeric columns:", numeric_cols.tolist())
        return

    stat = st.selectbox("Choose statistic to visualize", stats)

    # Display some info about the selected statistic
    if stat in df.columns:
        player_row = df[df['Player'] == player]
        if len(player_row) > 0:
            player_value = player_row[stat].iloc[0]
            if pd.notna(player_value):
                rank = (df[stat] > player_value).sum() + 1
                total = len(df)
                st.write(f"{player}'s {stat}: **{player_value:.3f}** (Rank: {rank}/{total})")
            else:
                st.warning(f"{player} has no data for {stat}")

    # Create the scatterplot
    try:
        # Remove rows with NaN values for the selected statistic
        plot_df = df.dropna(subset=[stat])

        chart = (
            alt.Chart(plot_df)
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
            .properties(height=min(600, len(plot_df) * 20 + 100), width=800)
            .resolve_scale(y='independent')
        )

        st.altair_chart(chart, use_container_width=True)

        # Show summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean", f"{plot_df[stat].mean():.3f}")
        with col2:
            st.metric("Median", f"{plot_df[stat].median():.3f}")
        with col3:
            st.metric("Std Dev", f"{plot_df[stat].std():.3f}")

        # Show top and bottom 5 players
        st.subheader(f"Rankings by {stat}")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top 5:**")
            top_5 = plot_df.nlargest(5, stat)[['Player', stat]]
            st.dataframe(top_5, hide_index=True)

        with col2:
            st.write("**Bottom 5:**")
            bottom_5 = plot_df.nsmallest(5, stat)[['Player', stat]]
            st.dataframe(bottom_5, hide_index=True)

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.write("Data preview:")
        st.dataframe(df.head())


if __name__ == "__main__":
    main()