# pages/7_leaderboard.py
import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from pathlib import Path

@st.cache_data
def load_leaderboard():
    base = Path(__file__).resolve().parent  # points to pages/SVJ/pages
    path = base / "leaderboard.csv"  # or whatever the actual filename is
    df = pd.DataFrame(path)
    # Map IDs → full names
    player_list = players.get_players()
    id_to_name = {p['id']: p['full_name'] for p in player_list}
    df['player_name'] = df['player_id'].map(id_to_name)
    # Reorder so name is first
    cols = ['player_name'] + [c for c in df.columns if c != 'player_name']
    return df[cols]

def main():
    BASE = Path(__file__).resolve().parent  # pages/SVJ/pages
    img_path = BASE / "output.png"
    st.image(str(img_path), use_container_width=True)
    st.header("Leaderboard Data")
    df_lb = load_leaderboard()
    st.dataframe(df_lb)

if __name__ == "__main__":
    main()
