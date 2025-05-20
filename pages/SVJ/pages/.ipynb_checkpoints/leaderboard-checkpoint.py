# pages/7_leaderboard.py
import streamlit as st
import pandas as pd
from nba_api.stats.static import players

@st.cache_data
def load_leaderboard(path: str = 'leaderboard.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map IDs → full names
    player_list = players.get_players()
    id_to_name = {p['id']: p['full_name'] for p in player_list}
    df['player_name'] = df['player_id'].map(id_to_name)
    # Reorder so name is first
    cols = ['player_name'] + [c for c in df.columns if c != 'player_name']
    return df[cols]

def main():
    st.image("output.png", use_container_width=True)
    st.header("Leaderboard Data")
    df_lb = load_leaderboard()
    st.dataframe(df_lb)

if __name__ == "__main__":
    main()
