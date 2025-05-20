import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs
from nba_api.stats.static import players

@st.cache_data
def load_player_logs(player_name: str, season: str) -> pd.DataFrame:
    pid = players.find_players_by_full_name(player_name)[0]["id"]
    df = PlayerGameLogs(
        season_nullable=season,
        player_id_nullable=pid,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]
    df = df[["GAME_DATE", "PTS", "MIN"]]
    df["MIN"] = pd.to_numeric(df["MIN"], errors='coerce')
    return df.dropna()


def main():
    st.image("output.png", use_container_width=True)
    st.header("Player Performance Over Season")
    player = st.selectbox("Player", ["Alperen Sengun", "Jalen Green"])
    season = st.selectbox("Season", ["2024-25", "2023-24"])
    df_player = load_player_logs(player, season)

    st.line_chart(df_player.set_index("GAME_DATE")["PTS"], use_container_width=True)
    st.bar_chart(df_player.set_index("GAME_DATE")["MIN"], use_container_width=True)


if __name__ == "__main__":
    main()