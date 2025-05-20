import streamlit as st
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import PlayerGameLogs

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
PLAYERS = {"Alperen Sengun": 1630174, "Jalen Green": 1630224}


def main():
    st.image("output.png", use_container_width=True)
    st.header("Points-per-Game Distribution by Season")
    player_name = st.selectbox("Select Player", list(PLAYERS.keys()))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, season in zip(axes.flatten(), SEASONS):
        logs = PlayerGameLogs(
            season_nullable=season,
            player_id_nullable=PLAYERS[player_name],
            season_type_nullable="Regular Season"
        ).get_data_frames()[0]
        ax.hist(logs["PTS"].dropna(), bins=12)
        ax.set_title(f"{player_name} – {season}")
        ax.set_xlabel("Points")
        ax.set_ylabel("Games")
        ax.grid(True)
    st.pyplot(fig)


if __name__ == "__main__":
    main()