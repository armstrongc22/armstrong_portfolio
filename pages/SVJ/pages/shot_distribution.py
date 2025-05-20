import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players, teams

player_dict = players.get_players()
team_dict = teams.get_teams()

def get_player_id(name):
    return [p['id'] for p in player_dict if p['full_name'] == name][0]

def get_team_id2(abbr):
    return [t['id'] for t in team_dict if t['abbreviation'] == abbr][0]

def create_court(ax, color='black'):
    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
    ax.plot([220, 220], [0, 140], linewidth=2, color=color)
    ax.add_artist(mpl.patches.Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))
    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
    ax.plot([80, 80], [0, 190], linewidth=2, color=color)
    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
    ax.plot([60, 60], [0, 190], linewidth=2, color=color)
    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
    ax.add_artist(mpl.patches.Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))
    ax.add_artist(mpl.patches.Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))
    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)
    ax.set_xlim(-250, 250); ax.set_ylim(0, 470)
    ax.set_xticks([]); ax.set_yticks([])

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
PLAYERS = {"Alperen Sengun": get_player_id("Alperen Sengun"), "Jalen Green": get_player_id("Jalen Green")}

def main():
    st.image("output.png", use_container_width=True)
    st.header("Shot Chart Distribution by Season")
    player_name = st.selectbox("Player", list(PLAYERS.keys()))
    pid = PLAYERS[player_name]
    team_id = get_team_id2('HOU')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, season in zip(axes.flatten(), SEASONS):
        data = ShotChartDetail(
            team_id=team_id,
            player_id=pid,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        x = data['LOC_X']; y = data['LOC_Y'] + 60
        create_court(ax)
        ax.hexbin(x, y, gridsize=30, bins='log', cmap='Blues', extent=(-250,250,0,470))
        ax.set_title(f"{player_name} – {season}")
    st.pyplot(fig)


if __name__ == "__main__":
    main()