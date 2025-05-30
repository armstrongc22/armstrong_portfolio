# app.py
import time
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import PlayerGameLog, PlayByPlayV2

# ─── Helper functions ───────────────────────────────────────────────────────────

@st.cache_data(ttl=24*3600)
def get_player_id(full_name: str) -> int:
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No NBA player found for '{full_name}'")
    return matches[0]["id"]

@st.cache_data(ttl=24*3600)
def fetch_assist_counts(player_id: int,
                        season: str = "2024-25",
                        season_type: str = "Regular Season"):
    """
    Returns two dicts:
      - two_pt: {shooter_name: assists_on_2PT}
      - three_pt: {shooter_name: assists_on_3PT}
    """
    # 1) games played
    gl = PlayerGameLog(player_id=player_id,
                       season=season,
                       season_type_all_star=season_type)
    games = gl.get_data_frames()[0]["Game_ID"].unique()

    two_pt = {}
    three_pt = {}

    # 2) play-by-play scan
    for gid in games:
        pbp = PlayByPlayV2(game_id=gid)
        df = pbp.get_data_frames()[0]

        mask = (
            (df["EVENTMSGTYPE"] == 1) &        # made FG
            (df["PLAYER2_ID"] == player_id)    # our guy assisted
        )
        for _, row in df.loc[mask].iterrows():
            desc = row["HOMEDESCRIPTION"] or row["VISITORDESCRIPTION"] or ""
            shooter = row["PLAYER1_NAME"]
            if "3PT" in desc:
                three_pt[shooter] = three_pt.get(shooter, 0) + 1
            else:
                two_pt[shooter] = two_pt.get(shooter, 0) + 1

        time.sleep(0.6)  # rate-limit

    return two_pt, three_pt

def build_graph(center_name: str, counts: dict):
    """
    Builds a NetworkX graph with:
      • center node (red)
      • one node per teammate (black)
      • edges weighted by assist count
    """
    G = nx.Graph()
    G.add_node(center_name)
    for teammate, cnt in counts.items():
        G.add_node(teammate)
        G.add_edge(center_name, teammate, weight=cnt)
    return G

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("NBA Assist Network")

# 1) Player selection
all_names = [p["full_name"] for p in players.get_active_players()]
player = st.selectbox("Pick a player", sorted(all_names))

# 2) Shot type
shot_type = st.radio("Assist on…", ["2PT", "3PT"])

# 3) Generate
if st.button("Show Network"):
    with st.spinner("Fetching data…"):
        try:
            pid = get_player_id(player)
            two_pt, three_pt = fetch_assist_counts(pid)
            counts = two_pt if shot_type=="2PT" else three_pt

            if not counts:
                st.warning(f"No {shot_type} assists found for {player}.")
            else:
                G = build_graph(player, counts)
                pos = nx.spring_layout(G, seed=42)

                # colors & sizes
                node_colors = ["red" if n==player else "black" for n in G.nodes()]
                node_sizes  = [300 if n==player else 100 + G[player].get(n,{}).get("weight",0)*20
                               for n in G.nodes()]

                # edge widths by weight
                edge_widths = [G[u][v]["weight"] for u,v in G.edges()]

                # draw
                fig, ax = plt.subplots(figsize=(8,8))
                nx.draw_networkx(
                    G, pos,
                    ax=ax,
                    node_color=node_colors,
                    node_size=node_sizes,
                    width=edge_widths,
                    with_labels=True,
                    font_size=9
                )
                ax.set_axis_off()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
if __name__ == "__main__":
    main()