# app.py
import time
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import PlayerGameLog, PlayByPlayV2


# ─── Helper functions ───────────────────────────────────────────────────────────

@st.cache_data(ttl=24 * 3600)
def get_player_id(full_name: str) -> int:
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No NBA player found for '{full_name}'")
    return matches[0]["id"]


@st.cache_data(ttl=24 * 3600)
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
                (df["EVENTMSGTYPE"] == 1) &  # made FG
                (df["PLAYER2_ID"] == player_id)  # our guy assisted
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
      • center node (primary player)
      • one node per teammate
      • edges weighted by assist count
    """
    G = nx.Graph()
    G.add_node(center_name)
    for teammate, cnt in counts.items():
        G.add_node(teammate)
        G.add_edge(center_name, teammate, weight=cnt)
    return G


def create_clean_visualization(G, center_player, shot_type):
    """
    Creates a clean, well-formatted network visualization
    """
    # Use circular layout for cleaner appearance
    pos = nx.circular_layout(G)

    # Move center player to actual center
    if center_player in pos:
        pos[center_player] = (0, 0)

    # Create figure with better sizing and DPI
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    fig.patch.set_facecolor('white')

    # Node styling
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == center_player:
            node_colors.append('#CE1141')  # Rockets red
            node_sizes.append(800)
        else:
            node_colors.append('#C4CED4')  # Rockets silver/gray
            node_sizes.append(400)

    # Edge styling based on assist counts
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(edge_weights) if edge_weights else 1

    # Normalize edge widths (1-6 range)
    edge_widths = [1 + (weight / max_weight) * 5 for weight in edge_weights]
    edge_colors = ['#000000' for _ in edges]  # Black edges

    # Draw network
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           ax=ax)

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color=edge_colors,
                           alpha=0.6,
                           ax=ax)

    # Add labels with better formatting
    labels = {}
    for node in G.nodes():
        if node == center_player:
            labels[node] = node
        else:
            # Show teammate name and assist count
            weight = G[center_player][node]['weight'] if G.has_edge(center_player, node) else 0
            labels[node] = f"{node}\n({weight})"

    nx.draw_networkx_labels(G, pos, labels,
                            font_size=9,
                            font_weight='bold',
                            font_color='black',
                            ax=ax)

    # Styling
    ax.set_facecolor('white')
    ax.set_title(f"{center_player} - {shot_type} Assist Network",
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CE1141',
                   markersize=15, label=f'{center_player} (Passer)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C4CED4',
                   markersize=12, label='Teammates (Shooters)'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='Assists (thickness = frequency)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    plt.tight_layout()
    return fig


# ─── Streamlit UI ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Rockets Assist Network", layout="wide")

    st.title("🚀 Houston Rockets Assist Network")
    st.markdown("*Visualize assist connections between key Rockets players*")

    # Rockets players only
    rockets_players = [
        "Jalen Green",
        "Amen Thompson",
        "Fred VanVleet",
        "Alperen Sengun"
    ]

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        player = st.selectbox("Select Rockets Player", rockets_players, key="player_select")

    with col2:
        shot_type = st.radio("Assist Type", ["2PT", "3PT"], key="shot_type")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        generate_btn = st.button("🏀 Generate Network", type="primary")

    if generate_btn:
        with st.spinner(f"Fetching {shot_type} assist data for {player}..."):
            try:
                pid = get_player_id(player)
                two_pt, three_pt = fetch_assist_counts(pid)
                counts = two_pt if shot_type == "2PT" else three_pt

                if not counts:
                    st.warning(f"No {shot_type} assists found for {player} this season.")
                    st.info(
                        "This could mean the player hasn't recorded assists for this shot type yet, or there may be limited data available.")
                else:
                    # Filter out very low assist counts for cleaner visualization
                    min_assists = 1
                    filtered_counts = {k: v for k, v in counts.items() if v >= min_assists}

                    if not filtered_counts:
                        st.warning(f"No significant {shot_type} assist connections found for {player}.")
                    else:
                        G = build_graph(player, filtered_counts)

                        # Display summary stats
                        total_assists = sum(filtered_counts.values())
                        top_target = max(filtered_counts.items(), key=lambda x: x[1])

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Assists", total_assists)
                        with col2:
                            st.metric("Teammates Assisted", len(filtered_counts))
                        with col3:
                            st.metric("Top Target", f"{top_target[0]} ({top_target[1]})")

                        # Create and display the visualization
                        fig = create_clean_visualization(G, player, shot_type)
                        st.pyplot(fig, use_container_width=True)

                        # Show detailed breakdown
                        with st.expander("📊 Detailed Breakdown"):
                            df = pd.DataFrame(list(filtered_counts.items()),
                                              columns=['Player', 'Assists'])
                            df = df.sort_values('Assists', ascending=False)
                            st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Please try again or select a different player. Some players may have limited data available.")


if __name__ == "__main__":
    main()