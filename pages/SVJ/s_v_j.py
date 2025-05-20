import streamlit as st
from pages import (
    player_stats,
    logistic_model,
    point_distribution,
    shot_distribution,
    leaderboard
)

st.set_page_config(page_title="Rockets Analytics Hub", layout="wide")
# 3️⃣ Project description
st.title("🔥Green vs. Goon: Battle for Cornerstone Supremacy🔥")

def main():
    st.subheader("Project Description")
    st.text(
        "For the last 4 years Rockets fans and the NBA lexicon as a whole have passionately argued the merits of both Jalen Green and Alperen Sengun's viability as franchise cornerstones. The impact of their success or failure in this role will shape the NBA's 5th most valuable franchise for the next 10 years, and leveraging the team one way or another in this regard, is a potentially multi-billion dollar decision. This project endeavors to weigh the facts dispassionately, and deliver a verdict on the debate based in statistical analysis, decision mathematics, and fundamental basketball principles."
    )
    selection = st.sidebar.radio(
        "Choose Dashboard:",
        [
            "Player Stats",
            "Logistic Model",
            "Point Distribution",
            "Shot Distribution",
            "Leaderboard"
        ]
    )

    if selection == "Player Stats":
        player_stats.main()
    elif selection == "Logistic Model":
        logistic_model.main()
    elif selection == "Point Distribution":
        point_distribution.main()
    elif selection == "Shot Distribution":
        shot_distribution.main()
    elif selection == "Leaderboard":
        leaderboard.main()


if __name__ == "__main__":
    main()