import streamlit as st
from pathlib import Path
from pages import (
    player_stats,
    logistic_model,
    point_distribution,
    shot_distribution,
    leaderboard, comparison
)

st.set_page_config(page_title="Rockets Analytics Hub", layout="wide")
img = Path(__file__).resolve().parent / "pages" / "output.png"
if img.exists():
        st.image(str(img), use_container_width=True)
else:
        st.info("Banner image not found; remove or fix the path if you don’t need it.")
# 3️⃣ Project description

# --- Sidebar selector with a "Home" default ---
PAGES = {
    "Player Stats": player_stats.main,
    "Player Stats 2": comparison.main, 
    "Logistic Model": logistic_model.main,
    "Point Distribution": point_distribution.main,
    "Shot Distribution": shot_distribution.main,
    "Leaderboard": leaderboard.main,
}

# --- Home content ---

st.subheader("Project Description")
st.write(
        """
        For the last 4 years Rockets fans and the NBA lexicon as a whole have passionately argued the merits of both Jalen Green and Alperen Sengun's viability as franchise cornerstones. The impact of their success or failure in this role will shape the NBA's 5th most valuable franchise for the next 10 years, and leveraging the team one way or another in this regard, is a potentially multi-billion dollar decision. This project endeavors to weigh the facts dispassionately, and deliver a verdict on the debate based in statistical analysis, decision mathematics, and fundamental basketball principles.
        """
    )
    # if you still want the big banner on Home:

# --- Delegate to the other pages only when picked ---