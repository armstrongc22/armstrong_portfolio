import streamlit as st
from pathlib import Path
from pages import (
    player_stats,
    logistic_model,
    point_distribution,
    shot_distribution,
    leaderboard
)

st.set_page_config(page_title="Rockets Analytics Hub", layout="wide")
# 3️⃣ Project description

# --- Sidebar selector with a "Home" default ---
PAGES = {
    "Player Stats": player_stats.main,
    "Logistic Model": logistic_model.main,
    "Point Distribution": point_distribution.main,
    "Shot Distribution": shot_distribution.main,
    "Leaderboard": leaderboard.main,
}
choice = st.sidebar.radio(
    "Choose Dashboard:",
    ["Home"] + list(PAGES),
    index=0,  # default to "Home"
)

# --- Home content ---
if choice == "Home":
    st.subheader("Project Description")
    st.write(
        """
        For the last 4 years Rockets fans … fundamental basketball principles.
        """
    )
    # if you still want the big banner on Home:
    img = Path(__file__).resolve().parent / "pages" / "output.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("Banner image not found; remove or fix the path if you don’t need it.")

# --- Delegate to the other pages only when picked ---
else:
    PAGES[choice]()