# pages/1_Home.py
import streamlit as st
from pathlib import Path

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Armstrong Portfolio",
    layout="wide",
)

# ─── Title & Intro ─────────────────────────────────────────────────────────────
st.title("🚀 Armstrong’s Data Portfolio")
st.markdown(
    """
    Welcome!  Here you’ll find four deep-dive dashboard projects I’ve built  
    in Streamlit—click any card below to explore.
    """,
    unsafe_allow_html=True
)

# ─── Card definitions ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent

# make sure you put your gradient thumbs next to this file (or adjust these paths)
projects = [
    {
        "name": "SVJ Cornerstone Debate",
        "thumb": BASE / "rockets.png",
        "url": "https://armstrongportfolio-9xwrmtbknbm2cnrtcvvpmq.streamlit.app"
    },
    {
        "name": "Cannabis Market Research",
        "thumb": BASE / "canna.png",
        "url": "https://armstrongportfolio-rj6cipckjilm3vdbacdvrp.streamlit.app"
    },
    {
        "name": "Business Opportunity Index",
        "thumb": BASE / "boi.png",
        "url": "https://armstrongportfolio-mq3htsuzxwozs2gmfwgrre.streamlit.app"
    },
    {
        "name": "Synthetic Data Marketing",
        "thumb": BASE / "neymar.png",
        "url": "https://armstrongportfolio-febhp4fxhsde5csvsgzrbx.streamlit.app/"
    },
]

# ─── Render cards in a 2×2 grid ─────────────────────────────────────────────────
cols = st.columns(2, gap="large")
for idx, proj in enumerate(projects):
    col = cols[idx % 2]
    # clickable image + caption
    col.markdown(
        f"""
        <a href="{proj['url']}" target="_blank" style="text-decoration:none">
          <img src="file://{proj['thumb']!s}" 
               style="width:100%; border-radius:8px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);" />
          <h3 style="text-align:center; margin-top:0.5em; color:#333;">{proj['name']}</h3>
        </a>
        """,
        unsafe_allow_html=True,
    )
