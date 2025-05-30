# pages/1_Home.py
import streamlit as st
from pathlib import Path

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Armstrong Portfolio",
    layout="wide",
)

# ─── Title & Intro ─────────────────────────────────────────────────────────────
st.title("🚀 Christian Armstrong’s Data Portfolio")
st.markdown(
    """
    Welcome! Below you will find data projects that showcase my committment to distilling and synthesizing actionable insights from data.
    This versatile portfolio uses a variety of tools to simplify complex problems in sports, market research, business intelligence, and marketing.
    My hope is that in one of these you will find a relatable problem that I have shown the capacity to solve using statistical, geospatial, or visual analysis.
    Perhaps a mix of all three! I am confident that whatever data related issue you find yourself needing a data scientist for, I can be of service. Please email me at cvarmstrong1993@gmail.com or phalanxneymaranalytics@gmail.com for queries or more information. 
    
    If you see a snoozing emoji simply click the blue button to wake the app back up and everything should work fine. For the Segment section of the synthetic data project I recommend checking the box to complete the analysis to see the full visualizaiton. 
    """,
    unsafe_allow_html=True
)

# ─── Card definitions ────────────────────────────────────────────────────────────

# this points at whatever folder home.py lives in
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR  = BASE_DIR / "images"

# make sure you put your gradient thumbs next to this file (or adjust these paths)
projects = [
    {
        "name": "SVJ Cornerstone Debate",
        "thumb": IMG_DIR / "rockets.png",
        "url": "https://armstrongportfolio-9xwrmtbknbm2cnrtcvvpmq.streamlit.app"
    },
    {
        "name": "Nuclear Energy Market Research",
        "thumb": IMG_DIR / "canna.png",
        "url": "https://armstrongportfolio-7jczhswxsuebtugxdhlpkp.streamlit.app/"
    },
    {
        "name": "Business Opportunity Index",
        "thumb": IMG_DIR / "boi.png",
        "url": "https://armstrongportfolio-mq3htsuzxwozs2gmfwgrre.streamlit.app"
    },
    {
        "name": "Synthetic Data Marketing",
        "thumb": IMG_DIR / "neymar.png",
        "url": "https://armstrongportfolio-febhp4fxhsde5csvsgzrbx.streamlit.app/"
    },
]

# ─── Render cards in a 2×2 grid ─────────────────────────────────────────────────
# ─── Render them in two columns ─────────────────────────────────────────────────
cols = st.columns(2, gap="large")
for idx, project in enumerate(projects):
    col = cols[idx % 2]
    thumb = project["thumb"]
    if not thumb.exists():
        col.error(f"❌ Image not found: {thumb.name}")
        continue

    # 1) show the thumbnail
    col.image(str(thumb), use_container_width=True)

    # 2) make the title itself a hyperlink
    col.markdown(f"[**{project['name']}**]({project['url']})")
