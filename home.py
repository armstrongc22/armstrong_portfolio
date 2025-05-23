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
        "name": "Cannabis Market Research",
        "thumb": IMG_DIR / "canna.png",
        "url": "https://armstrongportfolio-rj6cipckjilm3vdbacdvrp.streamlit.app"
    },
    {
        "name": "Business Opportunity Index",
        "thumb": IMG_DIR / "boi.png",
        "url": "https://armstrongportfolio-mq3htsuzxwozs2gmfwgrre.streamlit.app"
    },
    {
        "name": "Synthetic Data Marketing(Desktop Only)",
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
