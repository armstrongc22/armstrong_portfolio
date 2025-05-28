# nuclear/nuclear.py
import streamlit as st

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Nuclear Energy Dashboard",
    layout="wide",
)

# Import scripts AFTER set_page_config
from scripts import growing, performance, pipeline, capacity

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Home", "Growth Atlas", "Supply-Chain Risk", "Performance Benchmark/Opportunity", "Deal Pipeline")
)

if page == "Home":
    st.title("🏠 Nuclear Energy Insights")
    st.markdown(
        """
        Welcome to the **Nuclear Energy Dashboard**.  
        Use the menu on the left to explore:
        1. **Growth Atlas** – Identify the fastest-growing nuclear markets.  
        2. **Supply-Chain Risk** – Uranium feed production vs reactor demand. 
        3. **Performance Benchmark/Opportunity** - Where increase capacity leads to largest service opportunities. 
        4. **Deal Pipeline** – Live reactor financings with sovereign & ECA overlays.  
        """
    )
    st.image("https://www.world-nuclear.org/getmedia/0a212cba-1a5f-4de6-9d7a-5efb9728f691/World-Map-of-Nuclear-Power-Reactors.png", caption="Global Nuclear Reactor Map", use_column_width=True)

elif page == "Growth Atlas":
    growing.main()
    
elif page == "Supply-Chain Risk":
    capacity.main()
    
elif page == "Performance Benchmark/Opportunity":
    performance.main()
    
elif page == "Deal Pipeline":
    pipeline.main()