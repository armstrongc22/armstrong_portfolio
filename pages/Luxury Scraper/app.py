import streamlit as st
import asyncio
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from scrapers.ebay_scraper import EbayScraper
from services.email_service import EmailService
from models.database import Item, MarketValue, PriceHistory, engine, SessionLocal
import base64
from sqlalchemy.orm import Session

# Page configuration
st.set_page_config(
    page_title="Luxury Item Price Tracker",
    page_icon="✨",
    layout="wide"
)

# Custom CSS for white and gold theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --gold: #D4AF37;
        --light-gold: #F4E5B5;
        --dark-gold: #996515;
    }

    /* Main content styling */
    .stApp {
        background-color: white;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--dark-gold) !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #D4AF37, #996515) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox {
        border-color: var(--light-gold) !important;
    }

    /* Metrics */
    .stMetric {
        background-color: white !important;
        border: 1px solid var(--light-gold) !important;
        border-radius: 5px !important;
        padding: 1rem !important;
    }

    /* Cards */
    .element-container {
        background-color: white !important;
    }

    /* Dataframe */
    .dataframe {
        border: 1px solid var(--light-gold) !important;
    }

    /* Success message */
    .success {
        padding: 1rem;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
email_service = EmailService()
ebay_scraper = EbayScraper()

# Session state initialization
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Create a new event loop for async operations
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Title and description
st.title("✨ Luxury Item Price Tracker")
st.markdown("""
Track prices of luxury items across multiple platforms and get notified about the best deals.
Currently supporting luxury watches, with more categories coming soon!
""")

# Sidebar for search configuration
with st.sidebar:
    st.header("Search Settings")
    search_term = st.text_input(
        "Search Term",
        placeholder="e.g., Rolex Submariner",
        help="Enter the watch model you want to search for"
    )
    
    category = st.selectbox(
        "Category",
        ["watches", "art (coming soon)", "jewelry (coming soon)", "clothes (coming soon)"],
        index=0,
        disabled=True  # Simply disable it since we only support watches for now
    )
    
    threshold = st.slider(
        "Deal Threshold (%)",
        min_value=1,
        max_value=99,
        value=20,
        help="Minimum percentage below market value to be considered a good deal"
    )
    
    enable_notifications = st.checkbox(
        "Enable Email Notifications",
        value=True,
        help="Get notified when exceptional deals are found"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Search Deals", use_container_width=True):
        if not search_term:
            st.error("Please enter a search term")
        else:
            with st.spinner("Searching for deals..."):
                # Run the scraper
                items = run_async(ebay_scraper.scrape(search_term))
                
                if not items:
                    st.warning("No items found. Try a different search term.")
                else:
                    # Process items and get market values
                    for item in items:
                        if item.get('model'):
                            market_value = run_async(ebay_scraper.get_market_value(item['model']))
                            if market_value:
                                item['market_value'] = market_value
                                item['is_good_deal'] = ebay_scraper.is_good_deal(
                                    item['price'],
                                    market_value,
                                    threshold / 100
                                )
                                
                                # Send email for good deals if enabled
                                if enable_notifications and item['is_good_deal']:
                                    run_async(email_service.send_deal_alert(item))

                    # Store results in database
                    db = SessionLocal()
                    for item in items:
                        db_item = Item(
                            title=item['title'],
                            price=item['price'],
                            url=item['url'],
                            platform=item['platform'],
                            category=item['category'],
                            image_url=item['image_url'],
                            condition=item['condition'],
                            seller=item['seller'],
                            description=item['description'],
                            model=item['model'],
                            brand=item['brand'],
                            is_good_deal=item.get('is_good_deal', False),
                            market_value=item.get('market_value'),
                            deal_percentage=((item.get('market_value', 0) - item['price']) / item.get('market_value', 1)) if item.get('market_value') else None
                        )
                        db.add(db_item)
                    
                    db.commit()
                    db.close()

                    # Update session state
                    st.session_state.search_history = items

                    # Display results
                    good_deals = [item for item in items if item.get('is_good_deal', False)]
                    if good_deals:
                        st.success(f"Found {len(good_deals)} good deals!")

                    # Display items in a modern card layout
                    for item in items:
                        with st.container():
                            col_img, col_details = st.columns([1, 2])
                            
                            with col_img:
                                if item['image_url']:
                                    st.image(item['image_url'], use_column_width=True)
                            
                            with col_details:
                                st.subheader(item['title'])
                                st.write(f"💰 Price: ${item['price']:,.2f}")
                                if item.get('market_value'):
                                    st.write(f"📊 Market Value: ${item['market_value']:,.2f}")
                                    savings = ((item['market_value'] - item['price']) / item['market_value'] * 100)
                                    if item.get('is_good_deal'):
                                        st.success(f"🔥 Good Deal! Save {savings:.1f}%")
                                    else:
                                        st.warning(f"Regular Price ({savings:.1f}% difference)")
                                
                                st.write(f"✨ Condition: {item['condition']}")
                                st.write(f"👤 Seller: {item['seller']}")
                                st.write(f"🏢 Platform: {item['platform']}")
                                
                                if st.button(f"View Item →", key=item['url']):
                                    st.markdown(f"[Open in new tab]({item['url']})")
                            
                            st.markdown("---")

with col2:
    if st.session_state.search_history:
        st.header("Price Analysis")
        
        # Create price distribution plot
        prices = [item['price'] for item in st.session_state.search_history]
        market_values = [item.get('market_value', 0) for item in st.session_state.search_history]
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=prices,
            name="Listed Prices",
            marker_color="#D4AF37"
        ))
        fig.add_trace(go.Box(
            y=[mv for mv in market_values if mv > 0],
            name="Market Values",
            marker_color="#996515"
        ))
        
        fig.update_layout(
            title="Price Distribution",
            yaxis_title="Price ($)",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Playfair Display")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.header("Quick Stats")
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.metric(
                "Average Price",
                f"${sum(prices)/len(prices):,.2f}",
                delta=None
            )
            
        with col_stats2:
            good_deals_count = len([item for item in st.session_state.search_history if item.get('is_good_deal', False)])
            st.metric(
                "Good Deals Found",
                good_deals_count,
                delta=None
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with ❤️ by Your Name | 
    <a href="https://github.com/yourusername" style="color: #D4AF37;">GitHub</a>
</div>
""", unsafe_allow_html=True) 