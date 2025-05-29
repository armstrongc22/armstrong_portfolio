import streamlit as st
import pydeck as pdk
import h3, geopandas as gpd, shapely.geometry as shpg, numpy as np
import boi.storage_csv as bq
import boi.config as cfg

st.title("High Foot-Traffic Hexes  ×  Opportunity Gap")
st.markdown(
        """
        Welcome to the *Business Opportunity Index.  
        This application is designed to identify city segments where opening a new laundromat would meet underserved demand.
        The opportunity score combines demand-side indiccators like population and income with supply-side counts of the number of laundromats and foot-traffic in that area. 
        Further improvement will be increasing the amount of cities and business types available, and adding the apporximate income of the area(estimated by average monthly rent) to the opportunity score formula.
        
        **Method**
        1. **Data Ingestion** - three modules fetch data for each city and stores them locally as CSVs and in the Cloud via Confluent and BigQuery. 
        The data is ingested through the World Bank API, the Open Street Map API, and the Foursquare Places API. From the World Bank we take the total population and GDP per capita for each city.
        Open Street Map allowes us to count how many laundromats are within 10km raidus of the city center. 10km is the most the rate-limit will` allow and future iterations will look to expand the radius of the system.
        And the Foursquare Places API samples 50 points of interest to determiine the populatirty of areas and groups them into 53 resolution-8 haexes. 
        
        2. **Data Storage** - The inital iteration of the BOI sent the data ingested from the three modules to a Confulent Kafka server and then piped that into a BigQuery dataset(bq_sink.py). Later a local CSV structured configuration was designed(storage_csv.py).
        
        3. **Scoring Methodology** - scorer.py creates the opportunity score at two levles. First the city-level opportunity score takes the supply density per 10k people( count of each POI/(population/10,000)).
        Secondly the opportunity score is then calculated by normalizing the gap  in the benchmark of the services populaterity and the per 10k density. (1 -**MIN**((per_10k/benchmark),1)).
        The local opportunity is calculated by taking opportunity score and dividing it by 100 and multiplying it by the popularity of the service in that particular Foursquare hex on the map.
        
        4. **Pipeline(optional)** - there is a function that allows the data to be hosted completely in the cloud. However due to storage costs, a new function was added so that the data is hosted locally. The storage requirements are low but the continued expansion of the BOI may lead push passed acceptable storage thresholds.
        
        5. **Visualization** -The map leverages h3 hex resolution, geopandas/shapely for polygon construction, and pydeck for rendering the WebGL map. 
        
        **Results**
        This project establishes a scalable approach to visualizing the business opportunities in areas that an investor might not be able to reach. Being able to prospect emerging markets and identify what goods and services are in demand can streamline decision making decrease and time to close.  
        
        """
    )
# ── 1  City selector ─────────────────────────────────────────────────────
city = st.selectbox("City", sorted(cfg.CITIES.keys()))

# ── 2  Slider for “Local Opportunity” threshold 0–1 ─────────────────────
thresh = st.slider(
    "Show hexes with local-opportunity ≥",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="local_opportunity = popularity × (laundromat gap score / 100)",
)

# ── 3  Query the joined table (hex_opportunity) ─────────────────────────
hex_df = bq.read_sql("hex_opportunity")
hex_df = hex_df[
    (hex_df.city == city) &
    (hex_df.local_opportunity >= thresh)
]

st.caption(f"Rows fetched after filter ≥ {thresh}: **{len(hex_df)}**")

if hex_df.empty:
    st.info("No hexes meet the threshold. Lower the slider or rerun the pipeline.")
    st.stop()

# ── 4  Helper: H3 cell → shapely.Polygon, (lon,lat) order ───────────────
def hex_to_polygon(cell):
    if hasattr(h3, "cell_to_boundary"):  # v4.x
        try:
            verts = h3.cell_to_boundary(cell)
        except TypeError:
            verts = h3.cell_to_boundary(cell)
    else:                                # v3.x
        verts = h3.h3_to_geo_boundary(cell)

    coords = [(lon, lat) for lat, lon in verts]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return shpg.Polygon(coords)

# ── 5  Build GeoDataFrame + coords column ───────────────────────────────
geometry = [hex_to_polygon(h) for h in hex_df["hex"]]
hex_df["coords"] = [list(poly.exterior.coords) for poly in geometry]
gdf = gpd.GeoDataFrame(hex_df, geometry=geometry, crs="EPSG:4326")

# ── 6  PyDeck layer  (alpha = local_opportunity) ────────────────────────
layer = pdk.Layer(
    "PolygonLayer",
    data=gdf,
    get_polygon="coords",
    get_fill_color="[255, 120, 30, local_opportunity * 255]",
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
    pickable=True,
    auto_highlight=True,
)
TOOLTIP = {
    "html": "<b>Local Opp:</b> {local_opportunity}<br>"
            "<b>Foot traffic:</b> {popularity}",
    "style": {"color": "white"},
}

centre = cfg.CITIES[city]
view = pdk.ViewState(latitude=centre["lat"],
                     longitude=centre["lon"],
                     zoom=11, pitch=30)

st.pydeck_chart(
    pdk.Deck(layers=[layer], initial_view_state=view, tooltip=TOOLTIP)
)
