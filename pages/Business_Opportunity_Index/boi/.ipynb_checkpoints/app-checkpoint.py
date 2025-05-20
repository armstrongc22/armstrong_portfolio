import streamlit as st, pydeck as pdk
from boi import storage_bq as bq
import h3, geopandas as gpd, pandas as pd

st.title("High Foot-Traffic Hexes")

city = st.selectbox("City", sorted(bq.cfg.CITIES.keys()))
hex_df = bq.read_sql(
    f"""
    SELECT hex, popularity
    FROM `{bq.cfg.PROJECT}.{bq.cfg.DATASET}.foot_traffic`
    WHERE city = '{city}'
    """
)

if hex_df.empty():
    st.info("No foot-traffic data yet.")
else:
    # Convert H3 → lat/lon polygon for pydeck
    hex_df["geometry"] = hex_df["hex"].apply(
        lambda h: gpd.GeoSeries([h3.h3_to_geo_boundary(h, geo_json=True)]).iloc[0]
    )
    gdf = gpd.GeoDataFrame(hex_df, geometry="geometry", crs="EPSG:4326")

    layer = pdk.Layer(
        "PolygonLayer",
        gdf,
        get_polygon="geometry.coordinates",
        get_fill_color="[255, 140, 0, popularity * 180]",
        pickable=True,
        auto_highlight=True,
    )
    centre = bq.cfg.CITIES[city]
    view = pdk.ViewState(latitude=centre["lat"], longitude=centre["lon"],
                         zoom=11, pitch=30)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
