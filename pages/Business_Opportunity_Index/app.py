import streamlit as st
import pydeck as pdk
import h3, geopandas as gpd, shapely.geometry as shpg, numpy as np
import boi.storage_csv as bq
import boi.config as cfg

st.title("High Foot-Traffic Hexes  ×  Opportunity Gap")
st.write("Looking in:", cfg.LOCAL_DATA_DIR)
st.write("Found:", [f.name for f in cfg.LOCAL_DATA_DIR.glob("*.csv")])
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
hex_df = bq.read_sql(f"""
    SELECT hex, popularity, local_opportunity
    FROM `{cfg.PROJECT}.{cfg.DATASET}.hex_opportunity`
    WHERE city = '{city}'
      AND local_opportunity >= {thresh}
""")

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
