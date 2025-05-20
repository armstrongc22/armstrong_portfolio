import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import requests
import io
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
import pgeocode
from pathlib import Path
import json
# ---------- Configuration ----------
DATA_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
st.set_page_config(
    page_title="Cannabis Market Portfolio App",
    page_icon="🌿",
    layout="wide",
)

# ---------- Helpers ----------
def safe_read_csv(path: Path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    st.error(f"❌ Data file not found: `{path.name}` in `{DATA_DIR}`.")
    return pd.DataFrame()

# ---------- Colorado Geo & ZIP→County Lookup ----------
@st.cache_data(show_spinner=False)
def load_co_geo():
    """GeoJSON of Colorado counties (state FIPS 08)."""
    local_path = DATA_DIR / "co_counties.geojson"
    if local_path.exists():
        try:
            gj = json.loads(local_path.read_text())
            feats = [f for f in gj.get("features", []) if f.get("properties", {}).get("STATE") == "08"]
            return {"type": "FeatureCollection", "features": feats}
        except Exception:
            st.warning("⚠️ Failed to parse local co_counties.geojson; will try remote.")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    try:
        gj = requests.get(url, timeout=30).json()
        feats = [f for f in gj.get("features", []) if f.get("properties", {}).get("STATE") == "08"]
        return {"type": "FeatureCollection", "features": feats}
    except Exception:
        st.warning("⚠️ Could not load Colorado county GeoJSON; choropleth disabled.")
        return {"type": "FeatureCollection", "features": []}

@st.cache_data(show_spinner=False)
def co_zip2county() -> dict[str, tuple[str,str]]:
    """Return dict mapping 5‑digit ZIP to (county_lower, county_fips)."""
    # try public gist crosswalk
    gist_url = (
        "https://gist.githubusercontent.com/rogerallen/1583593/raw/"
        "2e1822f4ea5d20ef5ad7e70ffb62a6be7d998e9f/zip_code_database.csv"
    )
    try:
        df = pd.read_csv(gist_url, dtype=str, low_memory=False)
        df_co = df[df["state"] == "CO"].drop_duplicates(subset=["zip"]).copy()
        return {
            z: (row["county"].lower(), row["county_fips"].zfill(5))
            for z, row in df_co.set_index("zip").iterrows()
        }
    except Exception:
        # silent fallback to city-level mapping
        return {}

# ---------- License preprocessing ----------
def preprocess_license(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns=lambda c: c.strip())
    # unify business name
    for col in [
        "Business Name","Licensee Name","Entity Name",
        "Trade Name","Organization Name","Facility Name",
    ]:
        if col in df.columns:
            df = df.rename(columns={col: "Business Name"})
            break
    # derive County & FIPS from Zip Code
    zmap = co_zip2county()
    if "Zip Code" in df.columns and zmap:
        df["County"] = df["Zip Code"].astype(str).str.zfill(5).map(lambda z: zmap.get(z,(None,None))[0])
        df["FIPS"]   = df["Zip Code"].astype(str).str.zfill(5).map(lambda z: zmap.get(z,(None,None))[1])
    # fallback: from City if missing
    if "FIPS" in df.columns:
        missing = df["FIPS"].isna()
    else:
        df["FIPS"] = None
        missing = pd.Series(True, index=df.index)
    if missing.any() and "City" in df.columns:
        city2county = {"denver":"denver","aurora":"arapahoe","colorado springs":"el paso"}
        df.loc[missing, "County"] = df.loc[missing, "City"].str.lower().map(city2county)
        df.loc[missing, "FIPS"] = df.loc[missing, "County"].str.lower().map(co_zip2county())
    if "County" in df.columns:
        df["County"] = df["County"].str.replace(" county","",case=False,regex=False)
    for col in ["Date Updated","LastUpdated","Updated","DATE UPDATED","Date Updated "]:
        if col in df.columns:
            df = df.rename(columns={col:"Date Updated"})
            break
    return df

# ---------- Data loaders ----------
@st.cache_data(show_spinner=False)
def load_wa():
    wa = safe_read_csv(DATA_DIR / "wa_cannabis_sales - Sheet1.csv")
    if wa.empty: return wa
    wa["Sales Last Month"] = wa["Sales Last Month"].replace(r"[\$,]","",regex=True).astype(float)
    wa["Period"] = pd.to_datetime(wa["Period"],errors="coerce")
    return wa
@st.cache_data(show_spinner=False)
def load_wa_data(path: str = 'wa_cannabis_sales - Sheet1.csv') -> pd.DataFrame:
    """Load and preprocess Washington cannabis sales data."""
    df = pd.read_csv(path)
    df['Sales'] = df['Sales Last Month'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Date']    = pd.to_datetime(df['Period'], errors='coerce')
    df['Year']    = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['County_clean'] = (
        df['County']
          .str.replace(r'(?i)\s*county,\s*wa$', '', regex=True)
          .str.replace(r'\s+', '', regex=True)
          .str.lower()
    )
    # map to FIPS
    wa_name2fips = {
        'adams':'53001','asotin':'53003','benton':'53005',
        # ... other counties ...
    }
    df['FIPS'] = df['County_clean'].map(wa_name2fips)
    return df[df['FIPS'].notna()].copy()

@st.cache_data(show_spinner=False)
def update_wa_figures(df: pd.DataFrame, year: int, quarter: int, top_n: int = 20):
    """Return WA choropleth and bar charts for given year, quarter, and top_n."""
    filtered = df[(df.Year == year) & (df.Quarter == quarter)]
    # choropleth
    us_counties = requests.get(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ).json()
    wa_geo = {
        'type': 'FeatureCollection',
        'features': [f for f in us_counties['features'] if f['id'].startswith('53')]
    }
    map_df = filtered.groupby('FIPS', as_index=False)['Sales'].sum()
    fig_map = px.choropleth_mapbox(
        map_df, geojson=wa_geo,
        locations='FIPS', featureidkey='id',
        color='Sales', color_continuous_scale='Greens',
        mapbox_style='carto-positron', zoom=5,
        center={'lat':47.5,'lon':-120.7}, opacity=0.7,
        labels={'Sales':'$ Sales'}
    )
    # license types
    lic = (filtered.groupby('License Type', as_index=False)['Sales']
                .sum().sort_values('Sales', ascending=False))
    fig_lic = px.bar(lic, x='License Type', y='Sales', labels={'Sales':'$ Sales'})
    # top businesses
    biz = (filtered.groupby('Business Name', as_index=False)['Sales']
                .sum().sort_values('Sales', ascending=False).head(top_n))
    fig_biz = px.bar(biz, x='Business Name', y='Sales', labels={'Sales':'$ Sales'})
    return fig_map, fig_lic, fig_biz


@st.cache_data(show_spinner=False)
def load_co_license_data(retail_path: str, medical_path: str) -> pd.DataFrame:
    """Load Colorado license CSVs and compute dominant license type by county."""
    # 1) Read files and label
    df_r = pd.read_csv(retail_path)
    df_r['license_type'] = 'Recreational'
    df_m = pd.read_csv(medical_path)
    df_m['license_type'] = 'Medical'
    df = pd.concat([df_r, df_m], ignore_index=True)

    # 2) ZIP → county via pgeocode
    nomi = pgeocode.Nominatim('us')
    df['Zip Code'] = df['ZIP Code'].astype(str).str.zfill(5)
    df['county_name'] = df['ZIP Code'].apply(lambda z: nomi.query_postal_code(z).county_name)

    # 3) Count and select dominant license per county
    counts = (
        df.groupby(['county_name', 'license_type'])
          .size()
          .rename('count')
          .reset_index()
    )
    idx = counts.groupby('county_name')['count'].idxmax()
    dominant = counts.loc[idx, ['county_name', 'license_type']]
    dominant = dominant.rename(columns={'license_type': 'dominant'})
    return dominant

@st.cache_data(show_spinner=False)
def generate_co_count_map_html(counts_df: pd.DataFrame, title: str) -> str:
    """Generate HTML for a Folium choropleth of license counts by county."""
    # Load Colorado counties
    co = gpd.read_file(
        'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip'
    )
    co = co[co['STATEFP'] == '08'].copy()
    co['county_name'] = co['NAME'].str.lower()

    # Merge counts
    counts_df['county_name'] = counts_df['county_name'].str.lower()
    merged = co.merge(counts_df, on='county_name', how='left')
    merged['count'] = merged['count'].fillna(0)

    # Build map
    m = folium.Map(location=[39.0, -105.5], zoom_start=6, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=json.loads(merged.to_crs(epsg=4326).to_json()),
        data=merged,
        columns=['county_name','count'],
        key_on='feature.properties.county_name',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title
    ).add_to(m)
    folium.GeoJson(
        json.loads(merged.to_crs(epsg=4326).to_json()),
        style_function=lambda feat: {'fillOpacity':0,'color':'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME','count'],
            aliases=['County','Count'],
            localize=True
        )
    ).add_to(m)
    return m.get_root().render()

@st.cache_data(show_spinner=False)
def load_price():
    ppd = safe_read_csv(DATA_DIR / "price_per_ounce_mass_2025.csv")
    if ppd.empty: return ppd
    ppd["CCCLastUpdated"] = pd.to_datetime(ppd["CCCLastUpdated"],errors="coerce")
    melted = ppd.melt(id_vars=["CCCLastUpdated","SOLDDATE"],
                       value_vars=["CONSUMER_OZ","PATIENT_OZ","GRANDTOTAL_OZ"],
                       var_name="Category",value_name="Price_per_oz")
    melted["SOLDDATE"] = pd.to_datetime(melted["SOLDDATE"],errors="coerce")
    return melted

@st.cache_data(show_spinner=False)
def load_state():
    raw = safe_read_csv(DATA_DIR / "Colorado Marijuana_Sales_2014_To_2024_Report.xlsx - State Report.csv",
                        skiprows=2,header=None,dtype=str)
    if raw.empty: return raw
    raw.columns = raw.iloc[0].str.replace("\n"," ").str.strip()
    df = raw.iloc[1:].copy()
    df = df.rename(columns={
        "Total Medical  Marijuana Sales ¹":"Medical_Sales",
        "Total Retail  Marijuana Sales ²":"Retail_Sales",
        "Total  Marijuana Sales":"Total_Sales",
    })
    mask = pd.to_numeric(df["Month"],errors="coerce").notnull() & pd.to_numeric(df["Year"],errors="coerce").notnull()
    df = df[mask].copy()
    df["Month"] = df["Month"].astype(int)
    df["Year"]  = df["Year"].astype(int)
    for col in ["Medical_Sales","Retail_Sales","Total_Sales"]:
        df[col] = df[col].replace(r"[\$,]","",regex=True).astype(float)
    df["Date"] = pd.to_datetime({"year":df["Year"],"month":df["Month"],"day":1})
    return df

@st.cache_data(show_spinner=False)
def load_licenses():
    def _cat(name):
        return preprocess_license(
            safe_read_csv(DATA_DIR / name, parse_dates=["Date Updated"], low_memory=False)
        )
    prod = pd.concat([
        _cat("Product Manufacturers - Retail.csv"),
        _cat("Product Manufacturers - Medical.csv")
    ],ignore_index=True)
    stores = pd.concat([
        _cat("Colorado_Stores - Retail.csv"),
        _cat("Colorado_Stores - Medical.csv")
    ],ignore_index=True)
    cultiv = pd.concat([
        _cat("Colorado_Cultivations - Retail.csv"),
        _cat("Colorado_Cultivations - Medical.csv")
    ],ignore_index=True)
    return prod,stores,cultiv

# ---------- Visualization helper ----------

# ---------- Page builders ----------
def colorado_sales_dashboard(state):
    if state.empty: return
    st.header("Colorado Total Cannabis Sales (2014 – 2024)")
    fig = px.bar(state, x="Date", y="Total_Sales", labels={"Total_Sales":"Sales ($)"})
    st.plotly_chart(fig, use_container_width=True)


def price_dashboard(price_long):
    if price_long.empty: return
    st.header("Average Price per Ounce Over Time")
    cats = price_long["Category"].unique().tolist()
    chosen = st.multiselect("Category", cats, default=cats)
    df = price_long[price_long["Category"].isin(chosen)]
    fig = px.line(df, x="SOLDDATE", y="Price_per_oz", color="Category")
    st.plotly_chart(fig, use_container_width=True)


def washington_dashboard(wa):
    if wa.empty: return
    st.header("Washington State – Sales Last Month")
    county = st.selectbox("County", sorted(wa["County"].dropna().unique()))
    df = wa[wa["County"] == county]
    fig = px.bar(df, x="Business Name", y="Sales Last Month",
                 labels={"Sales Last Month":"Sales ($)"},
                 title=f"{county} County")
    st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("Show raw data"): st.dataframe(df)

def external_viz_dashboard():
    """Embed external Tableau dashboards for OR, IL, and CA."""
    st.header("State-Sourced Cannabis Dashboards")
    tab_or, tab_il, tab_ca = st.tabs(["Oregon", "Illinois", "California"])

    # ── Oregon ──
    oregon_html = """
<script type='module' src='https://data.olcc.state.or.us/javascripts/api/tableau.embedding.3.latest.min.js'></script>
<tableau-viz id='olcc-prices'
    src='https://data.olcc.state.or.us/t/OLCCPublic/views/MarketDataTableau/Prices'
    width='800' height='850' toolbar='bottom'>
</tableau-viz>
"""
    with tab_or:
        components.html(oregon_html, height=900, scrolling=True)

    # ── Illinois ──
    illinois_html = """
<script type='module' src='https://public.data.illinois.gov/javascripts/api/tableau.embedding.3.latest.min.js'></script>
<tableau-viz id='il-wholesale'
    src='https://public.data.illinois.gov/t/Public/views/CannabisSalesFigures/Wholesale'
    width='900' height='667' toolbar='bottom'>
</tableau-viz>
"""
    with tab_il:
        components.html(illinois_html, height=700, scrolling=True)

    # ── California ──
    california_html = """
<div class='tableauPlaceholder' id='viz_ca'>
  <noscript><a href='#'><img alt='' src='https://public.tableau.com/static/images/Ha/HarvestReport_17064223647370/HarvestReportbyCultivationType/1_rss.png' style='border:none'/></a></noscript>
  <object class='tableauViz' style='display:none;'>
    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F'/>
    <param name='embed_code_version' value='3'/>
    <param name='site_root' value=''/>
    <param name='name' value='HarvestReport_17064223647370/HarvestReportbyCultivationType'/>
    <param name='tabs' value='yes'/>
    <param name='toolbar' value='yes'/>
    <param name='static_image' value='https://public.tableau.com/static/images/Ha/HarvestReport_17064223647370/HarvestReportbyCultivationType/1.png'/>
    <param name='animate_transition' value='yes'/>
    <param name='display_static_image' value='yes'/>
    <param name='display_spinner' value='yes'/>
    <param name='display_overlay' value='yes'/>
    <param name='display_count' value='yes'/>
    <param name='language' value='en-US'/>
  </object>
</div>
<script type='text/javascript'>
  var divElement = document.getElementById('viz_ca');
  var vizElement = divElement.getElementsByTagName('object')[0];
  if (divElement.offsetWidth > 800) {
    vizElement.style.width='100%'; vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
  } else if (divElement.offsetWidth > 500) {
    vizElement.style.width='100%'; vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
  } else {
    vizElement.style.width='100%'; vizElement.style.height='2250px';
  }
  var scriptElement = document.createElement('script');
  scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
  vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
"""
    with tab_ca:
        components.html(california_html, height=900, scrolling=True)

def license_dashboard(prod: pd.DataFrame, stores: pd.DataFrame, cultiv: pd.DataFrame):
    """Render three choropleth tabs for license counts by county, mapping ZIP to county manually."""
    if prod.empty or stores.empty or cultiv.empty:
        st.info("At least one license CSV couldn’t be loaded—check earlier error messages.")
        return

    st.header("Colorado Licenses Snapshot – Choropleth by County")
    tabs = st.tabs(["Manufacturers","Stores","Cultivations"])

    for df, tab, title in zip([prod,stores,cultiv], tabs, ["Manufacturers","Stores","Cultivations"]):
        with tab:
            st.metric(f"Total {title}", f"{len(df):,}")

                        # ZIP→County mapping (using co_zip2county or fallback to pgeocode)
            zip_map = co_zip2county()
            if zip_map:
                df['county_name'] = (
                    df['ZIP Code'].astype(str).str.zfill(5)
                      .map(lambda z: zip_map.get(z, (None,None))[0])
                )
            else:
                import pgeocode
                nomi = pgeocode.Nominatim('us')
                # Gracefully handle non-string county_name (nan floats)
                def zip_to_county(z):
                    rec = nomi.query_postal_code(z)
                    cn = getattr(rec, 'county_name', None)
                    return cn.lower() if isinstance(cn, str) else None
                df['county_name'] = (
                    df['ZIP Code'].astype(str).str.zfill(5)
                      .apply(zip_to_county)
                )
            missing = df['county_name'].isna().sum()
            if missing:
                st.warning(f"{missing} records missing county mapping; some licenses may be excluded.")
            df_clean = df.dropna(subset=['county_name'])

            # Count licenses per county
            counts = df_clean.groupby('county_name').size().reset_index(name='count')

            # Generate and display choropleth
            map_html = generate_co_count_map_html(counts, f"{title} Licenses by County")
            components.html(map_html, height=600, scrolling=False)

            with st.expander("Raw data"):
                st.dataframe(df)

# ---------- Main ----------

def main():
    # Load data for all dashboards
    wa = load_wa_data()
    price_long = load_price()
    state = load_state()
    prod, stores, cultiv = load_licenses()

    # Sidebar selection
    page = st.sidebar.radio(
        "Choose a dashboard",
        [
            "Colorado Sales Trends",
            "Price per Ounce",
            "Washington County Sales",
            "Colorado Licenses Overview",
            "State-Sourced Dashboards",
        ],
    )

    # Render selected dashboard
    if page == "Colorado Sales Trends":
        colorado_sales_dashboard(state)
    elif page == "Price per Ounce":
        price_dashboard(price_long)
    elif page == "Washington County Sales":
        washington_dashboard(wa)
    elif page == "Colorado Licenses Overview":
        license_dashboard(prod, stores, cultiv)
    elif page == "State-Sourced Dashboards":
        external_viz_dashboard()

if __name__ == "__main__":
    main()