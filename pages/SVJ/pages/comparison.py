import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# ─── shared clean + loader functions ┎┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉┉
# adjust these BASE paths to your download folders
tBASE = Path(r"C:/Users/Armstrong/NBA Projects/sengun_vs_jalen")
tGUARD = tBASE / "jalen_distance"
tCENTER = tBASE / "sengun_distance"

def clean_sheet(path: Path) -> pd.DataFrame:
    """
    Read the CSV, realign the first 7 metadata columns into Player/GP,
    then drop the originals and any unnamed/duplicate cols.
    """
    df = pd.read_csv(path)
    # fix misaligned header: 2nd col is name, 4th col is GP
    df['Player'] = df.iloc[:, 1].astype(str)
    df['GP']     = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    # drop the first 7 original cols: ord, name, team, age, GP, W, L
    df = df.drop(columns=df.columns[0:7])
    # drop any Unnamed or duplicate columns
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# individual loaders just point at their file:
@st.cache_data
def load_guard_basic():      return clean_sheet(tGUARD / "guard_basic.csv")
@st.cache_data
def load_guard_advanced():   return clean_sheet(tGUARD / "guard_advanced.csv")
@st.cache_data
def load_isolation():        return clean_sheet(tBASE  / "isolation.csv")
@st.cache_data
def load_pnr_handler():     return clean_sheet(tBASE  / "pnr_handler.csv")
@st.cache_data
def load_centers_basics():   return clean_sheet(tCENTER / "centers_basics.csv")
@st.cache_data
def load_centers_advanced(): return clean_sheet(tCENTER / "centers_advanced.csv")
@st.cache_data
def load_post_ups():         return clean_sheet(tBASE  / "post_ups.csv")
@st.cache_data
def load_pnr_big():          return clean_sheet(tBASE  / "pnr_big.csv")
def load_guard_basic():
    df = pd.read_csv("guard_basic.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_guard_advanced():
    df = pd.read_csv("guard_advanced.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_isolation():
    df = pd.read_csv("isolation.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_pnr_handler():
    df = pd.read_csv("pnr_handler.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_centers_basics():
    df = pd.read_csv("centers_basics.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_centers_advanced():
    df = pd.read_csv("centers_advanced.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_post_ups():
    df = pd.read_csv("post_ups.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def load_pnr_big():
    df = pd.read_csv("pnr_big.csv", header=0)
    df.columns = df.columns.str.strip().str.replace(r"Unnamed:.*", "", regex=True)
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ─── Streamlit App ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Player Segmentation & Ranking Explorer")

    # 1) Player selection
    player = st.selectbox("Choose player", ["Jalen Green", "Alperen Sengun"])

    # 2) Segment-to-function mapping
    func_map = {
        "Guard Basic":      load_guard_basic,
        "Guard Advanced":   load_guard_advanced,
        "Isolation":        load_isolation,
        "Pick & Roll Handler": load_pnr_handler,
        "Center Basic":     load_centers_basics,
        "Center Advanced":  load_centers_advanced,
        "Post Ups":         load_post_ups,
        "Pick & Roll Big":  load_pnr_big
    }
    # 3) Available segments per player
    segs = {
        "Jalen Green": ["Guard Basic","Guard Advanced","Isolation","Pick & Roll Handler"],
        "Alperen Sengun": ["Center Basic","Center Advanced","Post Ups","Isolation","Pick & Roll Big"]
    }

    # 4) Segment selection
    choices = segs[player]
    segment = st.selectbox("Choose segment", choices)

    # 5) Load the chosen dataset
    df = func_map[segment]()

    # 6) Ensure 'Player' column exists
    if 'Player' not in df.columns:
        # assume the first column is player names
        df = df.rename(columns={df.columns[0]: 'Player'})

    # 7) Filter: drop GP<50 except chosen player
    if 'GP' in df.columns:
        df = df[(df['GP'] >= 50) | (df['Player'] == player)]

    # 8) Choose statistic (exclude meta columns)
    exclude = {'Player','GP','Min','W','L'}
    stats = [c for c in df.select_dtypes(include='number').columns if c not in exclude]
    stat = st.selectbox("Choose statistic to visualize", stats)

    # 9) Scatterplot
    chart = (
        alt.Chart(df)
           .mark_circle()
           .encode(
               x=alt.X(f"{stat}:Q", title=stat),
               y=alt.Y("Player:N", sort=alt.EncodingSortField(stat, order="descending")),
               color=alt.condition(
                   alt.datum.Player == player, alt.value("red"), alt.value("black")
               ),
               size=alt.condition(
                   alt.datum.Player == player, alt.value(300), alt.value(60)
               ),
               tooltip=["Player", stat]
           )
           .properties(height=600, width=800)
    )
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
