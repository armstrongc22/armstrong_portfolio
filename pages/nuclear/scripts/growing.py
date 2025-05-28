# nuclear/scripts/growing.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import unidecode

def main():
    st.title("🌱 Growing Nuclear Markets")

    # 0) Paths
    BASE   = Path(__file__).resolve().parent.parent
    DATA   = BASE / "data"
    FILE_OP = DATA / "wn_all_countries_reactors.csv"
    FILE_UC = DATA / "under_construction.csv"
    FILE_GEN= DATA / "nuclear_generation_by_country.csv"

    # 1) Load fleets
    df_op = pd.read_csv(FILE_OP)
    df_uc = pd.read_csv(FILE_UC)

    # 2) normalize helper
    def norm(s): return unidecode.unidecode(str(s)).lower().strip()

    # 3) Aggregate existing MW by country-key
    df_op["ck"] = df_op["Country"].map(norm)
    df_exist = (
        df_op.groupby("ck", as_index=False)["Capacity (MWe)"]
             .sum().rename(columns={"Capacity (MWe)": "Existing_MW"})
    )

    # 4) Under-construction MW
    df_uc = df_uc.rename(columns={"Total Net Electrical Capacity [MW]":"UC_MW"})
    df_uc["ck"] = df_uc["Country"].map(norm)

    # 5) fix a few mismatched keys
    M = {
        "korea, republic of": "south korea",
        "turkiye":            "turkey",
        "iran, islamic republic of":"iran",
        "united states of america":"usa",
    }
    df_exist["ck"].replace(M, inplace=True)
    df_uc["ck"].replace(M, inplace=True)

    # 6) merge & compute pipeline %
    df = (
        df_exist.merge(df_uc[["ck","UC_MW"]], on="ck", how="left")
                .fillna({"UC_MW":0})
    )
    df["PipelinePct"] = (df.UC_MW/df.Existing_MW*100).round(1)

    # 7) recover display name
    name_map = {**dict(zip(df_uc.ck, df_uc.Country)),
                **dict(zip(df_op.ck, df_op.Country))}
    df["Country"] = df["ck"].map(name_map)

    top10 = df.nlargest(10, "PipelinePct")

    # —— Panel A: Pipeline bar chart —— #
    st.subheader("Under-Construction as % of Current Fleet (Top 10)")
    st.dataframe(
        top10.set_index("Country")[["Existing_MW","UC_MW","PipelinePct"]]
             .rename(columns={
                 "Existing_MW":"Existing MW",
                 "UC_MW":"Under-Construction MW",
                 "PipelinePct":"% Pipeline"
             })
    )
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(top10.Country, top10.PipelinePct, color="C0")
    ax.set_ylabel("% Pipeline")
    ax.set_xticklabels(top10.Country, rotation=45, ha="right")
    for i,v in enumerate(top10.PipelinePct):
        ax.text(i, v+1, f"{v}%", ha="center")
    fig.tight_layout()
    st.pyplot(fig)

    # —— Panel B: Share growth & Geo-map —— #
    # 1) load share data
    df_gen = pd.read_csv(FILE_GEN)
    df_gen.columns = df_gen.columns.str.strip()
    df_gen["ck"] = df_gen["Country"].map(norm).replace(M)

    # find the two "nuclear share" columns
    shares = [c for c in df_gen.columns if "nuclear share" in c.lower()]
    if len(shares)<2:
        st.error("Couldn't find two 'nuclear share' columns in generation data.")
        return
    old_col, new_col = shares[0], shares[-1]

    # build a tiny df for top10
    heat = (
        df_gen.set_index("ck")[[old_col,new_col]]
              .reindex(top10.ck)
              .rename(columns={old_col:"Share_Old", new_col:"Share_New"})
              .reset_index()
    )
    heat["Country"] = top10.set_index("ck")["Country"].loc[heat.ck].values

    # 2) get world centroids
    try:
        world = gpd.read_file(
            "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        )
        # find a name col
        name_col = next(c for c in ("ADMIN","name","Country") if c in world.columns)
        world["ck"] = world[name_col].map(norm).replace(M)
        # filter to matching top10 
        sub = world[world.ck.isin(top10.ck)].copy()
        if sub.empty:
            raise ValueError("No matching countries in GeoJSON")
        # ensure valid geoms
        sub = sub[sub.geometry.notnull() & sub.geometry.is_valid]
        sub["centroid"] = sub.geometry.centroid

    except Exception as e:
        st.warning(f"Geo-map failed ({e}). Skipping.")
        sub = None

    if sub is not None:
        # merge centroids + share
        plot_df = pd.merge(heat, sub[["ck","centroid"]], on="ck", how="inner")
        
        # Check if we have valid data to plot
        if plot_df.empty:
            st.warning("No matching geographic data found for plotting.")
            return
        
        # Clean the data - remove rows with NaN values in critical columns
        plot_df = plot_df.dropna(subset=['Share_Old', 'Share_New'])
        
        # Check centroids are valid
        plot_df = plot_df[plot_df['centroid'].notnull()]
        
        if plot_df.empty:
            st.warning("No valid data points for mapping after cleaning.")
            return

        # Create neon sonar-style map
        fig2, ax2 = plt.subplots(1,1, figsize=(14,8))
        ax2.set_facecolor("black")
        ax2.axis("off")
        ax2.set_title("Nuclear Share Growth: Sonar View (2018 → 2023)", 
                     color="white", fontsize=16, pad=20)

        # Plot world countries as dark outlines
        try:
            world.boundary.plot(ax=ax2, color="#1a1a1a", linewidth=0.3)
        except:
            pass  # Skip if world boundary plotting fails
        
        # Track if we actually plot anything
        plotted_points = 0
        
        # Create custom neon green colormap
        from matplotlib.colors import LinearSegmentedColormap
        neon_colors = ['#001100', '#003300', '#00ff00', '#88ff88', '#ffffff']
        neon_cmap = LinearSegmentedColormap.from_list("neon_green", neon_colors)
        
        for _,r in plot_df.iterrows():
            try:
                # Extract coordinates safely
                if hasattr(r.centroid, 'x') and hasattr(r.centroid, 'y'):
                    x, y = r.centroid.x, r.centroid.y
                else:
                    continue
                
                # Check for valid coordinates and shares
                if (pd.isna(x) or pd.isna(y) or np.isinf(x) or np.isinf(y) or 
                    pd.isna(r.Share_Old) or pd.isna(r.Share_New)):
                    continue
                
                # Calculate actual growth
                old_share = max(r.Share_Old, 0.1)  # avoid division by zero
                new_share = max(r.Share_New, 0.1)
                growth_ratio = new_share / old_share
                
                # Base size on nuclear share magnitude
                base_size = max(new_share * 50, 100)  # minimum visible size
                inner_size = base_size * 0.7  # 2018 (inner circle)
                outer_size = base_size * growth_ratio  # 2023 (shows growth)
                
                # Color intensity based on share level
                color_intensity = min(new_share / 50, 1.0)  # normalize to 0-1
                
                # Plot outer circle (2023) - larger if growth occurred
                ax2.scatter(x, y, s=outer_size, 
                           c=neon_cmap(color_intensity), 
                           alpha=0.4, edgecolor='#00ff00', 
                           linewidth=1.5, zorder=4)
                
                # Plot inner circle (2018) - baseline
                ax2.scatter(x, y, s=inner_size, 
                           c='#003300', alpha=0.8, 
                           edgecolor='#00ff00', linewidth=0.8, zorder=5)
                
                # Add country label
                country_name = r.get('Country', 'Unknown')
                ax2.annotate(f"{country_name}\n{old_share:.1f}% → {new_share:.1f}%", 
                           (x, y), xytext=(5, 5), textcoords='offset points',
                           color='#00ff88', fontsize=8, alpha=0.9,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='black', alpha=0.7))
                
                plotted_points += 1
                
            except Exception as e:
                st.warning(f"Error plotting point for {r.get('Country', 'unknown')}: {e}")
                continue
        
        if plotted_points == 0:
            st.warning("No valid points could be plotted on the map.")
            return

        # Create neon legend
        from matplotlib.lines import Line2D
        leg_elements = [
            Line2D([0],[0], marker='o', color='w', label="2018 Nuclear Share (Inner)",
                   markerfacecolor='#003300', markeredgecolor='#00ff00', 
                   markersize=8, alpha=0.8),
            Line2D([0],[0], marker='o', color='w', label="2023 Nuclear Share (Outer)", 
                   markerfacecolor='#00ff00', markeredgecolor='#00ff00',
                   markersize=12, alpha=0.6),
            Line2D([0],[0], marker='s', color='w', label="Larger outer = Growth",
                   markerfacecolor='none', markeredgecolor='#88ff88', 
                   markersize=6, alpha=0.8)
        ]
        legend = ax2.legend(handles=leg_elements, loc="upper left", 
                           facecolor='black', edgecolor='#00ff00', 
                           labelcolor='white', framealpha=0.8)
        legend.get_frame().set_linewidth(1)

        # Set map bounds to show all points with some padding
        if plotted_points > 0:
            coords = [(r.centroid.x, r.centroid.y) for _, r in plot_df.iterrows() 
                     if hasattr(r.centroid, 'x') and not pd.isna(r.centroid.x)]
            if coords:
                xs, ys = zip(*coords)
                margin = 20
                ax2.set_xlim(min(xs)-margin, max(xs)+margin)
                ax2.set_ylim(min(ys)-margin, max(ys)+margin)

        st.pyplot(fig2)

if __name__=="__main__":
    main()