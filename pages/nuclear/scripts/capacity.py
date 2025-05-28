# nuclear/scripts/supply_chain.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    BASE = Path(__file__).resolve().parent.parent
    DATA = BASE / "data"
    PROD_CSV = DATA / "uranium_production_cleaned.csv"
    GEN_CSV = DATA / "nuclear_generation_by_country.csv"
    
    st.title("Supply-Chain Concentration & Risk Dashboard (Uranium Feed)")
    
    # 1) Load and process uranium production data (supply) by country, 2013–2022
    df_prod = pd.read_csv(PROD_CSV)
    years = [c for c in df_prod.columns if c.isdigit()]
    df_prod_long = (
        df_prod
        .melt(
            id_vars="Country",
            value_vars=years,
            var_name="Year",
            value_name="Supply_tU"
        )
        .assign(Year=lambda d: d["Year"].astype(int))
    )
    df_supply = df_prod_long.groupby("Year", as_index=False)["Supply_tU"].sum()
    
    # 2) Process reactor fleet via generation data for 2022
    df_gen = pd.read_csv(GEN_CSV)
    df_gen.columns = df_gen.columns.str.strip()
    
    # Find the 2022 generation column
    gen2022_col = next(c for c in df_gen.columns if "production" in c.lower() and "2022" in c)
    df_gen = df_gen.rename(columns={gen2022_col: "Gen_TWh_2022"})
    
    # Convert TWh to GW-year (fleet size): GW = TWh / 8.76
    df_gen["Fleet_GW"] = df_gen["Gen_TWh_2022"] / 8.76
    
    # Compute country-level demand (tU): 200 tU per GWe-year
    df_gen["Demand_tU_2022"] = df_gen["Fleet_GW"] * 200
    
    # 3) Calculate global demand by year (held flat at 2022 level)
    total_demand_2022 = df_gen["Demand_tU_2022"].sum()
    years_sd = df_supply["Year"].unique()
    df_demand = pd.DataFrame({
        "Year": years_sd,
        "Demand_tU": total_demand_2022
    })
    
    # 4) Merge supply & demand data
    df_sd = pd.merge(df_supply, df_demand, on="Year")
    df_sd["Gap_tU"] = df_sd["Supply_tU"] - df_sd["Demand_tU"]
    df_sd["GapPct"] = df_sd["Gap_tU"] / df_sd["Demand_tU"]
    
    # 5) Scenario analysis: +5 ktU/yr new mine from 2025
    df_sd["Supply_withNew"] = np.where(
        df_sd["Year"] >= 2025,
        df_sd["Supply_tU"] + 5000,
        df_sd["Supply_tU"]
    )
    df_sd["Gap_withNew"] = df_sd["Supply_withNew"] - df_sd["Demand_tU"]
    
    # 6) Create visualizations
    
    # Chart 1: Global uranium production vs reactor demand
    st.subheader("Global Uranium Production vs. Reactor Demand")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_sd["Year"], df_sd["Supply_tU"], label="Production (tU)", marker="o")
    ax1.plot(df_sd["Year"], df_sd["Demand_tU"], label="Demand (tU)", marker="s")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Tonnes Uranium")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Chart 2: Supply–demand gap percentage
    st.subheader("Supply–Demand Gap %")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ['red' if x < 0 else 'green' for x in df_sd["GapPct"]]
    ax2.bar(df_sd["Year"], df_sd["GapPct"] * 100, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Gap %")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Chart 3: Impact of new mine scenario
    st.subheader("Impact of Adding 5 ktU/yr New Mine (from 2025)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(df_sd["Year"], df_sd["Gap_withNew"], marker="o", linewidth=2)
    ax3.axhline(0, color="red", linestyle="--", alpha=0.8)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Tonnes U Gap (with new mine)")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    
    # 7) White-space markets analysis
    prod_countries = set(df_prod["Country"])
    
    df_ws = df_gen[
        (df_gen["Fleet_GW"] > 0.5) &
        (~df_gen["Country"].isin(prod_countries))
    ].copy()
    
    st.subheader("Growing Reactor Markets without Domestic Uranium Production")
    if not df_ws.empty:
        st.dataframe(
            df_ws[["Country", "Fleet_GW"]]
            .rename(columns={"Fleet_GW": "Fleet (GW)"})
            .sort_values("Fleet (GW)", ascending=False)
            .round(2)
            .set_index("Country")
        )
    else:
        st.write("No white-space markets found with the current criteria.")
    
    # 8) Investment ROI analysis
    st.subheader("New 5 ktU Mine — Revenue Estimate")
    price_spot = 100_000  # $ per tU
    price_contract = 80_000  # $ per tU
    annual_production = 5000  # tU
    
    rev_spot = annual_production * price_spot
    rev_contract = annual_production * price_contract
    
    roi_data = {
        "Scenario": ["Spot Price", "Contract Price"],
        "Price ($/tU)": [f"${price_spot:,}", f"${price_contract:,}"],
        "Annual Revenue ($M)": [f"${rev_spot/1e6:.0f}", f"${rev_contract/1e6:.0f}"]
    }
    
    roi_df = pd.DataFrame(roi_data)
    st.table(roi_df.set_index("Scenario"))
    
    # Key insights summary
    supply_2022 = df_sd.loc[df_sd["Year"] == 2022, "Supply_tU"].values[0]
    gap_2022 = df_sd.loc[df_sd["Year"] == 2022, "Gap_tU"].values[0]
    supply_coverage = 100 * supply_2022 / total_demand_2022
    
    st.markdown(f"""
    > **Key Insights**
    > 
    > - By 2022, production meets only **{supply_coverage:.1f}%** of demand → a gap of **{gap_2022:,.0f} tU**.
    > - A new 5 ktU mine from 2025 closes that gap and yields **$400M–$500M/yr**.
    > - Countries with >0.5 GW fleet but no domestic production are prime "white-space" targets.
    """)


if __name__ == "__main__":
    main()