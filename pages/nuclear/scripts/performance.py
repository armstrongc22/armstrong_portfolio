# nuclear/scripts/performance.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


def load_latest_factor(path: Path, metric_name: str, new_col: str) -> pd.DataFrame:
    """
    Reads a CSV with two header rows:
      • Row 0: may be blank or contain years
      • Row 1: metric names repeated
    We:
      1) skiprows=0, header=[0,1] to load both
      2) take cols[0] as Country
      3) find all cols where row-1 matches metric_name
      4) extract the one with the largest year in row-0
      5) return ['Country', new_col]
    """
    # 1) Read with two header rows
    df = pd.read_csv(path, header=[0, 1])
    cols = df.columns.tolist()  # list of tuples

    # 2) Country is the first column
    country_col = cols[0]

    # 3) Metric columns: those whose second-level header contains the keyword
    metric_keyword = metric_name.lower()
    metric_cols = [
        c for c in cols
        if metric_keyword in str(c[1]).lower().replace("\n", "").replace(" ", "")
    ]
    if not metric_cols:
        found = [c[1] for c in cols]
        raise ValueError(f"No metric '{metric_name}' in {path.name}: found {found}")

    # 4) Pick the one with the largest 4-digit year in the first-level
    def year_of(col):
        m = re.search(r"(\d{4})", str(col[0]))
        return int(m.group(1)) if m else -1

    latest_col = max(metric_cols, key=year_of)

    # 5) Build tidy DF
    df2 = df[[country_col, latest_col]].copy()
    df2.columns = ["Country", new_col]

    # 6) Normalize country text
    df2["Country"] = df2["Country"].astype(str).str.strip().str.title()

    # 7) Coerce factor to numeric
    df2[new_col] = pd.to_numeric(df2[new_col], errors="coerce")

    return df2


def main():
    # Paths
    BASE = Path(__file__).resolve().parent.parent  # …/nuclear
    DATA = BASE / "data"
    
    # 1) Load & aggregate reactor fleet capacity
    df_op = pd.read_csv(DATA / "wn_all_countries_reactors.csv")
    df_op_agg = (
        df_op.groupby("Country", as_index=False)["Capacity (MWe)"]
             .sum()
             .rename(columns={"Capacity (MWe)": "OperationalCapacity_MW"})
    )
    df_op_agg["Capacity_GW"] = df_op_agg["OperationalCapacity_MW"] / 1000
    
    # 2) Load performance factors using the helper function
    df_e = load_latest_factor(
        DATA / "energy_availability_factor.csv",
        metric_name="eaf",           # <- just the acronym
        new_col="EnergyAvail_pct"
    )
    
    df_u = load_latest_factor(
        DATA / "unit_capability_factor.csv",
        metric_name="ucf",           # matches "UCF\n[%]"
        new_col="UnitCap_pct"
    )
    
    df_l = load_latest_factor(
        DATA / "unplanned_capability_loss_factor.csv",
        metric_name="ucl",           # matches "UCL\n[%]"
        new_col="Loss_pct"
    )
    
    # 3) Merge into one DataFrame
    df = df_op_agg[["Country", "Capacity_GW"]].copy()
    for df_m, col in [(df_e, "EnergyAvail_pct"),
                      (df_u, "UnitCap_pct"),
                      (df_l, "Loss_pct")]:
        df = df.merge(df_m, on="Country", how="left")
    
    # 4) Compute top-quartile benchmarks
    th_e = df["EnergyAvail_pct"].quantile(0.75)
    th_u = df["UnitCap_pct"].quantile(0.75)
    th_l = df["Loss_pct"].quantile(0.25)  # lower is better
    
    # 5) Calculate performance shortfalls
    df["ΔEnergy"] = (th_e - df["EnergyAvail_pct"]).clip(lower=0)
    df["ΔUnit"] = (th_u - df["UnitCap_pct"]).clip(lower=0)
    df["ΔLoss"] = (df["Loss_pct"] - th_l).clip(lower=0)
    
    # 6) Calculate service opportunity value ($5M × %-pt × GW)
    df["Energy_$M"] = df["ΔEnergy"] * df["Capacity_GW"] * 5
    df["Unit_$M"] = df["ΔUnit"] * df["Capacity_GW"] * 5
    df["Loss_$M"] = df["ΔLoss"] * df["Capacity_GW"] * 5
    df["Total_$M"] = df[["Energy_$M", "Unit_$M", "Loss_$M"]].sum(axis=1)
    
    # 7) Get top 10 countries by opportunity
    top10 = df.nlargest(10, "Total_$M").reset_index(drop=True)
    
    # 8) Streamlit UI
    st.title("Performance Benchmark & Service-Opportunity Map")
    
    st.markdown("""
    This dashboard shows the **Top 10 countries** where closing the gap to top-quartile 
    performance (Energy Availability, Unit Capability, Unplanned Loss)
    unlocks the largest service opportunities, assuming **$5M per %-point per GW**.
    """)
    
    # Display benchmark thresholds
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Energy Availability Benchmark", f"{th_e:.1f}%")
    with col2:
        st.metric("Unit Capability Benchmark", f"{th_u:.1f}%")
    with col3:
        st.metric("Unplanned Loss Benchmark", f"{th_l:.1f}%")
    
    st.subheader("Top 10 Service Opportunities by Country")
    
    # Format the dataframe for display
    display_df = top10[[
        "Country", "Capacity_GW",
        "EnergyAvail_pct", "ΔEnergy", "Energy_$M",
        "UnitCap_pct", "ΔUnit", "Unit_$M",
        "Loss_pct", "ΔLoss", "Loss_$M",
        "Total_$M"
    ]].rename(columns={
        "Capacity_GW": "Capacity (GW)",
        "EnergyAvail_pct": "Energy Avail (%)",
        "ΔEnergy": "ΔEnergy (%)",
        "Energy_$M": "Energy $M",
        "UnitCap_pct": "Unit Cap (%)",
        "ΔUnit": "ΔUnit (%)",
        "Unit_$M": "Unit $M",
        "Loss_pct": "Unplanned Loss (%)",
        "ΔLoss": "ΔLoss (%)",
        "Loss_$M": "Loss $M",
        "Total_$M": "Total $M"
    }).round(1).set_index("Country")
    
    st.dataframe(display_df, use_container_width=True)
    
    # 9) Create visualization
    st.subheader("Service Opportunity Breakdown")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart - Total opportunities
    ax1.bar(range(len(top10)), top10["Total_$M"], color="steelblue", alpha=0.7)
    ax1.set_xticks(range(len(top10)))
    ax1.set_xticklabels(top10["Country"], rotation=45, ha="right")
    ax1.set_ylabel("Total Opportunity ($M)")
    ax1.set_title("Total Service Opportunities by Country")
    ax1.grid(True, alpha=0.3)
    
    # Stacked bar chart - Breakdown by category
    width = 0.8
    countries = range(len(top10))
    
    ax2.bar(countries, top10["Energy_$M"], width, label="Energy Availability", alpha=0.8)
    ax2.bar(countries, top10["Unit_$M"], width, bottom=top10["Energy_$M"], 
            label="Unit Capability", alpha=0.8)
    ax2.bar(countries, top10["Loss_$M"], width, 
            bottom=top10["Energy_$M"] + top10["Unit_$M"], 
            label="Loss Reduction", alpha=0.8)
    
    ax2.set_xticks(countries)
    ax2.set_xticklabels(top10["Country"], rotation=45, ha="right")
    ax2.set_ylabel("Opportunity Value ($M)")
    ax2.set_title("Service Opportunity Breakdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Market Size", f"${df['Total_$M'].sum():.0f}M")
    with col2:
        st.metric("Top 10 Market Size", f"${top10['Total_$M'].sum():.0f}M")
    with col3:
        st.metric("Average per Country (Top 10)", f"${top10['Total_$M'].mean():.0f}M")
    with col4:
        st.metric("Largest Single Opportunity", f"${top10['Total_$M'].max():.0f}M")


if __name__ == "__main__":
    main()