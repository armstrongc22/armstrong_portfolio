# nuclear/uranium_gap.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 0) Paths
BASE = Path(__file__).parent       # nuclear/
DATA = BASE / "data"               # nuclear/data/

# 1) Load your datasets
df_prod = pd.read_csv(DATA / "uranium_production_cleaned.csv")
df_gen  = pd.read_csv(DATA / "nuclear_generation_by_country.csv")

# 2) Clean & prepare
# Identify the right generation column (strip spaces and match 'production' + '2022')
df_gen.columns = df_gen.columns.str.strip()
gen_col = next(c for c in df_gen.columns 
               if "electricity" in c.lower() and "production" in c.lower() and "2022" in c)
df_gen = df_gen.rename(columns={gen_col: "Production_TWh_2022"})

# Compute demand (200 tU/GWe-yr → 200/8.76 tU/TWh)
tU_per_TWh = 200 / 8.76
df_gen["Demand_tU_2022"] = df_gen["Production_TWh_2022"] * tU_per_TWh

# Prepare supply
df_prod.columns = df_prod.columns.str.strip()
df_prod["Supply_tU_2022"] = df_prod["2022"]

# Merge & compute gaps
df_gap = (
    df_gen[["Country", "Demand_tU_2022"]]
    .merge(df_prod[["Country", "Supply_tU_2022"]], on="Country", how="left")
    .fillna(0)
)
df_gap["Gap_tU_2022"]   = df_gap["Demand_tU_2022"]  - df_gap["Supply_tU_2022"]
df_gap["Gap_pct_2022"]  = df_gap["Gap_tU_2022"] / df_gap["Demand_tU_2022"] * 100

# Top 10 by absolute and percentage gap
top10_abs = df_gap.nlargest(10, "Gap_tU_2022").copy()
top10_pct = df_gap.nlargest(10, "Gap_pct_2022").copy()
top10_pct["Gap_pct_2022"] = top10_pct["Gap_pct_2022"].round(1)

# --- Streamlit UI ---
st.title("Uranium Supply–Demand Gap (2022)")

st.subheader("Top 10 Countries by Absolute Gap (tU)")
fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.bar(top10_abs["Country"], top10_abs["Gap_tU_2022"])
ax1.set_xticks(range(len(top10_abs)))
ax1.set_xticklabels(top10_abs["Country"], rotation=45, ha="right")
ax1.set_ylabel("Gap (tonnes U)")
fig1.tight_layout()
st.pyplot(fig1)

st.subheader("Top 10 Countries by % Gap")
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(top10_pct["Country"], top10_pct["Gap_pct_2022"])
ax2.set_xticks(range(len(top10_pct)))
ax2.set_xticklabels(top10_pct["Country"], rotation=45, ha="right")
ax2.set_ylabel("Gap (%)")
for i, v in enumerate(top10_pct["Gap_pct_2022"]):
    ax2.text(i, v + 1, f"{v}%", ha="center")
fig2.subplots_adjust(bottom=0.3)
st.pyplot(fig2)
