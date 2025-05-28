# nuclear/scripts/pipeline.py

import streamlit as st
import pandas as pd
from pathlib import Path
def main():
    # 0) Paths
    BASE       = Path(__file__).resolve().parent.parent
    UC_CSV     = BASE / "data" / "under_construction.csv"
    REACT_SUM  = BASE / "data" / "wn_all_countries_reactors.csv"
    
    st.title("Project-Finance Pipeline & Risk Assessment")
    
    # 1) Load under-construction capacities
    df_uc = pd.read_csv(UC_CSV)
    # Rename for clarity
    df_uc = df_uc.rename(
        columns={
            "Total Net Electrical Capacity [MW]": "Capacity_MW",
            "Number of Reactors":                "Units"
        }
    )
    # Build a “project” per country
    df_pipeline = df_uc[["Country","Capacity_MW"]].copy()
    df_pipeline["Project"] = df_pipeline["Country"] + " UC Package"
    df_pipeline["Status"]  = "Under Construction"
    df_pipeline["Sponsor"] = pd.NA
    
    # 2) Sovereign ratings (fill in real data)
    sovereign_ratings = pd.DataFrame([
        {"Country":"India",    "SovereignRating":"BBB-", "RatingScore":3},
        {"Country":"China",    "SovereignRating":"A+",  "RatingScore":2},
        {"Country":"Russia",   "SovereignRating":"BB-", "RatingScore":5},
        {"Country":"USA",      "SovereignRating":"AA+", "RatingScore":1},
        {"Country":"UK",       "SovereignRating":"AA",  "RatingScore":1},
        # … add your host-state ratings here
    ])
    df_pipeline = df_pipeline.merge(sovereign_ratings, on="Country", how="left")
    
    # 3) ECA partner list (fill in your known mappings)
    eca_partners = pd.DataFrame([
        {"Project":"India UC Package", "ECA":"US EXIM",     "WrapCoverage":0.80},
        {"Project":"China UC Package", "ECA":"Sinosure",     "WrapCoverage":0.75},
        {"Project":"Russia UC Package","ECA":"Euler Hermes","WrapCoverage":0.70},
        # …
    ])
    df_pipeline = df_pipeline.merge(eca_partners, on="Project", how="left")
    
    # 4) Financing assumptions
    EQUITY_RATIO = 0.30
    DEBT_RATIO   = 0.70
    BASE_PREM_BPS = {"AAA":100, "AA":150, "A":200, "BBB":250, "BB":500, "B":750}
    
    def pick_premium(rat):
        base = rat.split("+")[0].split("-")[0] if pd.notna(rat) else "BB"
        return BASE_PREM_BPS.get(base, 500)
    
    df_pipeline["RiskPremium_bps"] = df_pipeline["SovereignRating"].apply(pick_premium)
    
    # 5) Hurdle IRR
    df_pipeline["HurdleRate_%"] = (
        3.0 + df_pipeline["RiskPremium_bps"] / 100
    ) * (1 - df_pipeline["WrapCoverage"].fillna(0))
    
    # 6) Capital stack ($5M/MW)
    df_pipeline["ProjectCost_M$"] = df_pipeline["Capacity_MW"] * 5
    df_pipeline["Equity_M$"]      = df_pipeline["ProjectCost_M$"] * EQUITY_RATIO
    df_pipeline["Debt_M$"]        = df_pipeline["ProjectCost_M$"] * DEBT_RATIO
    
    # 7) Deal book
    st.subheader("Under-Construction Reactor Deal Book")
    cols = [
        "Project","Country","Capacity_MW","Status","Sponsor",
        "ECA","WrapCoverage","SovereignRating","RiskPremium_bps",
        "ProjectCost_M$","Equity_M$","Debt_M$","HurdleRate_%"
    ]
    st.dataframe(
        df_pipeline[cols]
          .sort_values("HurdleRate_%", ascending=False)
          .reset_index(drop=True)
    )
    
    st.markdown("""
    ### Investor Takeaways
    - **Equity tickets**: ~30% of \$5 M/MW → \$1.5 M/MW (e.g. \$1.5 B for 1 GW).  
    - **Hurdle IRR**: ~3% + sovereign premium, net of ECA wrap.  
    - **ECA support**: reduces your cost of capital by up to 80%.  
    - **Next steps**: swap in exact sovereign ratings and ECA mappings, then build detailed cash-flow IRR models per project.
    """)

if __name__=="__main__":
    main()