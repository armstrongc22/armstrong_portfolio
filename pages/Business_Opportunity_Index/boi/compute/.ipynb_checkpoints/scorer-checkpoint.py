import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
import boi.config as cfg
from boi.storage_bq import read_sql, write_df, client   # client already carries creds

# ----------------------------------------------------------------------
# 1.  CITY-LEVEL SCORE  (same as before)
# ----------------------------------------------------------------------
SQL = f"""
SELECT s.city, s.category, s.count,
       d.population
FROM   `{cfg.PROJECT}.{cfg.DATASET}.supply`  AS s
JOIN   `{cfg.PROJECT}.{cfg.DATASET}.demand`  AS d
USING  (iso3)
"""

def compute():
    df = read_sql(SQL)

    df["per_10k"] = df["count"] / (df["population"] / 1e4)
    df["gap_pct"] = 1 - df.apply(
        lambda r: r["per_10k"] / cfg.BENCHMARK[r["category"]],
        axis=1,
    ).clip(lower=0)

    scaler = MinMaxScaler((0, 100))
    df["opportunity_score"] = scaler.fit_transform(df[["gap_pct"]]).round(1)

    write_df(
        df[["city", "category", "opportunity_score"]],
        "scores",
        mode="replace",
    )

    # ── NEW: build hex-level composite right after city scores ─────────
    build_hex_opportunity()

    return df


# ----------------------------------------------------------------------
# 2.  HEX-LEVEL “local_opportunity” TABLE
# ----------------------------------------------------------------------
def build_hex_opportunity():
    """
    Join hex foot-traffic with laundromat (or any category) scores and
    compute   local_opportunity = popularity × (opportunity_score / 100)
    """

    sql = f"""
    CREATE OR REPLACE TABLE `{cfg.PROJECT}.{cfg.DATASET}.hex_opportunity` AS
    SELECT
      f.city,
      f.hex,
      f.popularity,
      s.opportunity_score / 100      AS opp_norm,
      ROUND(f.popularity * (s.opportunity_score / 100), 3)
                                     AS local_opportunity
    FROM `{cfg.PROJECT}.{cfg.DATASET}.foot_traffic` AS f
    JOIN `{cfg.PROJECT}.{cfg.DATASET}.scores`       AS s
      ON s.city = f.city
    WHERE s.category = 'laundromat';
    """
    client.query(sql).result()
    print("✅ hex_opportunity table refreshed")


# ----------------------------------------------------------------------
# 3.  The module’s __all__ is still just “compute”
# ----------------------------------------------------------------------
__all__ = ["compute"]
