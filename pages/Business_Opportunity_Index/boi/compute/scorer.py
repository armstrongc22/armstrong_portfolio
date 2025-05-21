import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
import boi.config as cfg
from boi.storage_csv import write_df, read_sql
   # client already carries creds

# ----------------------------------------------------------------------
# 1.  CITY-LEVEL SCORE  (same as before)
# ----------------------------------------------------------------------
def load_joined_data():
    s = read_sql("supply")
    d = read_sql("demand")
    df = pd.merge(s, d, on="iso3", how="inner")
    return df


def compute():
    df = load_joined_data()

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
    f = read_sql("foot_traffic")
    s = read_sql("scores")
    df = f.merge(s[s["category"] == "laundromat"], on="city")
    df["opp_norm"] = df["opportunity_score"] / 100
    df["local_opportunity"] = (df["popularity"] * df["opp_norm"]).round(3)
    write_df(df[["city", "hex", "popularity", "opp_norm", "local_opportunity"]],
             "hex_opportunity", mode="replace")



# ----------------------------------------------------------------------
# 3.  The module’s __all__ is still just “compute”
# ----------------------------------------------------------------------
__all__ = ["compute"]
