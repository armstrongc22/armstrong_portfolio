"""
boi.ingest.worldbank
--------------------
Fetch demand-side indicators (population, GDP per capita) from the
World Bank open API and either:

1.  Write them straight to BigQuery    (USE_KAFKA = False)
2.  Publish each row to a Kafka topic  (USE_KAFKA = True)

Usage
-----
from boi.ingest import worldbank
worldbank.fetch(["NGA", "COL", "KHM"])
"""

from __future__ import annotations

import json
from typing import Iterable, List

import pandas as pd
import wbgapi as wb

import boi.config as cfg
from boi.storage_bq import write_df  # always safe to import; used only if needed

# ── Optional Kafka producer ────────────────────────────────────────────────
if getattr(cfg, "USE_KAFKA", False):
    from confluent_kafka import Producer

    _kafka_producer = Producer(
        {
            "bootstrap.servers": cfg.KAFKA_BOOTSTRAP,
            "sasl.username": cfg.KAFKA_API_KEY,
            "sasl.password": cfg.KAFKA_SECRET,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
        }
    )
else:
    _kafka_producer = None
# ───────────────────────────────────────────────────────────────────────────

# World Bank indicator codes → human labels
_INDICATORS = {
    "SP.POP.TOTL": "population",       # total population
    "NY.GDP.PCAP.CD": "gdp_pc",        # GDP per capita, current US$
}

def _pull_country_record(iso3: str) -> dict:
    """Return {iso3, population, gdp_pc} for the latest year (2023)."""
    df = wb.data.DataFrame(_INDICATORS, economy=iso3, time=2023)
    if df.empty:
        raise ValueError(f"No WB data for {iso3}")
    latest = df.iloc[0]
    return {
        "iso3": iso3,
        "population": latest["SP.POP.TOTL"],
        "gdp_pc": latest["NY.GDP.PCAP.CD"],
    }

def fetch(iso3_list: Iterable[str]) -> pd.DataFrame:
    """
    Fetch indicators for each ISO-3 code.

    Parameters
    ----------
    iso3_list : iterable of str
        e.g. ["NGA", "COL"]

    Returns
    -------
    pd.DataFrame
        Columns: iso3, population, gdp_pc
    """
    records: List[dict] = [_pull_country_record(iso) for iso in iso3_list]
    df = pd.DataFrame(records)

    if _kafka_producer is not None:
        # ── Streaming path ────────────────────────────────────────────────
        for rec in records:
            _kafka_producer.produce(
                cfg.TOPIC_DEMAND,
                key=rec["iso3"].encode(),
                value=json.dumps(rec).encode(),
            )
        _kafka_producer.flush()
    else:
        # ── Direct BigQuery upload ───────────────────────────────────────
        write_df(df, "demand", mode="replace")

    return df


# Convenience when executing module directly: quick sanity pull
if __name__ == "__main__":
    # Example test run
    print(fetch(["NGA", "COL"]).head())
