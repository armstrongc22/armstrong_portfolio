# boi/ingest/osm_pois.py
import json, pandas as pd, osmnx as ox
from osmnx import _errors as oxerr              # <-- import exceptions
import boi.config as cfg
from boi.storage_bq import write_df
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests.exceptions as req_exc
# optional Kafka (unchanged) ..............................
if getattr(cfg, "USE_KAFKA", False):
    from confluent_kafka import Producer
    kafka = Producer({
        "bootstrap.servers": cfg.KAFKA_BOOTSTRAP,
        "sasl.username":     cfg.KAFKA_API_KEY,
        "sasl.password":     cfg.KAFKA_SECRET,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms":   "PLAIN",
    })
else:
    kafka = None
# .........................................................


# ...
@retry(
    retry=retry_if_exception_type((oxerr.InsufficientResponseError, req_exc.ConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=False
)
def _count(lat, lon, key, value, radius_m=10_000):
    try:
        gdf = ox.features_from_point((lat, lon), {key: value}, dist=radius_m)
        return len(gdf)
    except oxerr.InsufficientResponseError:
        return 0          # no POIs



def fetch(cities: dict) -> pd.DataFrame:
    rows = []
    for city, meta in cities.items():
        for cat, tag in cfg.POI_CATEGORIES.items():
            cnt = _count(meta["lat"], meta["lon"],
                         tag["key"], tag["value"])
            rows.append({
                "city":      city,
                "iso3":      meta["iso3"],
                "category":  cat,
                "count":     cnt,
            })

    df = pd.DataFrame(rows)

    if kafka:
        for row in rows:
            kafka.produce(
                cfg.TOPIC_SUPPLY,
                key=row["city"].encode(),
                value=json.dumps(row).encode(),
            )
        kafka.flush()
    else:
        write_df(df, "supply", mode="replace")

    return df
