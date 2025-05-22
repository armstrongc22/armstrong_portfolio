"""
boi.ingest.fsq_popularity  –  Free-tier foot-traffic sampler
"""
from __future__ import annotations
import time, json, requests, pandas as pd, h3
import boi.config as cfg
from boi.storage_bq import write_df
import h3        # works for all recent h3-py versions

# compatibility alias  ──────────────────────────────────────────────
if not hasattr(h3, "geo_to_h3"):        # h3 4.x
    h3.geo_to_h3 = h3.latlng_to_cell    # type: ignore
# ───────────────────────────────────────────────────────────────────

HEADERS = {
    "Authorization": cfg.FSQ_API_KEY,
    "Accept": "application/json",
}

# Numeric category IDs (see https://location.foursquare.com/developer/docs/categories)
CATEGORY_IDS = {
    "grocery_store": 13065,
    "supermarket":   13077,
    "college":       19009,
    "bus_station":   17069,
    "train_station": 19046,
}

RADIUS  = 4000         # metres
LIMIT   = 50           # max per call
H3_RES  = 8            # ~460 m hexes

def _one_search(lat: float, lon: float, cat_id: int, cursor: str | None):
    params = {
        "ll":        f"{lat},{lon}",
        "radius":    RADIUS,
        "categories": cat_id,
        "limit":     LIMIT,
        "fields":    "geocodes,popularity",
    }
    if cursor:
        params["cursor"] = cursor
    r = requests.get(
        "https://api.foursquare.com/v3/places/search",
        headers=HEADERS,
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

def fetch(cities: dict) -> pd.DataFrame:
    rows = []
    for city, meta in cities.items():
        lat, lon = meta["lat"], meta["lon"]

        for cat_name, cat_id in CATEGORY_IDS.items():
            cursor = None
            while True:
                js = _one_search(lat, lon, cat_id, cursor)
                for place in js.get("results", []):
                    pop = place.get("popularity")
                    if pop is None:
                        continue
                    plat = place["geocodes"]["main"]["latitude"]
                    plon = place["geocodes"]["main"]["longitude"]
                    rows.append({
                        "city": city,
                        "hex":  h3.geo_to_h3(plat, plon, H3_RES),
                        "popularity": pop,
                    })
                cursor = js.get("context", {}).get("next_cursor")
                if not cursor:
                    break
                time.sleep(0.25)     # stay way below 950 calls/day
        print(f"✓ {city}: {len(rows)} POIs aggregated")

    df = (
        pd.DataFrame(rows)
        .groupby(["city", "hex"], as_index=False)
        .popularity.max()
    )
    write_df(df, "foot_traffic", mode="replace")
    return df
