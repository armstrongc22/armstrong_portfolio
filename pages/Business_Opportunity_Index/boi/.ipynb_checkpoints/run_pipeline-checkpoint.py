import argparse, boi.config as cfg
from boi.ingest import worldbank, osm_pois
from boi.compute import scorer
import boi.ingest.fsq_popularity as fsq

def main(mode):
    cfg.USE_KAFKA = (mode == "stream")

    iso = {c["iso3"] for c in cfg.CITIES.values()}
    worldbank.fetch(iso)
    osm_pois.fetch(cfg.CITIES)
    fsq.fetch(cfg.CITIES)  
    scorer.compute()
    print("✅ pipeline finished")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local","direct","stream"],
                    default="direct",
                    help="local CSV/SQLite, direct BigQuery, or Kafka stream")
    args = ap.parse_args()
    main(args.mode)
