import json, pandas as pd, time
from confluent_kafka import Consumer
import boi.config as cfg
from boi.storage_bq import write_df

def _consumer(topic):
    return Consumer({
        "bootstrap.servers": cfg.KAFKA_BOOTSTRAP,
        "group.id":          f"boi-{topic}",
        "auto.offset.reset": "earliest",
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms":   "PLAIN",
        "sasl.username":     cfg.KAFKA_API_KEY,
        "sasl.password":     cfg.KAFKA_SECRET,
    })

def sink(topic, table):
    c = _consumer(topic)
    c.subscribe([topic])
    buf = []
    while True:
        msg = c.poll(1.0)
        if msg is None:
            if buf:
                write_df(pd.DataFrame(buf), table, mode="append")
                buf.clear()
            time.sleep(5); continue
        buf.append(json.loads(msg.value()))

if __name__ == "__main__":
    import argparse, boi.storage_bq as bq
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    args = ap.parse_args()
    table = args.topic.split(".")[-1]   # "boi.demand" -> "demand"
    sink(args.topic, table)
