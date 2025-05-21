import pandas as pd
from boi.config import LOCAL_DATA_DIR

def table_path(name: str) -> str:
    return LOCAL_DATA_DIR / f"{name}.csv"

def read_sql(name: str) -> pd.DataFrame:
    path = table_path(name)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def write_df(df: pd.DataFrame, name: str, mode: str = "append"):
    path = table_path(name)
    if mode == "replace" or not path.exists():
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False)
