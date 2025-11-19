import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.pct_change()
    return out.replace([np.inf, -np.inf], np.nan)
