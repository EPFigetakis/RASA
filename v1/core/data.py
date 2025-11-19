from __future__ import annotations
import pandas as pd
from pathlib import Path
import yfinance as yf
from .utils import log, ensure_dir

def load_prices(
    tickers: list[str],
    start: str = "2016-01-01",
    end: str | None = None,
    data_dir: Path = Path("data_cache"),
    cache_key: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download/cached daily adjusted close prices for tickers.
    Returns wide DataFrame indexed by date, columns=tickers.
    """
    ensure_dir(data_dir)
    if cache_key is None:
        cache_key = f"adjclose_{tickers[0]}_{len(tickers)}_{start}"
    cache_path = data_dir / f"{cache_key}.parquet"

    if cache_path.exists() and not force_refresh:
        log(f"Loading cached prices from {cache_path.name} …")
        return pd.read_parquet(cache_path)

    log(f"Downloading {len(tickers)} tickers from Yahoo Finance …")
    px = yf.download(
        tickers, start=start, end=end, interval="1d", auto_adjust=True, progress=False
    )["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.sort_index().dropna(how="all").ffill()

    # Keep columns with >98% non-NaN coverage
    keep = px.notna().mean().loc[lambda s: s > 0.98].index
    px = px[keep]

    px.to_parquet(cache_path)
    log(f"Saved cache to {cache_path}")
    return px
