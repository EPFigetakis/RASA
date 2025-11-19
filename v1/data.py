# ra_stat_arb/data.py
from __future__ import annotations
import os
from typing import List, Optional
import pandas as pd

def get_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    source: str = "yahoo",
    csv_dir: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Return DataFrame[date x tickers] of prices (Adj Close via auto_adjust=True).

    - source='yahoo': requires yfinance; imported lazily.
    - source='csv'  : expects {csv_dir}/{TICKER}.csv with columns ['Date','Adj Close'] or ['Date','Close'].
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames = []
    for t in tickers:
        if source == "yahoo":
            try:
                import yfinance as yf  # lazy import
            except Exception as e:
                raise RuntimeError("yfinance needed for source='yahoo'") from e
            df = yf.download(t, start=start, end=end, auto_adjust=auto_adjust, progress=False)
            if df is None or df.empty:
                print(f"[WARN] No data for {t}")
                continue
            s = df["Close"].copy() if "Close" in df.columns else df["Adj Close"].copy()
            s.name = t
        else:
            if not csv_dir:
                raise ValueError("csv_dir must be provided when source='csv'")
            path = os.path.join(csv_dir, f"{t}.csv")
            if not os.path.exists(path):
                print(f"[WARN] Missing {path}")
                continue
            raw = pd.read_csv(path, parse_dates=["Date"])
            col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            s = raw.set_index("Date")[col].copy()
            s.name = t
        frames.append(s)
    if not frames:
        raise RuntimeError("No price series loaded.")
    prices = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all", axis=1)
    return prices

def get_market_series(prices_index: pd.DatetimeIndex, start: str = None) -> pd.Series:
    """
    Download a market proxy (SPY -> ^GSPC -> IVV), aligned to `prices_index`.
    """
    start = start or prices_index.min().strftime("%Y-%m-%d")
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance needed for market series") from e

    for t in ["SPY", "^GSPC", "IVV"]:
        df = yf.download(t, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty:
            continue
        s = df["Close"].copy() if "Close" in df.columns else df["Adj Close"].copy()
        s.name = "MKT"
        return s.reindex(prices_index).ffill().dropna()
    raise RuntimeError("Could not download market series.")
