"""
core.prices

Download & cache daily adjusted close prices for a universe of tickers.

Key ideas:
- Use yfinance to download data for MANY tickers at once.
- Store the result as a single parquet file (wide format):
      index = date
      columns = tickers
- Use a simple cache_key mechanism so repeated calls are fast.
- Optionally load tickers/start/end from a JSON config file.

Typical usages from project root:

    # 1) Directly specify tickers
    from core import prices
    px = prices.load_prices(
        tickers=["SPY", "QQQ", "IWM"],
        start="2016-01-01",
    )

    # 2) Use config/data_config.json
    px = prices.load_prices_from_config()

Parquet files are stored under:
    cache/prices/adjclose_*.parquet
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Import paths helper (repo-style)
try:
    import core.paths as paths
except ModuleNotFoundError:
    # Fallback if run from inside core/ directly
    import paths  # type: ignore


# ---------------------------------------------------------------------------
# Small local utilities (log + ensure_dir)
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Simple logging helper."""
    print(f"[prices] {msg}")


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

def build_cache_key(
    tickers: List[str],
    start: str,
    end: Optional[str],
    interval: str = "1d",
) -> str:
    """
    Deterministic cache key covering the FULL universe and download settings.

    Two different universes (or date ranges/intervals) never collide, and
    the same universe always hashes to the same key regardless of ticker
    order. Unlike a key built from just the first ticker and ticker count,
    this can't silently reuse another universe's cache.
    """
    fingerprint = json.dumps(
        {
            "tickers": sorted(t.upper() for t in tickers),
            "start": start,
            "end": end,
            "interval": interval,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"adjclose_{digest}"


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _write_cache_metadata(
    cache_path: Path,
    tickers: List[str],
    start: str,
    end: Optional[str],
    interval: str,
    px: pd.DataFrame,
) -> None:
    """Write a sidecar JSON file recording what's actually in the cache."""
    meta = {
        "tickers": sorted(t.upper() for t in tickers),
        "start": start,
        "end": end,
        "interval": interval,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers_kept": int(px.shape[1]),
        "n_rows": int(px.shape[0]),
        "last_bar_date": px.index.max().date().isoformat() if len(px) else None,
    }
    _metadata_path(cache_path).write_text(json.dumps(meta, indent=2))


def read_cache_metadata(cache_path: Path) -> Optional[dict]:
    """Read the sidecar metadata for a cached parquet file, if present."""
    meta_path = _metadata_path(cache_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Core price loader with parquet caching
# ---------------------------------------------------------------------------

def load_prices(
    tickers: List[str],
    start: str = "2016-01-01",
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download/cached daily adjusted close prices for `tickers`.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame:
            - index = date (DatetimeIndex)
            - columns = tickers
            - values = adjusted close prices (via auto_adjust=True)

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g. ["SPY", "QQQ", "IWM"]).
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD) or None for "up to today".
    data_dir : Path or None
        Directory where parquet cache is stored.
        If None, defaults to: <PROJECT_ROOT>/cache/prices
    cache_key : str or None
        Key for naming the parquet file. If None, a deterministic hash of
        the full sorted ticker universe, start, end, and interval is used
        (see `build_cache_key`), so different universes/settings never
        collide on the same cache file.
    force_refresh : bool
        If True, ignore any existing cache and re-download.

    Notes
    -----
    - Uses yfinance.download with `auto_adjust=True` and `interval='1d'`.
    - Keeps only tickers with >98% non-NaN coverage.
    """
    # Decide where to store caches
    if data_dir is None:
        data_dir = paths.cache_dir() / "prices"
    ensure_dir(data_dir)

    tickers = [t.upper() for t in tickers]

    # Build a default cache key if none is given. This hashes the FULL
    # universe + settings so different universes/date ranges/intervals
    # can never collide on the same cache file (see build_cache_key).
    if cache_key is None:
        cache_key = build_cache_key(tickers, start, end, interval="1d")

    cache_path = data_dir / f"{cache_key}.parquet"

    # Load from cache if available
    if cache_path.exists() and not force_refresh:
        log(f"Loading cached prices from {cache_path.name} …")
        meta = read_cache_metadata(cache_path)
        if meta and meta.get("last_bar_date"):
            log(f"Cache last bar: {meta['last_bar_date']} (use force_refresh=True to update)")
        px = pd.read_parquet(cache_path)
        # Ensure datetime index
        if not isinstance(px.index, pd.DatetimeIndex):
            px.index = pd.to_datetime(px.index)
        return px

    # Otherwise, download from Yahoo
    log(f"Downloading {len(tickers)} tickers from Yahoo Finance …")

    # yfinance returns:
    #   - MultiIndex columns when multiple tickers: (field, ticker)
    #   - Single-index columns DataFrame when a single ticker
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # We only care about the 'Close' field (which is adjusted due to auto_adjust=True)
    try:
        px = raw["Close"]
    except Exception:
        # If something weird happens, show raw columns
        log(f"Unexpected columns from yfinance: {raw.columns}")
        raise

    # If a single ticker is requested, px may be a Series
    if isinstance(px, pd.Series):
        px = px.to_frame()

    # Sort by date index, drop all-NaN rows, forward-fill gaps
    px = (
        px.sort_index()
          .dropna(how="all")
          .ffill()
    )

    # Keep only columns with >98% non-NaN coverage
    coverage = px.notna().mean()
    keep_cols = coverage.loc[lambda s: s > 0.98].index
    px = px[keep_cols]

    # Ensure DatetimeIndex
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)

    # Save to parquet, plus sidecar metadata recording universe/settings/freshness
    px.to_parquet(cache_path)
    _write_cache_metadata(cache_path, tickers, start, end, "1d", px)
    log(f"Saved cache to {cache_path}")

    return px


# ---------------------------------------------------------------------------
# Config-based helper
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    """
    Default path for the data config JSON:

        <PROJECT_ROOT>/config/data_config.json

    Expected keys (example):
        {
          "data_source": "yahoo",
          "tickers": ["SPY", "QQQ", "IWM"],
          "start_date": "2016-01-01",
          "end_date": null,
          "interval": "1d"
        }
    """
    return paths.PROJECT_ROOT / "config" / "data_config.json"


def load_prices_from_config(
    config_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: read tickers/dates from a JSON config,
    then call `load_prices`.

    Parameters
    ----------
    config_path : Path or None
        JSON config file path. If None, uses the default config path.
    data_dir : Path or None
        Directory for parquet cache. If None, defaults to cache/prices.
    force_refresh : bool
        If True, force re-download even if cache exists.
    cache_key : str or None
        Optional explicit cache key. If None, auto key is used.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of adjusted close prices.
    """
    if config_path is None:
        config_path = _default_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Create it first (see example in config/data_config.json)."
        )

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "tickers" not in cfg or not isinstance(cfg["tickers"], list):
        raise ValueError("Config must contain key 'tickers' as a list of symbols.")

    tickers = [str(t).upper() for t in cfg["tickers"]]
    start = str(cfg.get("start_date", "2016-01-01"))
    end = cfg.get("end_date", None)
    if end is not None:
        end = str(end)

    # For now we ignore cfg["interval"] and always use '1d'.
    # If we later support intraday, we can thread it through.

    return load_prices(
        tickers=tickers,
        start=start,
        end=end,
        data_dir=data_dir,
        cache_key=cache_key,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/prices.py
    #
    # It will:
    #   - read config/data_config.json
    #   - download & cache prices if needed
    #   - print a small summary
    try:
        px = load_prices_from_config()
    except FileNotFoundError as e:
        print(e)
        print(
            "\nCreate config/data_config.json first. Example:\n"
            '{\n'
            '  "data_source": "yahoo",\n'
            '  "tickers": ["SPY", "QQQ", "IWM"],\n'
            '  "start_date": "2016-01-01",\n'
            '  "end_date": null,\n'
            '  "interval": "1d"\n'
            '}\n'
        )
    else:
        print("\n=== Price panel summary ===")
        print(f"Shape:   {px.shape[0]} rows × {px.shape[1]} tickers")
        print(f"Start:   {px.index.min().date()}")
        print(f"End:     {px.index.max().date()}")
        print(f"Tickers: {', '.join(px.columns.astype(str))}")
