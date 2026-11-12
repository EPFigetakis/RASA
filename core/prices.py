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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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


def panel_fingerprint(px: pd.DataFrame) -> str:
    """
    Content hash of a price panel (columns, dates and every value).

    Used to tie derived artifacts (e.g. saved pair selections) to the exact
    prices they were computed from: any change to a ticker, date or price
    changes the fingerprint.
    """
    h = hashlib.sha256()
    h.update(",".join(map(str, px.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(px, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def _write_cache_metadata(
    cache_path: Path,
    tickers: List[str],
    start: str,
    end: Optional[str],
    interval: str,
    px: pd.DataFrame,
    created_at: Optional[str] = None,
) -> None:
    """Write a sidecar JSON file recording what's actually in the cache."""
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "tickers": sorted(t.upper() for t in tickers),
        "start": start,
        "end": end,
        "interval": interval,
        "created_at": created_at or now,
        "updated_at": now,
        "n_tickers_kept": int(px.shape[1]),
        "n_rows": int(px.shape[0]),
        "last_bar_date": px.index.max().date().isoformat() if len(px) else None,
        "panel_fingerprint": panel_fingerprint(px),
    }
    _metadata_path(cache_path).write_text(json.dumps(meta, indent=2))


def read_cache_metadata(cache_path: Path) -> Optional[dict]:
    """Read the sidecar metadata for a cached parquet file, if present."""
    meta_path = _metadata_path(cache_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

DEFAULT_STALE_AFTER_BUSINESS_DAYS = 2


def latest_expected_bar(today: Optional[date] = None) -> date:
    """
    Most recent weekday whose daily bar should be complete: the weekday
    strictly before `today`. Exchange holidays are not modelled, which is why
    staleness is judged with a small tolerance rather than exactly.
    """
    today_ts = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
    return (today_ts - pd.offsets.BDay(1)).date()


def business_days_behind(last_bar: date, today: Optional[date] = None) -> int:
    """Whole weekdays between the cache's last bar and the latest expected bar."""
    expected = latest_expected_bar(today)
    if last_bar >= expected:
        return 0
    return int(np.busday_count(last_bar, expected))


def is_stale(
    last_bar: date,
    today: Optional[date] = None,
    stale_after_business_days: int = DEFAULT_STALE_AFTER_BUSINESS_DAYS,
) -> bool:
    """True if the last bar is more than `stale_after_business_days` weekdays behind."""
    return business_days_behind(last_bar, today) > stale_after_business_days


def get_cache_status(
    tickers: List[str],
    start: str,
    end: Optional[str],
    data_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    today: Optional[date] = None,
    stale_after_business_days: int = DEFAULT_STALE_AFTER_BUSINESS_DAYS,
) -> Dict[str, Any]:
    """
    Describe the price cache for a universe without loading or downloading it.

    Returns a dict with: exists, cache_path, last_bar_date, updated_at,
    business_days_behind, is_stale, fixed_end (True if `end` is set, in which
    case new data can never arrive and staleness is not an issue).
    """
    data_dir = data_dir or (paths.cache_dir() / "prices")
    tickers = [t.upper() for t in tickers]
    cache_key = cache_key or build_cache_key(tickers, start, end, interval="1d")
    cache_path = data_dir / f"{cache_key}.parquet"

    status: Dict[str, Any] = {
        "exists": cache_path.exists(),
        "cache_path": cache_path,
        "last_bar_date": None,
        "updated_at": None,
        "business_days_behind": None,
        "is_stale": False,
        "fixed_end": end is not None,
    }
    if not status["exists"]:
        return status

    meta = read_cache_metadata(cache_path) or {}
    last_bar = meta.get("last_bar_date")
    if last_bar is None:  # legacy cache without sidecar: read the index
        idx = pd.read_parquet(cache_path).index
        last_bar = pd.Timestamp(idx.max()).date().isoformat()
    last_bar_d = date.fromisoformat(last_bar)
    status["last_bar_date"] = last_bar
    status["updated_at"] = meta.get("updated_at") or meta.get("created_at")
    status["business_days_behind"] = business_days_behind(last_bar_d, today)
    status["is_stale"] = (
        end is None and is_stale(last_bar_d, today, stale_after_business_days)
    )
    return status


# ---------------------------------------------------------------------------
# Download + incremental update
# ---------------------------------------------------------------------------

def _download_close_panel(
    tickers: List[str],
    start: str,
    end: Optional[str],
) -> pd.DataFrame:
    """Download adjusted closes from Yahoo and return a cleaned wide panel
    (sorted, all-NaN rows dropped, gaps forward-filled; coverage NOT filtered)."""
    # yfinance returns MultiIndex columns (field, ticker) for several tickers.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    try:
        px = raw["Close"]
    except Exception:
        log(f"Unexpected columns from yfinance: {raw.columns}")
        raise

    if isinstance(px, pd.Series):  # single ticker
        px = px.to_frame()

    px = px.sort_index().dropna(how="all").ffill()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    return px


# Adjusted closes are restated retroactively when a dividend or split happens,
# so appending new bars to an old cache is only safe if the overlap still agrees.
OVERLAP_DAYS = 7
OVERLAP_REL_TOL = 1e-4


def update_cached_prices(
    cached: pd.DataFrame,
    tickers: List[str],
    start: str,
    end: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Extend a cached panel with bars newer than its last date.

    Re-downloads a short overlap window and compares it with the cache. If any
    overlapping price differs by more than OVERLAP_REL_TOL (a dividend/split
    restated history), returns None so the caller does a full re-download
    instead of splicing two inconsistent adjustment bases together.

    Only the cached columns are extended (tickers dropped by the coverage
    filter stay dropped).
    """
    if cached.empty:
        return None
    cols = list(cached.columns)
    last_bar = cached.index.max()
    fetch_from = (last_bar - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()

    fresh = _download_close_panel(cols, fetch_from, end)
    fresh = fresh.reindex(columns=cols)
    if fresh.empty:
        return cached

    overlap_idx = fresh.index.intersection(cached.index)
    if len(overlap_idx) == 0:
        return None
    old = cached.loc[overlap_idx]
    new = fresh.loc[overlap_idx]
    both = old.notna() & new.notna()
    rel = ((new - old).abs() / old.abs().where(old.abs() > 0)).where(both)
    if np.nanmax(rel.to_numpy(dtype=float), initial=0.0) > OVERLAP_REL_TOL:
        log("Overlap with cache disagrees (adjusted history was restated); full re-download needed.")
        return None

    added = fresh.loc[fresh.index > last_bar]
    if added.empty:
        return cached
    out = pd.concat([cached, added]).sort_index().ffill()
    return out


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
    update_if_stale: bool = False,
    stale_after_business_days: int = DEFAULT_STALE_AFTER_BUSINESS_DAYS,
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
    update_if_stale : bool
        If True and the cache's last bar is more than `stale_after_business_days`
        weekdays behind (only when `end` is None), append the missing bars
        incrementally (see `update_cached_prices`); falls back to a full
        re-download if history was restated, and to the old cache (with a
        warning) if the download fails.
    stale_after_business_days : int
        Tolerance before a cache counts as stale (absorbs holidays).

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
    need_full_download = True
    if cache_path.exists() and not force_refresh:
        need_full_download = False
        log(f"Loading cached prices from {cache_path.name} …")
        meta = read_cache_metadata(cache_path)
        px = pd.read_parquet(cache_path)
        if not isinstance(px.index, pd.DatetimeIndex):
            px.index = pd.to_datetime(px.index)

        if len(px):
            last_bar = px.index.max().date()
            behind = business_days_behind(last_bar)
            stale = end is None and behind > stale_after_business_days
            log(f"Cache last bar: {last_bar} ({behind} business day(s) behind)")
            if stale and update_if_stale:
                try:
                    updated = update_cached_prices(px, tickers, start, end)
                except Exception as e:  # network/vendor failure: keep serving old data
                    log(f"[WARN] Incremental update failed ({e}); using existing cache.")
                    return px
                if updated is None:
                    log("Falling back to a full re-download.")
                    need_full_download = True
                else:
                    if len(updated) != len(px):
                        log(f"Appended {len(updated) - len(px)} new bar(s).")
                    updated.to_parquet(cache_path)
                    _write_cache_metadata(
                        cache_path, tickers, start, end, "1d", updated,
                        created_at=(meta or {}).get("created_at"),
                    )
                    return updated
            elif stale:
                log("[WARN] Cache is stale; pass update_if_stale=True or force_refresh=True.")
        if not need_full_download:
            return px

    # Otherwise, download from Yahoo
    log(f"Downloading {len(tickers)} tickers from Yahoo Finance …")
    px = _download_close_panel(tickers, start, end)

    # Keep only columns with >98% non-NaN coverage
    coverage = px.notna().mean()
    keep_cols = coverage.loc[lambda s: s > 0.98].index
    px = px[keep_cols]

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
    update_if_stale: bool = False,
    stale_after_business_days: int = DEFAULT_STALE_AFTER_BUSINESS_DAYS,
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
    update_if_stale : bool
        Append missing recent bars if the cache is stale (see `load_prices`).

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
        update_if_stale=update_if_stale,
        stale_after_business_days=stale_after_business_days,
    )


def get_cache_status_from_config(
    config_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    """`get_cache_status` for the universe in config/data_config.json."""
    config_path = Path(config_path) if config_path else _default_config_path()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    end = cfg.get("end_date")
    return get_cache_status(
        tickers=[str(t).upper() for t in cfg["tickers"]],
        start=str(cfg.get("start_date", "2016-01-01")),
        end=str(end) if end is not None else None,
        data_dir=data_dir,
        cache_key=cache_key,
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
