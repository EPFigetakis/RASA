"""
core.panel

Utilities for working with the price panel:

- Compute simple or log returns from the wide price matrix.
- Provide a convenience helper to go from config -> prices -> returns.

Assumes prices are loaded using `core.prices.load_prices(...)`, which returns:

    px: pd.DataFrame
        index  = DatetimeIndex (trading days)
        columns = tickers (symbols)
        values = adjusted close prices

This module is deliberately small and focused. It's the "bridge" between
raw price data and downstream models (regimes, stat-arb, etc.).
"""

from __future__ import annotations

from typing import Tuple, Literal, Optional

import numpy as np
import pandas as pd

# Local import for price loading
try:
    import core.prices as prices
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import prices  # type: ignore


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------

ReturnMethod = Literal["simple", "log"]


def compute_returns(
    px: pd.DataFrame,
    method: ReturnMethod = "log",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Compute returns from a wide price panel.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price DataFrame:
            - index: DatetimeIndex (sorted)
            - columns: tickers
            - values: prices (typically adjusted close)
    method : {"simple", "log"}
        "simple":  r_t = P_t / P_{t-1} - 1
        "log":     r_t = log(P_t) - log(P_{t-1})
    drop_na : bool
        If True, drop the first row (which is NaN by definition) and any rows
        that end up all-NaN after differencing. If False, NaNs are kept.

    Returns
    -------
    pd.DataFrame
        Wide return DataFrame with the same columns as `px` and aligned index.
    """
    if not isinstance(px.index, pd.DatetimeIndex):
        # enforce DatetimeIndex for sanity downstream
        px = px.copy()
        px.index = pd.to_datetime(px.index)

    # Ensure sorted by date
    px = px.sort_index()

    if method == "simple":
        ret = px.pct_change()
    elif method == "log":
        # Avoid numerical issues with zeros/negatives by clipping
        px_safe = px.clip(lower=1e-12)
        ret = np.log(px_safe).diff()
    else:
        raise ValueError(f"Unknown return method: {method!r}. Use 'simple' or 'log'.")

    if drop_na:
        # Drop rows that are all NaN (typically the first one)
        ret = ret.dropna(how="all")

    return ret


def normalize_prices(
    px: pd.DataFrame,
    base_value: float = 1.0,
) -> pd.DataFrame:
    """
    Normalize each price series to start at a common base value.

    This is useful for quick visualization of relative performance between
    tickers.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price DataFrame.
    base_value : float
        Starting value after normalization (e.g., 1.0 or 100.0).

    Returns
    -------
    pd.DataFrame
        Normalized price panel where each column starts at `base_value`.
    """
    if not isinstance(px.index, pd.DatetimeIndex):
        px = px.copy()
        px.index = pd.to_datetime(px.index)

    px = px.sort_index()

    # Divide each column by its first non-NaN value
    first_vals = px.ffill().iloc[0]
    norm = px.divide(first_vals) * base_value

    return norm


# ---------------------------------------------------------------------------
# Config-based helper (prices + returns)
# ---------------------------------------------------------------------------

def load_prices_and_returns_from_config(
    method: ReturnMethod = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
    update_if_stale: bool = False,
    stale_after_business_days: int = prices.DEFAULT_STALE_AFTER_BUSINESS_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function:
    - Read tickers/dates from config/data_config.json
    - Load (or download + cache) prices via core.prices
    - Compute returns using the chosen method

    Parameters
    ----------
    method : {"simple", "log"}
        Return definition:
            "simple":  r_t = P_t / P_{t-1} - 1
            "log":     r_t = log(P_t) - log(P_{t-1})
    force_refresh : bool
        Passed down to `prices.load_prices_from_config` to force re-download.
    cache_key : str or None
        Optional explicit cache key for the price parquet. If None, the
        automatic key from `load_prices` is used.
    update_if_stale : bool
        Append missing recent bars if the cached prices are stale.

    Returns
    -------
    (px, rets) : (pd.DataFrame, pd.DataFrame)
        px   : price panel (adjusted close, wide format)
        rets : return panel (same shape, shifted by 1 in time)
    """
    px = prices.load_prices_from_config(
        force_refresh=force_refresh,
        cache_key=cache_key,
        update_if_stale=update_if_stale,
        stale_after_business_days=stale_after_business_days,
    )
    rets = compute_returns(px, method=method, drop_na=True)
    return px, rets


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/panel.py
    #
    # It will:
    #   - read config/data_config.json
    #   - load prices (using cache if available)
    #   - compute log returns
    #   - print a small summary
    try:
        px, rets = load_prices_and_returns_from_config(method="log")
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
        print("\n=== Panel summary ===")
        print(f"Price panel:    {px.shape[0]} rows × {px.shape[1]} tickers")
        print(f"Return panel:   {rets.shape[0]} rows × {rets.shape[1]} tickers")
        print(f"Price span:     {px.index.min().date()} -> {px.index.max().date()}")
        print(f"Return span:    {rets.index.min().date()} -> {rets.index.max().date()}")
        print(f"Tickers:        {', '.join(px.columns.astype(str))}")
