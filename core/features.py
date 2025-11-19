"""
core.features

Feature engineering utilities built on top of the price/return panel.

This module is intentionally *generic*:
- It doesn't know about "pairs" or "stat arb" yet.
- It just turns prices/returns into time-series features like
  rolling returns, volatility, and Sharpe-like ratios.

Later, we'll build more specialized modules for:
- Regime detection (e.g., HMM on these features).
- Pair selection and spread/z-score computation for stat arb.

Typical usage:

    from core import panel, features

    # 1) From config: prices + log-returns
    px, rets = panel.load_prices_and_returns_from_config(method="log")

    # 2) Build basic rolling features
    feat = features.build_basic_features(
        rets,
        return_windows=(1, 5, 20),
        vol_windows=(20, 60),
        make_multiindex=True,
    )

    # feat is a DataFrame with MultiIndex columns: (feature_name, symbol)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Literal, Optional, Dict

import numpy as np
import pandas as pd

# Local imports (repo-style)
try:
    import core.panel as panel
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import panel  # type: ignore


ReturnMethod = Literal["simple", "log"]


# ---------------------------------------------------------------------------
# Core feature primitives
# ---------------------------------------------------------------------------

def compute_rolling_mean(
    rets: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling mean of returns.

    Parameters
    ----------
    rets : pd.DataFrame
        Wide return panel (index = date, columns = tickers).
    window : int
        Rolling window size in days.
    min_periods : int or None
        Minimum number of observations in window to have a value.
        If None, defaults to window // 2 (half the window).

    Returns
    -------
    pd.DataFrame
        Rolling mean, same shape as `rets`.
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    rets = _ensure_datetime_index(rets)
    return rets.sort_index().rolling(window=window, min_periods=min_periods).mean()


def compute_rolling_vol(
    rets: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    annualize: bool = True,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Rolling volatility (standard deviation) of returns.

    Parameters
    ----------
    rets : pd.DataFrame
        Wide return panel.
    window : int
        Rolling window size in days.
    min_periods : int or None
        Minimum observations in window. If None, defaults to window // 2.
    annualize : bool
        If True, multiply by sqrt(trading_days_per_year).
    trading_days_per_year : int
        Annualization factor for volatility.

    Returns
    -------
    pd.DataFrame
        Rolling vol, same shape as `rets`.
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    rets = _ensure_datetime_index(rets)
    vol = (
        rets.sort_index()
            .rolling(window=window, min_periods=min_periods)
            .std()
    )

    if annualize:
        vol = vol * np.sqrt(trading_days_per_year)

    return vol


def compute_rolling_sharpe(
    rets: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Rolling Sharpe-like ratio (ex-ante) based on returns.

    Definition (approximate):
        Sharpe_t ≈ (E[r] * trading_days_per_year) / (σ * sqrt(trading_days_per_year))

    Where E[r] and σ are computed over the rolling window.

    Parameters
    ----------
    rets : pd.DataFrame
        Wide return panel.
    window : int
        Rolling window in days.
    min_periods : int or None
        Minimum observations in window. If None, defaults to window // 2.
    trading_days_per_year : int
        Annualization factor.

    Returns
    -------
    pd.DataFrame
        Rolling Sharpe ratio, same shape as `rets`.
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    rets = _ensure_datetime_index(rets)
    rets = rets.sort_index()

    roll = rets.rolling(window=window, min_periods=min_periods)
    mean = roll.mean() * trading_days_per_year
    vol = roll.std() * np.sqrt(trading_days_per_year)

    sharpe = mean / vol
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    return sharpe


def compute_rolling_window_return(
    rets: pd.DataFrame,
    window: int,
    method: ReturnMethod = "log",
) -> pd.DataFrame:
    """
    Multi-day window return based on daily returns.

    For log returns:
        R_{t,window} = sum_{i=0}^{window-1} r_{t-i}
    (equivalent to log(P_t / P_{t-window})).

    For simple returns:
        R_{t,window} = (P_t / P_{t-window}) - 1
    which is approximated here via compounding simple daily returns.

    Parameters
    ----------
    rets : pd.DataFrame
        Daily returns panel.
    window : int
        Window length in days.
    method : {"simple", "log"}
        Interpretation of `rets`.

    Returns
    -------
    pd.DataFrame
        Multi-day return panel, same shape as `rets`.
    """
    rets = _ensure_datetime_index(rets)
    rets = rets.sort_index()

    if method == "log":
        # Rolling sum of log returns
        win_ret = rets.rolling(window=window, min_periods=1).sum()
    elif method == "simple":
        # (1 + r_1) * (1 + r_2) * ... * (1 + r_window) - 1
        win_ret = (1.0 + rets).rolling(window=window, min_periods=1).apply(
            lambda x: float(np.prod(x) - 1.0),
            raw=True,
        )
    else:
        raise ValueError(f"Unknown return method: {method!r}. Use 'simple' or 'log'.")

    return win_ret


# ---------------------------------------------------------------------------
# Feature set builder
# ---------------------------------------------------------------------------

def build_basic_features(
    rets: pd.DataFrame,
    return_windows: Sequence[int] = (1, 5, 20),
    vol_windows: Sequence[int] = (20, 60),
    sharpe_windows: Sequence[int] = (60,),
    return_method: ReturnMethod = "log",
    make_multiindex: bool = True,
    annualize_vol: bool = True,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Build a basic set of rolling features for each ticker.

    Features (by default):
        - ret_{wd}     : rolling window return over w days
        - vol_{wd}     : rolling volatility over w days
        - sharpe_{wd}  : rolling Sharpe over w days

    Parameters
    ----------
    rets : pd.DataFrame
        Daily return panel, wide format.
    return_windows : sequence of int
        Window sizes for multi-day returns (1 means just daily returns).
    vol_windows : sequence of int
        Window sizes for volatility.
    sharpe_windows : sequence of int
        Window sizes for Sharpe.
    return_method : {"simple", "log"}
        Interpretation of `rets` when computing multi-day returns.
    make_multiindex : bool
        If True, returns a DataFrame with MultiIndex columns:
            (feature_name, symbol)
        If False, returns a DataFrame with flat columns like:
            "ret_5d_SPY", "vol_20d_QQQ", etc.
    annualize_vol : bool
        Whether to annualize volatility in `compute_rolling_vol`.
    trading_days_per_year : int
        Annualization factor for volatility and Sharpe.

    Returns
    -------
    pd.DataFrame
        Feature panel indexed by date. Columns either:
            - MultiIndex (feature, symbol), if make_multiindex=True
            - flat strings "{feature_name}_{symbol}", otherwise.
    """
    rets = _ensure_datetime_index(rets)
    rets = rets.sort_index()

    feature_frames: Dict[str, pd.DataFrame] = {}

    # Rolling returns (multi-day)
    for w in return_windows:
        name = f"ret_{w}d"
        if w == 1:
            # 1-day window is just the daily returns
            feature_frames[name] = rets.copy()
        else:
            feature_frames[name] = compute_rolling_window_return(
                rets, window=w, method=return_method
            )

    # Rolling volatilities
    for w in vol_windows:
        name = f"vol_{w}d"
        feature_frames[name] = compute_rolling_vol(
            rets,
            window=w,
            annualize=annualize_vol,
            trading_days_per_year=trading_days_per_year,
        )

    # Rolling Sharpe ratios
    for w in sharpe_windows:
        name = f"sharpe_{w}d"
        feature_frames[name] = compute_rolling_sharpe(
            rets,
            window=w,
            trading_days_per_year=trading_days_per_year,
        )

    # Combine all feature DataFrames along columns
    feature_names = list(feature_frames.keys())
    frames = [feature_frames[name] for name in feature_names]

    # Make sure all frames share the same index
    # (they should, but we reindex to the union just in case)
    full_index = rets.index
    frames = [f.reindex(full_index) for f in frames]

    combined = pd.concat(frames, axis=1)

    if make_multiindex:
        # Columns: MultiIndex (feature_name, symbol)
        arrays_feature = []
        arrays_symbol = []
        for name, df in zip(feature_names, frames):
            arrays_feature.extend([name] * df.shape[1])
            arrays_symbol.extend(df.columns.astype(str).tolist())

        combined.columns = pd.MultiIndex.from_arrays(
            [arrays_feature, arrays_symbol],
            names=["feature", "symbol"],
        )
    else:
        # Flat columns: "<feature>_<symbol>"
        new_cols = []
        for name, df in zip(feature_names, frames):
            for sym in df.columns.astype(str):
                new_cols.append(f"{name}_{sym}")
        combined.columns = new_cols

    return combined


# ---------------------------------------------------------------------------
# Config-based helper (prices + returns + features)
# ---------------------------------------------------------------------------

def build_basic_features_from_config(
    return_method: ReturnMethod = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
    return_windows: Sequence[int] = (1, 5, 20),
    vol_windows: Sequence[int] = (20, 60),
    sharpe_windows: Sequence[int] = (60,),
    make_multiindex: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level convenience function:

    - Read tickers/dates from config/data_config.json.
    - Load (or download + cache) prices + returns.
    - Build a basic feature set on top of the returns.

    Parameters
    ----------
    return_method : {"simple", "log"}
        Method for computing returns.
    force_refresh : bool
        Force fresh download of prices.
    cache_key : str or None
        Optional explicit cache key for price parquet.
    return_windows, vol_windows, sharpe_windows :
        Passed through to `build_basic_features`.
    make_multiindex : bool
        Whether to return features with MultiIndex columns.

    Returns
    -------
    (px, rets, feat) : tuple of DataFrames
        px   : price panel (wide, adjusted close)
        rets : return panel
        feat : feature panel (MultiIndex or flat columns)
    """
    px, rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )
    feat = build_basic_features(
        rets=rets,
        return_windows=return_windows,
        vol_windows=vol_windows,
        sharpe_windows=sharpe_windows,
        return_method=return_method,
        make_multiindex=make_multiindex,
    )
    return px, rets, feat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/features.py
    #
    # It will:
    #   - load prices + log returns from config
    #   - build basic features
    #   - print a small summary
    try:
        px, rets, feat = build_basic_features_from_config()
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
        print("\n=== Features summary ===")
        print(f"Prices:    {px.shape[0]} rows × {px.shape[1]} tickers")
        print(f"Returns:   {rets.shape[0]} rows × {rets.shape[1]} tickers")
        print(f"Features:  {feat.shape[0]} rows × {feat.shape[1]} columns")
        print(f"Date span: {feat.index.min().date()} -> {feat.index.max().date()}")
        if isinstance(feat.columns, pd.MultiIndex):
            print("Feature levels:", feat.columns.levels[0].tolist())
        else:
            print("Example feature columns:", feat.columns[:5].tolist())
