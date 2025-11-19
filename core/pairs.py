"""
core.pairs

Statistical arbitrage building blocks for *one* pair of assets.

This module is deliberately focused and simple:
- It does NOT pick pairs for you (no cointegration search yet).
- It assumes you *choose* a pair (X, Y) you believe is mean-reverting.
- It then:
    - extracts their price series from a wide panel,
    - estimates a hedge ratio,
    - builds a spread,
    - computes a rolling z-score,
    - generates basic entry/exit trading signals.

Later, you can:
- Add cointegration tests / pair selection on top.
- Make the thresholds regime-aware by conditioning on core.regimes labels.

Typical usage from a notebook:

    from core import panel, pairs

    # 1) Load prices from config
    px, rets = panel.load_prices_and_returns_from_config()

    # 2) Define the pair
    spec = pairs.PairSpec(x="SPY", y="QQQ")

    # 3) Run the full pipeline
    out = pairs.build_pair_signals_from_panel(px, spec)

    spread  = out.spread
    zscore  = out.zscore
    signals = out.signals

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Local import (repo-style)
try:
    import core.panel as panel
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import panel  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses for configuration
# ---------------------------------------------------------------------------

@dataclass
class PairSpec:
    """
    Definition of a pair (X, Y).

    By convention:
    - X is the "hedge" asset
    - Y is the "target" asset

    The spread is constructed as:
        spread_t = Y_t - hedge_ratio * X_t
    """
    x: str  # hedge asset symbol
    y: str  # target asset symbol


@dataclass
class HedgeConfig:
    """
    Configuration for hedge ratio estimation.

    Attributes
    ----------
    use_log_prices : bool
        If True, use log(prices) when estimating hedge ratio.
        If False, use raw prices.
    add_intercept : bool
        If True, estimate Y_t ≈ alpha + beta * X_t (with intercept).
        If False, estimate Y_t ≈ beta * X_t (through the origin).
    """
    use_log_prices: bool = False
    add_intercept: bool = False


@dataclass
class SignalConfig:
    """
    Trading rule parameters for a mean-reversion pair strategy.

    Attributes
    ----------
    entry_z : float
        Z-score magnitude at which to enter a position.
        - If z > entry_z: short spread (short Y, long X)
        - If z < -entry_z: long spread (long Y, short X)
    exit_z : float
        Z-score magnitude at which to close an existing position.
        Position is closed when |z| < exit_z.
    max_position : int
        Maximum absolute position size in "spread units".
        For now, signals are just -1, 0, +1, but this allows scaling later.
    """
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_position: int = 1


@dataclass
class PairPipelineResult:
    """
    Container for outputs of the pair signal pipeline.

    Attributes
    ----------
    spec : PairSpec
        Pair being traded.
    hedge_config : HedgeConfig
        Config used for hedge ratio estimation.
    signal_config : SignalConfig
        Config used for signal generation.
    hedge_ratio : float
        Estimated hedge ratio (beta) such that spread_t = Y_t - beta * X_t.
    spread : pd.Series
        Spread time series (index = date).
    zscore : pd.Series
        Rolling z-score of the spread.
    signals : pd.Series
        Trading signals:
            -1 : short spread (short Y, long X)
             0 : flat
            +1 : long spread (long Y, short X)
    metadata : dict
        Optional extra info (e.g., alpha, window, etc.).
    """
    spec: PairSpec
    hedge_config: HedgeConfig
    signal_config: SignalConfig
    hedge_ratio: float
    spread: pd.Series
    zscore: pd.Series
    signals: pd.Series
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is a DatetimeIndex and sorted."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def extract_pair_prices(
    px: pd.DataFrame,
    spec: PairSpec,
) -> pd.DataFrame:
    """
    Extract the price series for X and Y from a wide price panel.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price panel: index = date, columns = tickers.
    spec : PairSpec
        Pair definition.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns [X, Y] and aligned index.
    """
    px = _ensure_datetime_index(px)

    missing = [s for s in (spec.x, spec.y) if s not in px.columns]
    if missing:
        raise KeyError(f"Missing tickers in price panel: {missing}")

    df_pair = px[[spec.x, spec.y]].copy()
    df_pair.columns = ["X", "Y"]
    df_pair = df_pair.dropna(how="any")

    return df_pair


def estimate_hedge_ratio(
    df_pair: pd.DataFrame,
    cfg: HedgeConfig,
) -> Tuple[float, float]:
    """
    Estimate hedge ratio beta (and optionally intercept alpha)
    between X and Y.

    We model:
        Y_t ≈ alpha + beta * X_t       (if add_intercept=True)
    or:
        Y_t ≈ beta * X_t               (if add_intercept=False)

    Parameters
    ----------
    df_pair : pd.DataFrame
        DataFrame with columns ["X", "Y"].
    cfg : HedgeConfig
        Configuration for using log prices and intercept.

    Returns
    -------
    (alpha, beta) : tuple of floats
        Intercept and slope of the regression.
        If cfg.add_intercept is False, alpha will be 0.0.
    """
    x = df_pair["X"].astype(float).to_numpy()
    y = df_pair["Y"].astype(float).to_numpy()

    if cfg.use_log_prices:
        x = np.log(np.clip(x, 1e-12, np.inf))
        y = np.log(np.clip(y, 1e-12, np.inf))

    if cfg.add_intercept:
        # Design matrix with intercept: [1, X_t]
        X_mat = np.column_stack([np.ones_like(x), x])
        # Solve least squares: minimize ||Y - X_mat @ theta||
        theta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
        alpha, beta = float(theta[0]), float(theta[1])
    else:
        # Through-origin regression: minimize ||Y - beta X||
        # Closed-form: beta = (X'Y) / (X'X)
        denom = np.dot(x, x)
        if denom == 0:
            raise ValueError("Cannot estimate hedge ratio: X has zero variance.")
        beta = float(np.dot(x, y) / denom)
        alpha = 0.0

    return alpha, beta


def compute_spread(
    df_pair: pd.DataFrame,
    alpha: float,
    beta: float,
    cfg: HedgeConfig,
) -> pd.Series:
    """
    Compute spread_t = Y_t - (alpha + beta * X_t), using either raw or log prices
    according to cfg.use_log_prices.

    For trading, the raw spread (in price space) is usually used, but if we
    estimated on logs we keep it consistent and use the log-based spread.

    Parameters
    ----------
    df_pair : pd.DataFrame
        Columns ["X", "Y"], aligned and cleaned.
    alpha : float
        Intercept from hedge ratio estimation.
    beta : float
        Hedge ratio.
    cfg : HedgeConfig
        Determines whether we use raw or log prices for the spread.

    Returns
    -------
    pd.Series
        Spread time series, index = date.
    """
    if cfg.use_log_prices:
        x = np.log(np.clip(df_pair["X"].to_numpy(dtype=float), 1e-12, np.inf))
        y = np.log(np.clip(df_pair["Y"].to_numpy(dtype=float), 1e-12, np.inf))
    else:
        x = df_pair["X"].to_numpy(dtype=float)
        y = df_pair["Y"].to_numpy(dtype=float)

    spread_vals = y - (alpha + beta * x)
    spread = pd.Series(spread_vals, index=df_pair.index, name="spread")
    return spread


def compute_spread_zscore(
    spread: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute rolling z-score of the spread.

    z_t = (spread_t - mean_t) / std_t       over the given window.

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window length in days.
    min_periods : int or None
        Minimum observations in window. If None, defaults to window // 2.

    Returns
    -------
    pd.Series
        Z-score series aligned with 'spread'.
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    spread = spread.sort_index()
    roll = spread.rolling(window=window, min_periods=min_periods)

    mean = roll.mean()
    std = roll.std()

    z = (spread - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore"

    return z


def generate_pair_signals(
    zscore: pd.Series,
    cfg: SignalConfig,
) -> pd.Series:
    """
    Generate mean-reversion trading signals from a z-score series.

    Logic:
        - If currently flat (pos=0):
            - if z > entry_z:  pos -> -max_position (short spread)
            - if z < -entry_z: pos -> +max_position (long spread)
        - If currently in a position:
            - if |z| < exit_z: pos -> 0 (exit)
            - else: keep current position

    Parameters
    ----------
    zscore : pd.Series
        Spread z-score time series.
    cfg : SignalConfig
        Entry/exit thresholds and max_position.

    Returns
    -------
    pd.Series
        Position series:
            -cfg.max_position ... 0 ... +cfg.max_position
    """
    z = zscore.sort_index()
    pos_vals = np.zeros(len(z), dtype=float)

    current_pos = 0.0
    for i, z_t in enumerate(z.to_numpy()):
        if np.isnan(z_t):
            # keep previous position if z is NaN
            pos_vals[i] = current_pos
            continue

        if current_pos == 0.0:
            # Flat: consider entry
            if z_t > cfg.entry_z:
                current_pos = -float(cfg.max_position)  # short spread
            elif z_t < -cfg.entry_z:
                current_pos = float(cfg.max_position)   # long spread
        else:
            # In a position: consider exit
            if abs(z_t) < cfg.exit_z:
                current_pos = 0.0

        pos_vals[i] = current_pos

    signals = pd.Series(pos_vals, index=z.index, name="position")
    return signals


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def build_pair_signals_from_panel(
    px: pd.DataFrame,
    spec: PairSpec,
    hedge_cfg: Optional[HedgeConfig] = None,
    signal_cfg: Optional[SignalConfig] = None,
    zscore_window: int = 60,
) -> PairPipelineResult:
    """
    End-to-end pipeline for one pair (X, Y):

    - Extract X, Y prices from wide panel.
    - Estimate hedge ratio (alpha, beta).
    - Compute spread.
    - Compute rolling z-score.
    - Generate trading signals.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price panel.
    spec : PairSpec
        Pair definition.
    hedge_cfg : HedgeConfig or None
        If None, use default HedgeConfig().
    signal_cfg : SignalConfig or None
        If None, use default SignalConfig().
    zscore_window : int
        Rolling window length for z-score.

    Returns
    -------
    PairPipelineResult
    """
    if hedge_cfg is None:
        hedge_cfg = HedgeConfig()
    if signal_cfg is None:
        signal_cfg = SignalConfig()

    df_pair = extract_pair_prices(px, spec)
    alpha, beta = estimate_hedge_ratio(df_pair, hedge_cfg)
    spread = compute_spread(df_pair, alpha, beta, hedge_cfg)
    zscore = compute_spread_zscore(spread, window=zscore_window)
    signals = generate_pair_signals(zscore, signal_cfg)

    metadata: Dict[str, Any] = {
        "alpha": alpha,
        "beta": beta,
        "zscore_window": zscore_window,
    }

    return PairPipelineResult(
        spec=spec,
        hedge_config=hedge_cfg,
        signal_config=signal_cfg,
        hedge_ratio=beta,
        spread=spread,
        zscore=zscore,
        signals=signals,
        metadata=metadata,
    )


def build_pair_signals_from_config(
    spec: PairSpec,
    hedge_cfg: Optional[HedgeConfig] = None,
    signal_cfg: Optional[SignalConfig] = None,
    zscore_window: int = 60,
    return_method: str = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, PairPipelineResult]:
    """
    High-level convenience pipeline:

    - Load prices from config/data_config.json via core.panel.
    - Run the pair signal pipeline on those prices.

    Parameters
    ----------
    spec : PairSpec
        Pair definition (X, Y).
    hedge_cfg : HedgeConfig or None
        Hedge ratio configuration.
    signal_cfg : SignalConfig or None
        Trading signal configuration.
    zscore_window : int
        Z-score window length.
    return_method : {"simple", "log"}
        Currently not used directly here, but threaded through in case
        you want to also compute returns from core.panel.
    force_refresh : bool
        Force re-download of prices.
    cache_key : str or None
        Optional explicit cache key for price parquet.

    Returns
    -------
    (px, result) : (pd.DataFrame, PairPipelineResult)
        px     : price panel
        result : pipeline result for the specified pair.
    """
    px, _rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )

    result = build_pair_signals_from_panel(
        px=px,
        spec=spec,
        hedge_cfg=hedge_cfg,
        signal_cfg=signal_cfg,
        zscore_window=zscore_window,
    )
    return px, result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test from project root:
    #
    #   python core/pairs.py
    #
    # Requires that config/data_config.json has at least two tickers.
    try:
        # Load prices & find first two tickers
        px, rets = panel.load_prices_and_returns_from_config()
        tickers = list(px.columns.astype(str))
        if len(tickers) < 2:
            raise RuntimeError("Need at least two tickers in config to test pairs.")
        spec = PairSpec(x=tickers[0], y=tickers[1])

        result = build_pair_signals_from_panel(px, spec)

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
    except Exception as e:
        print(f"[pairs] Error in CLI test: {e}")
    else:
        print("\n=== Pair pipeline summary ===")
        print(f"Pair:          X={result.spec.x}, Y={result.spec.y}")
        print(f"Hedge ratio β: {result.hedge_ratio:.4f}")
        print(f"Spread length: {len(result.spread)}")
        print(f"Signal states: {sorted(result.signals.unique().tolist())}")
        print(f"Date span:     {result.spread.index.min().date()} -> {result.spread.index.max().date()}")
