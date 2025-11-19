"""
core.pair_selection

Scan a universe of tickers to find candidate stat-arb pairs.

This module:
- Works on top of the wide price panel from core.panel / core.prices.
- For each pair (X, Y) of tickers:
    * extracts their price series,
    * estimates the hedge ratio (via core.pairs),
    * runs an Engle-Granger cointegration test,
    * computes the spread and its half-life of mean reversion.
- Returns a ranked table of pair statistics.

You can:
- Use this table to pick which pairs to trade.
- Later, make the *trading* regime-aware using core.regimes.

Typical notebook usage:

    from core import panel, pair_selection

    px, rets = panel.load_prices_and_returns_from_config()

    cfg = pair_selection.PairSelectionConfig(
        max_pvalue=0.05,
        min_obs=252,
        use_log_prices=False,
        add_intercept=False,
        max_pairs=50,
    )

    pairs_df = pair_selection.find_candidate_pairs(px, cfg)
    print(pairs_df.head())

"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# External dependency for Engle-Granger cointegration test
try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    coint = None  # type: ignore

# Local imports (repo-style)
try:
    import core.panel as panel
    import core.pairs as pairs_mod
    import core.paths as paths
except ModuleNotFoundError:
    # Fallback if run from inside core/ directory
    import panel  # type: ignore
    import pairs as pairs_mod  # type: ignore
    import paths  # type: ignore


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Simple logging helper."""
    print(f"[pair_selection] {msg}")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is a DatetimeIndex and sorted."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PairSelectionConfig:
    """
    Configuration for scanning and ranking pairs.

    Attributes
    ----------
    max_pvalue : float
        Maximum Engle-Granger p-value for a pair to be considered.
    min_obs : int
        Minimum number of overlapping observations for a pair to be considered.
    use_log_prices : bool
        Whether to use log prices when estimating the hedge ratio.
    add_intercept : bool
        Whether to include an intercept term when estimating the hedge ratio.
    max_pairs : int or None
        If not None, keep only the top-N pairs sorted by p-value.
    save_to_csv : bool
        Whether to save the results to a CSV file under cache/pairs/.
    cache_subdir : str
        Subdirectory under cache/ where the CSV will be written.
    filename : str
        Name of the CSV file to write (if save_to_csv=True).
    """
    max_pvalue: float = 0.05
    min_obs: int = 252
    use_log_prices: bool = False
    add_intercept: bool = False
    max_pairs: Optional[int] = 50
    save_to_csv: bool = True
    cache_subdir: str = "pairs"
    filename: str = "pair_candidates.csv"


@dataclass
class PairStat:
    """
    Statistics for a single pair (X, Y).

    Attributes
    ----------
    x : str
        Hedge asset symbol.
    y : str
        Target asset symbol.
    pvalue : float
        Engle-Granger cointegration p-value.
    hedge_ratio : float
        Estimated hedge ratio beta in Y ≈ alpha + beta X.
    alpha : float
        Intercept in hedge regression (0 if add_intercept=False).
    half_life : float
        Estimated half-life of mean reversion for the spread (in days).
        np.inf or np.nan if not mean-reverting.
    n_obs : int
        Number of overlapping observations used.
    """
    x: str
    y: str
    pvalue: float
    hedge_ratio: float
    alpha: float
    half_life: float
    n_obs: int


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion for a spread using AR(1) on residuals.

    We model:
        eps_t = rho * eps_{t-1} + u_t

    Equivalent to:
        Δeps_t = (rho - 1) * eps_{t-1} + u_t

    We:
        - regress Δeps_t on eps_{t-1},
        - get slope λ ≈ (rho - 1),
        - so rho ≈ λ + 1.

    Half-life (in days):
        hl = -ln(2) / ln(rho),   for 0 < rho < 1

    If rho <= 0 or rho >= 1, we treat half-life as infinite (no mean reversion).

    Parameters
    ----------
    spread : pd.Series

    Returns
    -------
    float
        Half-life in days (may be np.inf or np.nan).
    """
    eps = spread.dropna().astype(float)
    if len(eps) < 3:
        return np.nan

    delta = eps.diff().dropna()
    lagged = eps.shift(1).dropna()

    # Align indices
    common_index = delta.index.intersection(lagged.index)
    delta = delta.loc[common_index]
    lagged = lagged.loc[common_index]

    if len(delta) < 3:
        return np.nan

    x = lagged.to_numpy()
    y = delta.to_numpy()

    # Regress y = const + λ x via least squares
    X_mat = np.column_stack([x, np.ones_like(x)])
    try:
        theta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    lam = float(theta[0])      # slope ≈ (rho - 1)
    rho = lam + 1.0

    # If rho out of (0,1), half-life not well-defined or infinite
    if rho <= 0 or rho >= 1:
        return np.inf

    hl = -np.log(2.0) / np.log(rho)
    return float(hl)


def evaluate_pair(
    px: pd.DataFrame,
    x: str,
    y: str,
    cfg: PairSelectionConfig,
) -> Optional[PairStat]:
    """
    Compute statistics for one pair (X, Y).

    Steps:
    - extract aligned price series for X and Y,
    - estimate hedge ratio,
    - compute cointegration p-value (Engle-Granger),
    - compute spread & half-life.

    Returns
    -------
    PairStat or None
        None if not enough data or cointegration computation fails.
    """
    # Extract aligned prices via core.pairs helpers
    spec = pairs_mod.PairSpec(x=x, y=y)
    try:
        df_pair = pairs_mod.extract_pair_prices(px, spec)
    except Exception:
        return None

    df_pair = df_pair.dropna(how="any")
    n_obs = len(df_pair)
    if n_obs < cfg.min_obs:
        return None

    # Estimate hedge ratio using same logic as trading module
    hedge_cfg = pairs_mod.HedgeConfig(
        use_log_prices=cfg.use_log_prices,
        add_intercept=cfg.add_intercept,
    )
    try:
        alpha, beta = pairs_mod.estimate_hedge_ratio(df_pair, hedge_cfg)
    except Exception:
        return None

    # Cointegration test (Engle-Granger)
    if coint is None:
        raise ImportError(
            "statsmodels is not installed. Install it with:\n"
            "    pip install statsmodels"
        )

    # statsmodels' coint expects y, x as arrays
    y_series = df_pair["Y"].astype(float)
    x_series = df_pair["X"].astype(float)

    try:
        _score, pvalue, _crit = coint(y_series, x_series)
    except Exception:
        return None

    # Spread & half-life
    spread = pairs_mod.compute_spread(df_pair, alpha, beta, hedge_cfg)
    half_life = _estimate_half_life(spread)

    return PairStat(
        x=x,
        y=y,
        pvalue=float(pvalue),
        hedge_ratio=float(beta),
        alpha=float(alpha),
        half_life=float(half_life),
        n_obs=int(n_obs),
    )


# ---------------------------------------------------------------------------
# Main scanning function
# ---------------------------------------------------------------------------

def find_candidate_pairs(
    px: pd.DataFrame,
    cfg: PairSelectionConfig,
) -> pd.DataFrame:
    """
    Scan all combinations of tickers in the price panel and return
    a ranked table of candidate pairs.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price panel (index = date, columns = tickers).
    cfg : PairSelectionConfig
        Configuration for scanning and ranking.

    Returns
    -------
    pd.DataFrame
        Table with one row per candidate pair, columns:
            - x, y
            - pvalue
            - hedge_ratio
            - alpha
            - half_life
            - n_obs
        Sorted by pvalue ascending. Filtered by:
            * pvalue <= cfg.max_pvalue
            * n_obs  >= cfg.min_obs
    """
    px = _ensure_datetime_index(px)
    tickers: List[str] = list(px.columns.astype(str))

    if len(tickers) < 2:
        raise ValueError("Need at least two tickers in the price panel to find pairs.")

    log(f"Scanning {len(tickers)} tickers -> {len(tickers) * (len(tickers) - 1) // 2} pairs …")

    stats: List[PairStat] = []

    for i, (x, y) in enumerate(combinations(tickers, 2), start=1):
        stat = evaluate_pair(px, x, y, cfg)
        if stat is None:
            continue

        # Basic filtering by p-value
        if stat.pvalue <= cfg.max_pvalue:
            stats.append(stat)

        if i % 50 == 0:
            log(f"Processed {i} pairs … (current kept: {len(stats)})")

    if not stats:
        log("No candidate pairs found with the given criteria.")
        return pd.DataFrame(
            columns=["x", "y", "pvalue", "hedge_ratio", "alpha", "half_life", "n_obs"]
        )

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "x": s.x,
                "y": s.y,
                "pvalue": s.pvalue,
                "hedge_ratio": s.hedge_ratio,
                "alpha": s.alpha,
                "half_life": s.half_life,
                "n_obs": s.n_obs,
            }
            for s in stats
        ]
    )

    # Sort by p-value ascending
    df = df.sort_values(["pvalue", "x", "y"]).reset_index(drop=True)

    # Keep only top-N if requested
    if cfg.max_pairs is not None and len(df) > cfg.max_pairs:
        df = df.iloc[: cfg.max_pairs].reset_index(drop=True)

    # Save to CSV if requested
    if cfg.save_to_csv:
        out_dir = paths.cache_dir() / cfg.cache_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cfg.filename
        df.to_csv(out_path, index=False)
        log(f"Saved {len(df)} candidate pairs to {out_path}")

    return df


# ---------------------------------------------------------------------------
# Config-based helper
# ---------------------------------------------------------------------------

def find_candidate_pairs_from_config(
    cfg: PairSelectionConfig,
    return_method: str = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level helper:

    - Load price panel from config/data_config.json via core.panel.
    - Scan for candidate pairs using the provided PairSelectionConfig.

    Parameters
    ----------
    cfg : PairSelectionConfig
        Scanning configuration.
    return_method : {"simple", "log"}
        Forwarded to panel.load_prices_and_returns_from_config (not critical here).
    force_refresh : bool
        Force re-download of prices.
    cache_key : str or None
        Optional explicit cache key for price parquet.

    Returns
    -------
    (px, pairs_df) : (pd.DataFrame, pd.DataFrame)
        px       : price panel
        pairs_df : table of candidate pairs.
    """
    px, _rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )
    pairs_df = find_candidate_pairs(px, cfg)
    return px, pairs_df


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/pair_selection.py
    #
    # It will:
    #   - load prices from config/data_config.json,
    #   - scan all ticker pairs,
    #   - print top 10 and save CSV in cache/pairs/.
    cfg = PairSelectionConfig(
        max_pvalue=0.05,
        min_obs=252,
        use_log_prices=False,
        add_intercept=False,
        max_pairs=50,
    )

    try:
        px, pairs_df = find_candidate_pairs_from_config(cfg)
    except FileNotFoundError as e:
        print(e)
        print(
            "\nCreate config/data_config.json first. Example:\n"
            '{\n'
            '  "data_source": "yahoo",\n'
            '  "tickers": ["SPY", "QQQ", "IWM", "EEM"],\n'
            '  "start_date": "2016-01-01",\n'
            '  "end_date": null,\n'
            '  "interval": "1d"\n'
            '}\n'
        )
    except ImportError as e:
        print(e)
    else:
        print("\n=== Pair selection summary ===")
        print(f"Universe tickers: {list(px.columns.astype(str))}")
        print(f"Found {len(pairs_df)} candidate pairs.")
        print(pairs_df.head(10))
