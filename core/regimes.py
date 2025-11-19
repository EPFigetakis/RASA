"""
core.regimes

Regime detection on top of feature panels.

This module:
- Takes a time series of features (from core.features).
- Flattens the feature panel into a 2D matrix X (time × features).
- Optionally scales features.
- Fits a Gaussian HMM (from hmmlearn) to detect regimes.
- Returns a time-indexed Series of regime labels.

Design goals:
- Keep it simple and transparent for learning.
- Make it easy to swap out the model later if needed.

Typical usage from a notebook:

    from core import features, regimes

    # 1) Build features from config
    px, rets, feat = features.build_basic_features_from_config()

    # 2) Fit regimes
    cfg = regimes.RegimeHMMConfig(n_regimes=3, random_state=42)
    result = regimes.fit_regimes_from_features(feat, cfg)

    regimes_series = result.regimes
    print(regimes_series.head())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# External dependencies
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # type: ignore

from sklearn.preprocessing import StandardScaler

# Local imports (repo-style)
try:
    import core.features as features
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import features  # type: ignore


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeHMMConfig:
    """
    Configuration for the Gaussian HMM regime model.

    Attributes
    ----------
    n_regimes : int
        Number of hidden regimes (states) to infer.
    covariance_type : str
        Covariance type for GaussianHMM: "full", "diag", "tied", or "spherical".
    n_iter : int
        Maximum number of EM iterations.
    random_state : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, HMM training will print progress.
    scale_features : bool
        If True, apply StandardScaler to features before fitting.
    """
    n_regimes: int = 3
    covariance_type: str = "full"
    n_iter: int = 200
    random_state: Optional[int] = 0
    verbose: bool = False
    scale_features: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeFitResult:
    """
    Container for regime fit outputs.

    Attributes
    ----------
    regimes : pd.Series
        Regime label per date (index = DatetimeIndex).
    model : GaussianHMM
        Fitted HMM model.
    scaler : StandardScaler or None
        Fitted scaler (if scale_features=True) else None.
    X : np.ndarray
        Feature matrix used for fitting (after scaling if applicable).
    index : pd.DatetimeIndex
        Index corresponding to rows of X and regimes.
    metadata : dict
        Optional metadata (e.g., feature names).
    """
    regimes: pd.Series
    model: Any
    scaler: Optional[StandardScaler]
    X: np.ndarray
    index: pd.DatetimeIndex
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def prepare_feature_matrix(
    feat: pd.DataFrame,
    drop_na: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex, Dict[str, Any]]:
    """
    Convert a feature panel into a 2D numpy array suitable for modeling.

    Handles both:
        - MultiIndex columns (feature, symbol)
        - Flat columns

    Parameters
    ----------
    feat : pd.DataFrame
        Feature panel, index = date, columns = features (possibly MultiIndex).
    drop_na : bool
        If True, drop any rows that contain NaNs across any feature.

    Returns
    -------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    index : pd.DatetimeIndex
        Index corresponding to rows of X.
    metadata : dict
        Contains information about feature names and column structure.
            - "columns_original": original column index
            - "columns_flat": list of str for each feature column
    """
    feat = _ensure_datetime_index(feat)
    feat = feat.sort_index()

    metadata: Dict[str, Any] = {}

    # Keep a copy of original columns
    metadata["columns_original"] = feat.columns

    # Flatten columns to strings if they are MultiIndex
    if isinstance(feat.columns, pd.MultiIndex):
        flat_cols = [f"{lvl0}__{lvl1}" for lvl0, lvl1 in feat.columns.to_list()]
    else:
        flat_cols = [str(c) for c in feat.columns]

    metadata["columns_flat"] = flat_cols

    feat_flat = feat.copy()
    feat_flat.columns = flat_cols

    if drop_na:
        feat_flat = feat_flat.dropna(how="any")

    X = feat_flat.to_numpy(dtype=float)
    index = feat_flat.index

    return X, index, metadata


def fit_hmm_regimes(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    cfg: RegimeHMMConfig,
) -> RegimeFitResult:
    """
    Fit a Gaussian HMM to feature matrix X and return inferred regimes.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    index : pd.DatetimeIndex
        Time index for rows of X.
    cfg : RegimeHMMConfig
        Configuration for the HMM and scaling.

    Returns
    -------
    RegimeFitResult
        Object containing regimes, model, scaler, etc.
    """
    if GaussianHMM is None:
        raise ImportError(
            "hmmlearn is not installed. Install it with:\n"
            "    pip install hmmlearn"
        )

    if cfg.scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X

    # Create and fit HMM
    model = GaussianHMM(
        n_components=cfg.n_regimes,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
    )

    model.fit(X_scaled)

    # Most likely state sequence
    regimes = model.predict(X_scaled)

    regimes_series = pd.Series(
        data=regimes,
        index=index,
        name="regime",
    )

    return RegimeFitResult(
        regimes=regimes_series,
        model=model,
        scaler=scaler,
        X=X_scaled,
        index=index,
        metadata={},
    )


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def fit_regimes_from_features(
    feat: pd.DataFrame,
    cfg: Optional[RegimeHMMConfig] = None,
    drop_na: bool = True,
) -> RegimeFitResult:
    """
    End-to-end pipeline:
    - Prepare feature matrix from a feature panel.
    - Fit Gaussian HMM.
    - Return regimes + model + scaler.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature panel (from core.features).
    cfg : RegimeHMMConfig or None
        Configuration. If None, use default settings.
    drop_na : bool
        Whether to drop rows with any NaNs before fitting.

    Returns
    -------
    RegimeFitResult
    """
    if cfg is None:
        cfg = RegimeHMMConfig()

    X, index, metadata = prepare_feature_matrix(feat, drop_na=drop_na)
    result = fit_hmm_regimes(X, index, cfg)
    result.metadata.update(metadata)
    return result


def fit_regimes_from_config(
    cfg: Optional[RegimeHMMConfig] = None,
    return_method: str = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegimeFitResult]:
    """
    High-level convenience pipeline:

    - Read tickers/dates from config/data_config.json.
    - Load prices + returns.
    - Build basic features.
    - Fit Gaussian HMM to those features.
    - Return prices, returns, features, and regime result.

    Parameters
    ----------
    cfg : RegimeHMMConfig or None
        Configuration for the HMM. If None, use defaults.
    return_method : {"simple", "log"}
        How to compute returns from prices.
    force_refresh : bool
        Force re-download of prices.
    cache_key : str or None
        Optional explicit cache key for price parquet.

    Returns
    -------
    (px, rets, feat, regime_result)
        px            : price panel (wide, adjusted close)
        rets          : return panel
        feat          : feature panel (MultiIndex or flat columns)
        regime_result : RegimeFitResult (regimes, model, scaler, etc.)
    """
    if cfg is None:
        cfg = RegimeHMMConfig()

    # 1) Build basic features from config
    px, rets, feat = features.build_basic_features_from_config(
        return_method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )

    # 2) Fit regimes on those features
    regime_result = fit_regimes_from_features(feat, cfg=cfg, drop_na=True)

    return px, rets, feat, regime_result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/regimes.py
    #
    # It will:
    #   - load prices + returns + features from config
    #   - fit a 3-regime HMM
    #   - print a small summary
    cfg = RegimeHMMConfig(
        n_regimes=3,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        verbose=False,
        scale_features=True,
    )

    try:
        px, rets, feat, res = fit_regimes_from_config(cfg=cfg)
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
    except ImportError as e:
        print(e)
    else:
        print("\n=== Regime model summary ===")
        print(f"Price panel:    {px.shape[0]} rows × {px.shape[1]} tickers")
        print(f"Return panel:   {rets.shape[0]} rows × {rets.shape[1]} tickers")
        print(f"Feature panel:  {feat.shape[0]} rows × {feat.shape[1]} cols")
        print(f"Regime series:  {len(res.regimes)} dates")
        print(f"Regimes found:  {sorted(res.regimes.unique().tolist())}")
        print(f"Date span:      {res.index.min().date()} -> {res.index.max().date()}")
