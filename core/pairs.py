from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
from .utils import zscore

# ----- internal: align, clean, ensure positivity -----

def _prep_xy(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    s1 = pd.Series(x, dtype="float64")
    s2 = pd.Series(y, dtype="float64")
    idx = s1.index.intersection(s2.index)
    if len(idx) == 0:
        return s1.iloc[0:0], s2.iloc[0:0]
    df = pd.DataFrame({"x": s1.loc[idx], "y": s2.loc[idx]})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["x"] > 0) & (df["y"] > 0)]  # logs require positive
    if df.empty:
        return df["x"], df["y"]
    return df["x"], df["y"]

# ----- stats primitives -----

def eg_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xx, yy = _prep_xy(x, y)
    if len(xx) < 250:
        return np.nan, np.nan
    lx, ly = np.log(xx.values), np.log(yy.values)
    score, pval, _ = coint(lx, ly)
    return float(score), float(pval)

def hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    xx, yy = _prep_xy(x, y)
    if len(xx) < 2:
        return np.nan
    lx = np.log(xx.values).reshape(-1, 1)
    ly = np.log(yy.values)
    lr = LinearRegression().fit(lx, ly)
    return float(lr.coef_[0])

def ou_half_life(spread: pd.Series, min_len: int = 126) -> float:
    """
    Estimate OU half-life via AR(1) on spread:
      ΔS_t = a + b * S_{t-1} + ε_t,  rho = 1 + b
      half-life = ln(2) / -ln(rho)
    """
    s = pd.Series(spread, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < min_len:
        return np.inf
    y = s.diff().iloc[1:].values.reshape(-1, 1)
    x = s.shift(1).iloc[1:].values.reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    b = float(lr.coef_[0])
    rho = float(np.clip(1.0 + b, -0.9999, 0.9999))
    if rho <= 0:
        rho = abs(rho)
    return float(np.log(2) / -np.log(rho))

def build_spread(x: pd.Series, y: pd.Series) -> tuple[pd.Series, float, float]:
    xx, yy = _prep_xy(x, y)
    if len(xx) < 2:
        return pd.Series(dtype="float64"), np.nan, np.nan
    beta = hedge_ratio(xx, yy)
    if not np.isfinite(beta):
        return pd.Series(dtype="float64"), np.nan, np.nan
    lx, ly = np.log(xx), np.log(yy)
    spr = lx - beta * ly
    z = zscore(spr.values)
    z_last = float(z[-1]) if len(z) else np.nan
    return spr, float(beta), z_last

def rolling_eg_pvalues(x: pd.Series, y: pd.Series, windows=(504, 252, 126)) -> list[float]:
    xx, yy = _prep_xy(x, y)  # clean first
    out = []
    for w in windows:
        if len(xx) >= w and len(yy) >= w:
            _, p = eg_pvalue(xx.iloc[-w:], yy.iloc[-w:])
            out.append(p)
        else:
            out.append(np.nan)
    return out

# ----- diagnostics -----

def explain_pair(
    x: pd.Series,
    y: pd.Series,
    windows=(504, 252, 126),
    min_overlap_days: int = 252 * 2,
    hl_lookback_days: int | None = 252,
    min_half_life_days: int = 1,
    max_half_life_days: int = 250,
    pval_threshold: float | None = None,
) -> dict:
    """
    Compute metrics and return a dict with 'reason' if the pair fails filtering.
    Reasons: overlap_lt_min, pvals_all_nan, pval_gt_thresh, spread_invalid,
             hl_nan, hl_lt_min, hl_gt_max, pass
    """
    xx, yy = _prep_xy(x, y)
    n_obs = int(len(xx))
    if n_obs < min_overlap_days:
        return {"n_obs": n_obs, "reason": "overlap_lt_min"}

    pvals = rolling_eg_pvalues(xx, yy, windows=windows)
    if np.all(np.isnan(pvals)):
        return {"n_obs": n_obs, "reason": "pvals_all_nan"}

    med_p = float(np.nanmedian(pvals))
    if pval_threshold is not None and (not np.isnan(med_p)) and med_p > pval_threshold:
        return {
            "n_obs": n_obs, "pval_med": med_p, "reason": "pval_gt_thresh",
            "pval_3y": pvals[0] if len(pvals) > 0 else np.nan,
            "pval_2y": pvals[1] if len(pvals) > 1 else np.nan,
            "pval_1y": pvals[2] if len(pvals) > 2 else np.nan
        }

    spr, beta, z_last = build_spread(xx, yy)
    if (not np.isfinite(beta)) or spr.empty:
        return {"n_obs": n_obs, "reason": "spread_invalid", "pval_med": med_p}

    # Half-life on recent window
    spr_tail = spr.iloc[-hl_lookback_days:] if (hl_lookback_days is not None and len(spr) > hl_lookback_days) else spr
    hl = ou_half_life(spr_tail, min_len=min(hl_lookback_days or 126, 252))

    if not np.isfinite(hl):
        return {"n_obs": n_obs, "reason": "hl_nan", "pval_med": med_p}
    if hl < min_half_life_days:
        return {"n_obs": n_obs, "reason": "hl_lt_min", "pval_med": med_p, "half_life_d": float(hl)}
    if hl > max_half_life_days:
        return {
            "n_obs": n_obs, "reason": "hl_gt_max", "pval_med": med_p,
            "half_life_d": float(hl),
            "pval_3y": pvals[0] if len(pvals) > 0 else np.nan,
            "pval_2y": pvals[1] if len(pvals) > 1 else np.nan,
            "pval_1y": pvals[2] if len(pvals) > 2 else np.nan
        }

    return {
        "n_obs": n_obs, "reason": "pass",
        "pval_3y": pvals[0] if len(pvals) > 0 else np.nan,
        "pval_2y": pvals[1] if len(pvals) > 1 else np.nan,
        "pval_1y": pvals[2] if len(pvals) > 2 else np.nan,
        "pval_med": med_p, "beta_AB": float(beta), "half_life_d": float(hl), "z_last": float(z_last)
    }

def diagnose_universe(
    prices_df: pd.DataFrame,
    windows=(504, 252, 126),
    min_overlap_days: int = 252 * 2,
    hl_lookback_days: int | None = 252,
    min_half_life_days: int = 1,
    max_half_life_days: int = 250,
    pval_threshold: float | None = None,
) -> pd.DataFrame:
    rows = []
    cols = list(prices_df.columns)
    for a, b in itertools.combinations(cols, 2):
        x, y = prices_df[a], prices_df[b]
        d = explain_pair(
            x, y,
            windows=windows,
            min_overlap_days=min_overlap_days,
            hl_lookback_days=hl_lookback_days,
            min_half_life_days=min_half_life_days,
            max_half_life_days=max_half_life_days,
            pval_threshold=pval_threshold
        )
        d.update({"A": a, "B": b})
        rows.append(d)
    df = pd.DataFrame(rows)
    cols_order = ["A","B","reason","n_obs","pval_3y","pval_2y","pval_1y","pval_med","beta_AB","half_life_d","z_last"]
    return df[[c for c in cols_order if c in df.columns]].sort_values(["reason","pval_med"], ascending=[True, True], na_position="last")

# ----- ranking (uses same criteria) -----

def rank_candidate_pairs(
    prices_df: pd.DataFrame,
    windows=(504, 252, 126),
    max_pairs: int = 80,
    min_overlap_days: int = 252 * 2,
    hl_lookback_days: int | None = 252,
    min_half_life_days: int = 1,
    max_half_life_days: int = 250,
    pval_threshold: float | None = None,
) -> pd.DataFrame:
    diag = diagnose_universe(
        prices_df,
        windows=windows,
        min_overlap_days=min_overlap_days,
        hl_lookback_days=hl_lookback_days,
        min_half_life_days=min_half_life_days,
        max_half_life_days=max_half_life_days,
        pval_threshold=pval_threshold
    )
    passed = diag.loc[diag["reason"] == "pass"].copy()
    if passed.empty:
        return passed
    passed = passed.sort_values(["pval_med","half_life_d"], ascending=[True, True]).reset_index(drop=True)
    return passed.head(max_pairs)

# ----- liquidity helper (restored) -----

def filter_by_liquidity(candidates: pd.DataFrame, activity_20d: pd.Series, drop_same_company: bool = True) -> pd.DataFrame:
    """
    Keep pairs where both legs are above 25th percentile of activity_20d.
    Optionally drop trivial same-company pairs (GOOG/GOOGL).
    """
    if candidates.empty:
        return candidates
    liq_cut = activity_20d.quantile(0.25)
    liquid = set(activity_20d.loc[activity_20d >= liq_cut].index)
    out = candidates[candidates["A"].isin(liquid) & candidates["B"].isin(liquid)].copy()

    if drop_same_company:
        mask = out.apply(lambda r: {r["A"], r["B"]} == {"GOOG", "GOOGL"}, axis=1)
        out = out.loc[~mask].reset_index(drop=True)
    return out
