"""
core.walkforward

Formation-window -> trading-window walk-forward loop for pair selection and
backtesting.

Problem this solves:
- core.pair_selection.find_candidate_pairs and core.strategy.backtest_spread_strategy
  are normally handed the *entire* price panel at once. A pair's cointegration
  p-value, its hedge ratio, and its backtest score all end up partly determined
  by data from *after* the dates being traded (look-ahead bias).

This module does not reimplement cointegration testing, hedge estimation, or
backtesting: it orchestrates the existing functions in core.pair_selection,
core.pairs, and core.strategy over a sequence of non-overlapping trading
windows, where each trading window's pair selection and hedge ratio come only
from the formation window strictly before it.

Typical usage:

    from core import panel, pair_selection, strategy, walkforward

    px, _ = panel.load_prices_and_returns_from_config()

    sel_cfg = pair_selection.PairSelectionConfig(max_pvalue=0.05, min_obs=252)
    bt_cfg = strategy.BacktestConfig(initial_capital=100_000)
    wf_cfg = walkforward.WalkForwardConfig()

    result = walkforward.run_walk_forward(px, sel_cfg, bt_cfg, wf_cfg)
    print(result.portfolio_summary)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple

import pandas as pd

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None  # type: ignore

# Local imports (repo-style)
try:
    import core.paths as paths
    import core.panel as panel
    import core.pair_selection as pair_selection
    import core.pairs as pairs_mod
    import core.strategy as strategy
except ModuleNotFoundError:
    # Fallback if run from inside core/ directory
    import paths  # type: ignore
    import panel  # type: ignore
    import pair_selection  # type: ignore
    import pairs as pairs_mod  # type: ignore
    import strategy  # type: ignore


def log(msg: str) -> None:
    """Simple logging helper."""
    print(f"[walkforward] {msg}")


# ---------------------------------------------------------------------------
# Config & fold dataclasses
# ---------------------------------------------------------------------------

WalkForwardMode = Literal["rolling", "expanding"]


@dataclass
class WalkForwardConfig:
    """
    Configuration for the formation/trading walk-forward loop.

    Attributes
    ----------
    formation_years : float
        Length of the formation (pair-selection) window, in years.
    trading_years : float
        Length of the trading (out-of-sample) window, in years.
    step_years : float
        How far to advance between folds, in years. Defaults to trading_years
        so trading windows are back-to-back and non-overlapping.
    mode : {"rolling", "expanding"}
        "rolling": the formation window slides forward each fold, keeping a
            fixed length (formation_years).
        "expanding": the formation window always starts at the beginning of
            the data and grows each fold.
    min_formation_obs : int
        Minimum number of formation-window observations required to run a
        fold at all (guards against a too-short first window).
    fdr_alpha : float
        Target false discovery rate for the Benjamini-Hochberg correction
        applied to formation-window cointegration p-values.
    min_stable_folds : int
        A pair must have passed the FDR-corrected cointegration test in at
        least this many of the last `stability_lookback` folds (including the
        current one) to be traded in the current fold.
    stability_lookback : int
        Number of most recent folds (including the current one) considered
        when evaluating pair stability.
    zscore_window : int
        Rolling window (in trading days) for the spread z-score, reused from
        core.pairs. Also used as the warmup buffer pulled from the formation
        window so the trading window's rolling z-score isn't NaN at its start.
    """

    formation_years: float = 3.0
    trading_years: float = 1.0
    step_years: Optional[float] = None
    mode: WalkForwardMode = "rolling"
    min_formation_obs: int = 252
    fdr_alpha: float = 0.10
    min_stable_folds: int = 2
    stability_lookback: int = 3
    zscore_window: int = 60

    def __post_init__(self) -> None:
        if self.step_years is None:
            self.step_years = self.trading_years


@dataclass
class Fold:
    """One formation/trading window pair."""

    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def _years_to_offset(years: float) -> pd.DateOffset:
    return pd.DateOffset(days=round(years * 365.25))


def generate_folds(
    px_index: pd.DatetimeIndex,
    cfg: WalkForwardConfig,
) -> List[Fold]:
    """
    Generate a sequence of non-overlapping formation/trading folds spanning
    a price panel's DatetimeIndex.

    Parameters
    ----------
    px_index : pd.DatetimeIndex
        The dates available in the price panel (sorted).
    cfg : WalkForwardConfig

    Returns
    -------
    list of Fold
        Trading windows are back-to-back when step_years == trading_years
        (the default). Formation windows always end exactly where the
        trading window begins (no gap, no overlap). Folds whose formation
        window has fewer than cfg.min_formation_obs observations, or whose
        trading window would run past the end of the data, are omitted.
    """
    if len(px_index) == 0:
        return []

    idx = px_index.sort_values()
    data_start = idx[0]
    data_end = idx[-1]

    formation_len = _years_to_offset(cfg.formation_years)
    trading_len = _years_to_offset(cfg.trading_years)
    step_len = _years_to_offset(cfg.step_years)

    folds: List[Fold] = []

    if cfg.mode == "rolling":
        formation_start = data_start
    else:  # expanding
        formation_start = data_start

    formation_end = data_start + formation_len

    while True:
        trading_start = formation_end
        if trading_start > data_end:
            break

        trading_end = trading_start + trading_len
        # Clip the trading window to the available data (last fold may be
        # shorter than trading_years). All windows are half-open [start, end),
        # so when clipping to the last available date we push the bound one
        # day past it, otherwise the final trading day would be excluded by
        # a strict `< trading_end` downstream.
        effective_trading_end = min(trading_end, data_end + pd.Timedelta(days=1))

        if trading_start >= effective_trading_end:
            # No trading days left to run this fold over.
            break

        n_formation_obs = int(
            ((idx >= formation_start) & (idx < formation_end)).sum()
        )
        if n_formation_obs >= cfg.min_formation_obs:
            folds.append(
                Fold(
                    formation_start=formation_start,
                    formation_end=formation_end,
                    trading_start=trading_start,
                    trading_end=effective_trading_end,
                )
            )

        if trading_end >= data_end:
            break

        if cfg.mode == "rolling":
            formation_start = formation_start + step_len
            formation_end = formation_end + step_len
        else:  # expanding
            formation_end = formation_end + step_len

    return folds


# ---------------------------------------------------------------------------
# Per-fold pair selection: FDR correction + on-disk cache
# ---------------------------------------------------------------------------

def _fold_cache_key(
    universe: List[str],
    fold: Fold,
    sel_cfg: pair_selection.PairSelectionConfig,
    wf_cfg: WalkForwardConfig,
) -> str:
    """Deterministic cache key for one fold's candidate-pair scan."""
    payload = "|".join(
        [
            ",".join(sorted(str(t).upper() for t in universe)),
            fold.formation_start.isoformat(),
            fold.formation_end.isoformat(),
            f"max_pvalue={sel_cfg.max_pvalue}",
            f"min_obs={sel_cfg.min_obs}",
            f"use_log_prices={sel_cfg.use_log_prices}",
            f"add_intercept={sel_cfg.add_intercept}",
            f"max_pairs={sel_cfg.max_pairs}",
            f"fdr_alpha={wf_cfg.fdr_alpha}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fold_cache_path(cache_key: str) -> Path:
    out_dir = paths.cache_dir() / "pairs" / "walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{cache_key}.csv"


def select_pairs_for_fold(
    px: pd.DataFrame,
    sel_cfg: pair_selection.PairSelectionConfig,
    wf_cfg: WalkForwardConfig,
    fold: Fold,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Run cointegration-based pair selection over one fold's formation window
    and apply multiple-testing (FDR) control on top of it.

    Note on the FDR correction: `pair_selection.find_candidate_pairs` already
    discards any pair with pvalue > sel_cfg.max_pvalue before returning (and
    further truncates to the top `sel_cfg.max_pairs` by p-value). The
    Benjamini-Hochberg correction here is therefore applied to that
    already-thresholded candidate set, not to the full universe of pairwise
    tests run internally — a practical approximation, not a textbook-exact
    family-wise correction. It still meaningfully tightens the set: pairs
    that only barely cleared max_pvalue purely by chance among many trials
    get pushed out by a higher rank-adjusted q-value.

    Results are cached to disk per fold (keyed on universe + fold dates +
    sel_cfg + fdr_alpha) under cache/pairs/walkforward/, since scanning
    ~185 tickers is CPU-intensive and walk-forward reruns this once per fold.

    Parameters
    ----------
    px : pd.DataFrame
        Full price panel (this function slices it to the fold's formation
        window itself).
    sel_cfg : pair_selection.PairSelectionConfig
    wf_cfg : WalkForwardConfig
    fold : Fold
    force_refresh : bool
        If True, ignore any cached result and recompute.

    Returns
    -------
    pd.DataFrame
        Same columns as `find_candidate_pairs`, plus:
            - qvalue     : Benjamini-Hochberg-adjusted p-value
            - fdr_reject : bool, True if the pair survives FDR control
    """
    universe = list(px.columns.astype(str))
    cache_key = _fold_cache_key(universe, fold, sel_cfg, wf_cfg)
    cache_path = _fold_cache_path(cache_key)

    if not force_refresh and cache_path.exists():
        return pd.read_csv(cache_path)

    formation_slice = px.loc[fold.formation_start : fold.formation_end]
    # Formation window is [start, end); loc is inclusive on both ends, so
    # drop a trailing row that lands exactly on formation_end if present.
    formation_slice = formation_slice.loc[formation_slice.index < fold.formation_end]

    inner_cfg = replace(sel_cfg, save_to_csv=False)
    df = pair_selection.find_candidate_pairs(formation_slice, inner_cfg)

    if df.empty:
        df = df.assign(qvalue=pd.Series(dtype=float), fdr_reject=pd.Series(dtype=bool))
    else:
        if multipletests is None:
            raise ImportError(
                "statsmodels is not installed. Install it with:\n"
                "    pip install statsmodels"
            )
        reject, qvalues, _, _ = multipletests(
            df["pvalue"].to_numpy(), alpha=wf_cfg.fdr_alpha, method="fdr_bh"
        )
        df = df.assign(qvalue=qvalues, fdr_reject=reject)

    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Pair stability across folds
# ---------------------------------------------------------------------------

def apply_stability_filter(
    pair_history: List[FrozenSet[Tuple[str, str]]],
    wf_cfg: WalkForwardConfig,
) -> Set[Tuple[str, str]]:
    """
    Decide which pairs are "stable" enough to trade in the current fold.

    Parameters
    ----------
    pair_history : list of frozenset of (x, y)
        Chronological list of the sets of pairs that passed FDR control in
        each fold evaluated so far, with the **current fold's set as the
        last element**.
    wf_cfg : WalkForwardConfig
        Uses `stability_lookback` (how many recent folds, including the
        current one, to consider) and `min_stable_folds` (how many of those
        a pair must appear in).

    Returns
    -------
    set of (x, y)
        The subset of the current fold's FDR-surviving pairs that are
        eligible to trade this fold. Note that with the default
        min_stable_folds=2, the very first fold(s) evaluated will always
        return an empty set — there isn't enough history yet to call
        anything "stable". This is intentional: a pair earns the right to
        trade only after showing up as cointegrated more than once.
    """
    if not pair_history:
        return set()

    current = pair_history[-1]
    recent = pair_history[-wf_cfg.stability_lookback :]

    counts: Dict[Tuple[str, str], int] = {}
    for fold_pairs in recent:
        for pair in fold_pairs:
            counts[pair] = counts.get(pair, 0) + 1

    return {pair for pair in current if counts.get(pair, 0) >= wf_cfg.min_stable_folds}


# ---------------------------------------------------------------------------
# Frozen-beta pair signals for one fold's trading window
# ---------------------------------------------------------------------------

def build_fold_signals(
    px: pd.DataFrame,
    x: str,
    y: str,
    alpha: float,
    beta: float,
    hedge_cfg: pairs_mod.HedgeConfig,
    signal_cfg: pairs_mod.SignalConfig,
    fold: Fold,
    wf_cfg: WalkForwardConfig,
) -> pairs_mod.PairPipelineResult:
    """
    Build spread/z-score/signals for one pair over one fold's trading window,
    using a hedge ratio (alpha, beta) that was already estimated from the
    formation window — never re-estimated here.

    This deliberately does NOT call pairs_mod.build_pair_signals_from_panel,
    because that function re-estimates alpha/beta from whatever panel slice
    it's given, which would leak trading-window information back into the
    hedge ratio. Instead it calls pairs_mod.compute_spread /
    compute_spread_zscore / generate_pair_signals directly with the frozen
    (alpha, beta).

    A warmup buffer of `wf_cfg.zscore_window` trading days immediately before
    the trading window (still strictly inside the formation period) is
    included so the rolling z-score isn't NaN at the start of the trading
    window; those warmup rows are dropped from the returned series.

    Returns
    -------
    pairs_mod.PairPipelineResult
        spread/zscore/signals cover only `[fold.trading_start, fold.trading_end)`.
    """
    full_idx = px.index
    pos = int(full_idx.searchsorted(fold.trading_start))
    warmup_start_pos = max(0, pos - wf_cfg.zscore_window)
    warmup_start = full_idx[warmup_start_pos]

    px_slice = px.loc[
        (px.index >= warmup_start) & (px.index < fold.trading_end)
    ]

    spec = pairs_mod.PairSpec(x=x, y=y)
    df_pair = pairs_mod.extract_pair_prices(px_slice, spec)

    spread = pairs_mod.compute_spread(df_pair, alpha, beta, hedge_cfg)
    zscore = pairs_mod.compute_spread_zscore(spread, window=wf_cfg.zscore_window)
    signals = pairs_mod.generate_pair_signals(zscore, signal_cfg)

    trading_mask = (spread.index >= fold.trading_start) & (spread.index < fold.trading_end)

    metadata = {
        "alpha": alpha,
        "beta": beta,
        "zscore_window": wf_cfg.zscore_window,
        "fold": fold,
    }

    return pairs_mod.PairPipelineResult(
        spec=spec,
        hedge_config=hedge_cfg,
        signal_config=signal_cfg,
        hedge_ratio=beta,
        spread=spread.loc[trading_mask],
        zscore=zscore.loc[trading_mask],
        signals=signals.loc[trading_mask],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Per-fold orchestration
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Everything produced by running one fold end-to-end."""

    fold: Fold
    candidates: pd.DataFrame
    fdr_pairs: FrozenSet[Tuple[str, str]]
    stable_pairs: Set[Tuple[str, str]]
    pair_results: Dict[Tuple[str, str], "pairs_mod.PairPipelineResult"]
    bt_results: Dict[Tuple[str, str], "strategy.BacktestResult"]


def run_fold(
    px: pd.DataFrame,
    fold: Fold,
    sel_cfg: pair_selection.PairSelectionConfig,
    backtest_cfg: strategy.BacktestConfig,
    wf_cfg: WalkForwardConfig,
    historical_fdr_pairs: List[FrozenSet[Tuple[str, str]]],
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    force_refresh: bool = False,
) -> FoldResult:
    """
    Run pair selection, the stability filter, and per-pair backtesting for
    one fold.

    Parameters
    ----------
    px : pd.DataFrame
        Full price panel.
    fold : Fold
    sel_cfg, backtest_cfg, wf_cfg
        Selection, backtest, and walk-forward configuration.
    historical_fdr_pairs : list of frozenset of (x, y)
        FDR-surviving pair sets from all *prior* folds (chronological, not
        including this fold). Used to evaluate stability; this fold's own
        result is returned separately (`FoldResult.fdr_pairs`) so the caller
        can append it before moving to the next fold.
    hedge_cfg, signal_cfg : optional
        Defaults to a HedgeConfig matching sel_cfg's log/intercept settings
        and a default SignalConfig if not provided.
    force_refresh : bool
        Forwarded to select_pairs_for_fold to bypass the per-fold cache.

    Returns
    -------
    FoldResult
    """
    candidates = select_pairs_for_fold(px, sel_cfg, wf_cfg, fold, force_refresh=force_refresh)

    if candidates.empty:
        fdr_pairs: FrozenSet[Tuple[str, str]] = frozenset()
    else:
        fdr_rows = candidates[candidates["fdr_reject"].astype(bool)]
        fdr_pairs = frozenset(zip(fdr_rows["x"], fdr_rows["y"]))

    full_history = list(historical_fdr_pairs) + [fdr_pairs]
    stable_pairs = apply_stability_filter(full_history, wf_cfg)

    if hedge_cfg is None:
        hedge_cfg = pairs_mod.HedgeConfig(
            use_log_prices=sel_cfg.use_log_prices,
            add_intercept=sel_cfg.add_intercept,
        )
    if signal_cfg is None:
        signal_cfg = pairs_mod.SignalConfig()

    pair_results: Dict[Tuple[str, str], pairs_mod.PairPipelineResult] = {}
    bt_results: Dict[Tuple[str, str], strategy.BacktestResult] = {}

    for x, y in stable_pairs:
        row = candidates[(candidates["x"] == x) & (candidates["y"] == y)]
        if row.empty:
            continue
        alpha = float(row.iloc[0]["alpha"])
        beta = float(row.iloc[0]["hedge_ratio"])

        try:
            pair_res = build_fold_signals(
                px, x, y, alpha, beta, hedge_cfg, signal_cfg, fold, wf_cfg
            )
        except Exception as e:
            log(f"  [WARN] Failed to build fold signals for {x}/{y}: {e}")
            continue

        if pair_res.spread.empty:
            continue

        bt_kwargs = {}
        if backtest_cfg.execution is not None:
            leg_prices = pairs_mod.extract_pair_prices(px, pairs_mod.PairSpec(x=x, y=y))
            bt_kwargs = dict(
                leg_prices=leg_prices.reindex(pair_res.spread.index),
                beta=beta,
                log_space=hedge_cfg.use_log_prices,
            )

        bt = strategy.backtest_spread_strategy(
            spread=pair_res.spread,
            signals=pair_res.signals,
            cfg=backtest_cfg,
            **bt_kwargs,
        )

        pair_results[(x, y)] = pair_res
        bt_results[(x, y)] = bt

    return FoldResult(
        fold=fold,
        candidates=candidates,
        fdr_pairs=fdr_pairs,
        stable_pairs=stable_pairs,
        pair_results=pair_results,
        bt_results=bt_results,
    )


# ---------------------------------------------------------------------------
# Full walk-forward loop: chaining + portfolio aggregation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """
    Aggregated output of a full walk-forward run.

    Attributes
    ----------
    fold_summary : pd.DataFrame
        One row per (fold, candidate pair): formation-window selection
        stats, qvalue/fdr_reject, whether it was stable enough to trade,
        and (if traded) that fold's own total_return/sharpe.
    per_pair_equity : dict[(x, y)] -> pd.Series
        Each pair's equity curve, compounded (chained) across every fold it
        traded in. Gaps exist for folds where the pair wasn't traded.
    portfolio_equity : pd.Series
        Combined equity curve across all concurrently-traded pairs, equal-
        notional per fold, chained fold-to-fold. Continuous across the full
        walk-forward span (flat during folds with zero traded pairs).
    portfolio_summary : dict
        Same metric shape as strategy.BacktestResult.summary, computed from
        portfolio_equity.
    diagnostics : pd.DataFrame
        One row per fold: candidate/FDR-survivor/traded counts, useful for
        sanity-checking that FDR + stability filtering are doing something.
    """

    fold_summary: pd.DataFrame
    per_pair_equity: Dict[Tuple[str, str], pd.Series]
    portfolio_equity: pd.Series
    portfolio_summary: Dict[str, float]
    diagnostics: pd.DataFrame
    # Portfolio equity before any transaction/execution costs, for gross-vs-net
    # comparison, and total execution costs by component (commission, spread,
    # slippage, borrow, legacy per-unit cost), each already weighted like the
    # portfolio (1/n_traded per pair).
    portfolio_gross_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cost_summary: Dict[str, float] = field(default_factory=dict)


def _empty_walk_forward_result() -> WalkForwardResult:
    return WalkForwardResult(
        fold_summary=pd.DataFrame(
            columns=[
                "fold_index", "formation_start", "formation_end",
                "trading_start", "trading_end", "x", "y", "pvalue", "qvalue",
                "fdr_reject", "stable", "traded", "fold_total_return", "fold_sharpe",
            ]
        ),
        per_pair_equity={},
        portfolio_equity=pd.Series(dtype=float),
        portfolio_summary={},
        diagnostics=pd.DataFrame(
            columns=[
                "fold_index", "formation_start", "formation_end",
                "trading_start", "trading_end", "n_candidates",
                "n_fdr_reject", "n_stable_traded",
            ]
        ),
    )


def _summarize_equity_curve(
    equity: pd.Series,
    initial_capital: float,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Same metric shape as strategy.BacktestResult.summary, computed standalone
    since portfolio_equity is assembled across many independent fold/pair
    backtests rather than produced by a single backtest_spread_strategy call."""
    if equity.empty:
        return {
            "n_days": 0, "total_pnl": 0.0, "total_return": 0.0,
            "annual_return": 0.0, "annual_vol": 0.0, "sharpe": float("nan"),
            "max_drawdown": 0.0,
        }

    n_days = len(equity)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    daily_returns = equity.pct_change().fillna(0.0)
    vol_daily = float(daily_returns.std(ddof=1)) if n_days > 1 else 0.0

    annual_ret = (1.0 + total_return) ** (annualization_factor / max(n_days, 1)) - 1.0
    annual_vol = vol_daily * (annualization_factor ** 0.5)
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol > 0 else float("nan")

    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min())

    return {
        "n_days": n_days,
        "total_pnl": float(equity.iloc[-1] - initial_capital),
        "total_return": total_return,
        "annual_return": float(annual_ret),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
    }


def run_walk_forward(
    px: pd.DataFrame,
    sel_cfg: pair_selection.PairSelectionConfig,
    backtest_cfg: strategy.BacktestConfig,
    wf_cfg: WalkForwardConfig,
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    force_refresh: bool = False,
) -> WalkForwardResult:
    """
    Run the full formation/trading walk-forward loop over a price panel.

    Generates folds, runs pair selection + FDR + stability filtering +
    backtesting per fold (via run_fold), then:
    - chains each traded pair's equity curve across the folds it traded in
      (compounding onto its last known equity value rather than restarting
      at initial_capital every fold), and
    - combines all pairs traded within each fold into one equal-notional
      portfolio daily-PnL series, also chained fold-to-fold.

    Returns
    -------
    WalkForwardResult
    """
    folds = generate_folds(px.index, wf_cfg)
    if not folds:
        log("No folds could be generated (not enough history for the configured formation window).")
        return _empty_walk_forward_result()

    historical_fdr_pairs: List[FrozenSet[Tuple[str, str]]] = []
    fold_results: List[FoldResult] = []

    for i, fold in enumerate(folds):
        log(
            f"Fold {i + 1}/{len(folds)}: formation "
            f"[{fold.formation_start.date()} -> {fold.formation_end.date()}), "
            f"trading [{fold.trading_start.date()} -> {fold.trading_end.date()})"
        )
        fr = run_fold(
            px, fold, sel_cfg, backtest_cfg, wf_cfg,
            historical_fdr_pairs, hedge_cfg, signal_cfg, force_refresh,
        )
        fold_results.append(fr)
        historical_fdr_pairs.append(fr.fdr_pairs)

    # ---- fold_summary ----
    summary_rows = []
    for i, (fold, fr) in enumerate(zip(folds, fold_results)):
        for row in fr.candidates.itertuples():
            pair = (row.x, row.y)
            traded = pair in fr.bt_results
            bt_summary = fr.bt_results[pair].summary if traded else {}
            summary_rows.append(
                {
                    "fold_index": i,
                    "formation_start": fold.formation_start,
                    "formation_end": fold.formation_end,
                    "trading_start": fold.trading_start,
                    "trading_end": fold.trading_end,
                    "x": row.x,
                    "y": row.y,
                    "pvalue": row.pvalue,
                    "qvalue": row.qvalue,
                    "fdr_reject": bool(row.fdr_reject),
                    "stable": pair in fr.stable_pairs,
                    "traded": traded,
                    "fold_total_return": bt_summary.get("total_return", float("nan")),
                    "fold_sharpe": bt_summary.get("sharpe", float("nan")),
                }
            )
    fold_summary = pd.DataFrame(summary_rows)

    # ---- diagnostics ----
    diagnostics = pd.DataFrame(
        [
            {
                "fold_index": i,
                "formation_start": fold.formation_start,
                "formation_end": fold.formation_end,
                "trading_start": fold.trading_start,
                "trading_end": fold.trading_end,
                "n_candidates": len(fr.candidates),
                "n_fdr_reject": len(fr.fdr_pairs),
                "n_stable_traded": len(fr.bt_results),
            }
            for i, (fold, fr) in enumerate(zip(folds, fold_results))
        ]
    )

    # ---- per-pair chained equity ----
    initial_capital = backtest_cfg.initial_capital
    pair_last_equity: Dict[Tuple[str, str], float] = {}
    pair_equity_chunks: Dict[Tuple[str, str], List[pd.Series]] = {}

    for fr in fold_results:
        for pair, bt in fr.bt_results.items():
            start_capital = pair_last_equity.get(pair, initial_capital)
            factor = bt.equity_curve / initial_capital
            chained = factor * start_capital
            pair_equity_chunks.setdefault(pair, []).append(chained)
            pair_last_equity[pair] = float(chained.iloc[-1])

    per_pair_equity = {
        pair: pd.concat(chunks).sort_index()
        for pair, chunks in pair_equity_chunks.items()
    }

    # ---- portfolio equity: equal-notional across concurrently-traded pairs ----
    net_start = initial_capital
    gross_start = initial_capital
    net_chunks: List[pd.Series] = []
    gross_chunks: List[pd.Series] = []
    cost_totals: Dict[str, float] = {
        "commission": 0.0, "spread": 0.0, "slippage": 0.0, "borrow": 0.0,
        "per_unit_cost": 0.0, "total": 0.0,
    }

    for fold, fr in zip(folds, fold_results):
        fold_dates = px.index[(px.index >= fold.trading_start) & (px.index < fold.trading_end)]
        net_pnl = pd.Series(0.0, index=fold_dates)
        gross_pnl = pd.Series(0.0, index=fold_dates)

        n_traded = len(fr.bt_results)
        if n_traded > 0:
            weight = 1.0 / n_traded
            for bt in fr.bt_results.values():
                net_pnl = net_pnl.add(bt.pnl.reindex(fold_dates).fillna(0.0) * weight, fill_value=0.0)
                gross_pnl = gross_pnl.add(
                    bt.gross_pnl.reindex(fold_dates).fillna(0.0) * weight, fill_value=0.0
                )
                total_cost = float(bt.costs.sum()) * weight
                cost_totals["total"] += total_cost
                if bt.cost_breakdown is not None:
                    for comp in ("commission", "spread", "slippage", "borrow"):
                        cost_totals[comp] += float(bt.cost_breakdown[comp].sum()) * weight
                    cost_totals["per_unit_cost"] += (
                        total_cost - float(bt.cost_breakdown["total"].sum()) * weight
                    )
                else:
                    cost_totals["per_unit_cost"] += total_cost

        net_equity = net_start + net_pnl.cumsum()
        gross_equity = gross_start + gross_pnl.cumsum()
        net_chunks.append(net_equity)
        gross_chunks.append(gross_equity)
        if not net_equity.empty:
            net_start = float(net_equity.iloc[-1])
            gross_start = float(gross_equity.iloc[-1])

    if net_chunks:
        portfolio_equity = pd.concat(net_chunks).sort_index()
        portfolio_gross_equity = pd.concat(gross_chunks).sort_index()
    else:
        portfolio_equity = pd.Series(dtype=float)
        portfolio_gross_equity = pd.Series(dtype=float)

    portfolio_summary = _summarize_equity_curve(portfolio_equity, initial_capital)

    return WalkForwardResult(
        fold_summary=fold_summary,
        per_pair_equity=per_pair_equity,
        portfolio_equity=portfolio_equity,
        portfolio_summary=portfolio_summary,
        diagnostics=diagnostics,
        portfolio_gross_equity=portfolio_gross_equity,
        cost_summary=cost_totals,
    )


def run_walk_forward_from_config(
    sel_cfg: pair_selection.PairSelectionConfig,
    backtest_cfg: strategy.BacktestConfig,
    wf_cfg: WalkForwardConfig,
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    return_method: str = "log",
    force_refresh_prices: bool = False,
    force_refresh_folds: bool = False,
    cache_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, WalkForwardResult]:
    """
    High-level entry point: load prices from config/data_config.json, then
    run the full walk-forward loop. Mirrors the shape of
    core.backtest_pairs.run_universe_backtest_from_config.

    Returns
    -------
    (px, result) : (pd.DataFrame, WalkForwardResult)
    """
    px, _rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh_prices,
        cache_key=cache_key,
    )
    result = run_walk_forward(
        px, sel_cfg, backtest_cfg, wf_cfg,
        hedge_cfg=hedge_cfg, signal_cfg=signal_cfg, force_refresh=force_refresh_folds,
    )
    return px, result
