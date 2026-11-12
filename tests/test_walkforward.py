# tests/test_walkforward.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import walkforward


def _business_day_index(start="2015-01-01", end="2023-01-01"):
    return pd.date_range(start, end, freq="B")


def test_generate_folds_rolling_windows_are_correct_and_contiguous():
    idx = _business_day_index()
    cfg = walkforward.WalkForwardConfig(
        formation_years=3.0,
        trading_years=1.0,
        mode="rolling",
        min_formation_obs=100,
    )

    folds = walkforward.generate_folds(idx, cfg)

    assert len(folds) >= 4

    for fold in folds:
        assert fold.formation_start < fold.formation_end
        assert fold.formation_end == fold.trading_start
        assert fold.trading_start < fold.trading_end
        # Rolling: formation window length stays ~constant.
        assert abs((fold.formation_end - fold.formation_start).days - 3 * 365.25) < 5

    # Trading windows should be back-to-back (no gap, no overlap) since
    # step_years defaults to trading_years.
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.trading_end == nxt.trading_start

    # No fold should claim any trading day beyond the available data (the
    # end bound is half-open, so it may sit one day past the last actual date).
    assert (idx >= folds[-1].trading_end).sum() == 0


def test_generate_folds_expanding_formation_window_grows():
    idx = _business_day_index()
    cfg = walkforward.WalkForwardConfig(
        formation_years=3.0,
        trading_years=1.0,
        mode="expanding",
        min_formation_obs=100,
    )

    folds = walkforward.generate_folds(idx, cfg)
    assert len(folds) >= 4

    # formation_start is fixed; formation_end grows each fold.
    assert all(fold.formation_start == folds[0].formation_start for fold in folds)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.formation_end > prev.formation_end


def test_generate_folds_respects_min_formation_obs():
    idx = _business_day_index("2015-01-01", "2016-06-01")
    cfg = walkforward.WalkForwardConfig(
        formation_years=3.0,
        trading_years=1.0,
        mode="rolling",
        min_formation_obs=252,
    )

    # Only ~1.4 years of data available; a 3-year formation window is never
    # satisfied, so no folds should be produced.
    folds = walkforward.generate_folds(idx, cfg)
    assert folds == []


def test_generate_folds_empty_index_returns_empty_list():
    idx = pd.DatetimeIndex([])
    cfg = walkforward.WalkForwardConfig()
    assert walkforward.generate_folds(idx, cfg) == []


def _cointegrated_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(scale=1.0, size=n))
    noise = rng.normal(scale=0.5, size=n)
    y = 2.0 * x + noise  # Y cointegrated with X
    z = 100 + np.cumsum(rng.normal(scale=1.0, size=n))  # independent random walk

    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=dates)


def test_select_pairs_for_fold_flags_fdr_and_caches(tmp_path, monkeypatch):
    from core import pair_selection

    monkeypatch.setattr(walkforward.paths, "cache_dir", lambda: tmp_path)

    px = _cointegrated_panel(n=300)
    fold = walkforward.Fold(
        formation_start=px.index[0],
        formation_end=px.index[-1] + pd.Timedelta(days=1),
        trading_start=px.index[-1] + pd.Timedelta(days=1),
        trading_end=px.index[-1] + pd.Timedelta(days=30),
    )
    sel_cfg = pair_selection.PairSelectionConfig(max_pvalue=0.2, min_obs=100, max_pairs=10)
    wf_cfg = walkforward.WalkForwardConfig(fdr_alpha=0.10)

    df = walkforward.select_pairs_for_fold(px, sel_cfg, wf_cfg, fold)

    assert "qvalue" in df.columns and "fdr_reject" in df.columns
    assert (df["qvalue"] >= df["pvalue"]).all()

    # The genuinely cointegrated X/Y pair should survive FDR control.
    xy_row = df[(df["x"] == "X") & (df["y"] == "Y")]
    assert not xy_row.empty
    assert bool(xy_row.iloc[0]["fdr_reject"])

    cache_files = list((tmp_path / "pairs" / "walkforward").glob("*.csv"))
    assert len(cache_files) == 1

    # A second call with identical inputs must hit the cache (same file,
    # not recomputed) rather than erroring or duplicating.
    df2 = walkforward.select_pairs_for_fold(px, sel_cfg, wf_cfg, fold)
    pd.testing.assert_frame_equal(df, df2)
    assert len(list((tmp_path / "pairs" / "walkforward").glob("*.csv"))) == 1


def test_apply_stability_filter_requires_repeated_appearance():
    wf_cfg = walkforward.WalkForwardConfig(min_stable_folds=2, stability_lookback=3)

    # Fold 1: only ("A", "B") passes FDR. Not enough history yet -> nothing stable.
    history = [frozenset({("A", "B")})]
    assert walkforward.apply_stability_filter(history, wf_cfg) == set()

    # Fold 2: ("A", "B") passes again, ("C", "D") passes for the first time.
    history.append(frozenset({("A", "B"), ("C", "D")}))
    stable = walkforward.apply_stability_filter(history, wf_cfg)
    assert stable == {("A", "B")}

    # Fold 3: both have now passed at least twice within the lookback window.
    history.append(frozenset({("A", "B"), ("C", "D")}))
    stable = walkforward.apply_stability_filter(history, wf_cfg)
    assert stable == {("A", "B"), ("C", "D")}


def test_apply_stability_filter_empty_history_returns_empty_set():
    wf_cfg = walkforward.WalkForwardConfig()
    assert walkforward.apply_stability_filter([], wf_cfg) == set()


def test_build_fold_signals_uses_frozen_beta_and_trims_to_trading_window():
    from core import pairs as pairs_mod

    px = _cointegrated_panel(n=400)
    wf_cfg = walkforward.WalkForwardConfig(zscore_window=30)
    hedge_cfg = pairs_mod.HedgeConfig(use_log_prices=False, add_intercept=False)
    signal_cfg = pairs_mod.SignalConfig()

    trading_start = px.index[300]
    trading_end = px.index[350]
    fold = walkforward.Fold(
        formation_start=px.index[0],
        formation_end=trading_start,
        trading_start=trading_start,
        trading_end=trading_end,
    )

    fixed_alpha, fixed_beta = 0.0, 2.0
    result = walkforward.build_fold_signals(
        px, "X", "Y", fixed_alpha, fixed_beta, hedge_cfg, signal_cfg, fold, wf_cfg
    )

    # Series are trimmed to exactly the trading window (half-open).
    assert result.spread.index.min() >= trading_start
    assert result.spread.index.max() < trading_end
    assert len(result.spread) == len(px.loc[(px.index >= trading_start) & (px.index < trading_end)])

    # The spread must match a manual calculation from compute_spread with
    # the SAME frozen alpha/beta -- confirming no re-estimation happened.
    df_pair = pairs_mod.extract_pair_prices(px, pairs_mod.PairSpec(x="X", y="Y"))
    manual_spread = pairs_mod.compute_spread(df_pair, fixed_alpha, fixed_beta, hedge_cfg)
    manual_spread = manual_spread.loc[result.spread.index]
    pd.testing.assert_series_equal(result.spread, manual_spread, check_names=False)


def test_run_fold_trades_only_stable_pairs(tmp_path, monkeypatch):
    from core import pair_selection, pairs as pairs_mod, strategy

    monkeypatch.setattr(walkforward.paths, "cache_dir", lambda: tmp_path)

    px = _cointegrated_panel(n=500)
    trading_start = px.index[300]
    trading_end = px.index[400]
    fold = walkforward.Fold(
        formation_start=px.index[0],
        formation_end=trading_start,
        trading_start=trading_start,
        trading_end=trading_end,
    )
    sel_cfg = pair_selection.PairSelectionConfig(max_pvalue=0.2, min_obs=100, max_pairs=10)
    backtest_cfg = strategy.BacktestConfig(initial_capital=100_000.0)
    wf_cfg = walkforward.WalkForwardConfig(min_stable_folds=2, stability_lookback=3, zscore_window=30)

    # No history yet -> nothing should be traded regardless of FDR outcome.
    result_first = walkforward.run_fold(px, fold, sel_cfg, backtest_cfg, wf_cfg, historical_fdr_pairs=[])
    assert result_first.stable_pairs == set()
    assert result_first.bt_results == {}
    assert ("X", "Y") in result_first.fdr_pairs  # still recorded for future stability checks

    # With enough prior history showing X/Y as a repeat FDR survivor, it
    # should now be traded and produce a real backtest result.
    prior_history = [frozenset({("X", "Y")}), frozenset({("X", "Y")})]
    result_second = walkforward.run_fold(
        px, fold, sel_cfg, backtest_cfg, wf_cfg, historical_fdr_pairs=prior_history, force_refresh=True
    )
    assert ("X", "Y") in result_second.stable_pairs
    assert ("X", "Y") in result_second.bt_results
    assert isinstance(result_second.bt_results[("X", "Y")], strategy.BacktestResult)


def test_run_walk_forward_end_to_end_on_synthetic_multi_fold_panel(tmp_path, monkeypatch):
    from core import pair_selection, strategy

    monkeypatch.setattr(walkforward.paths, "cache_dir", lambda: tmp_path)

    # ~6.3 years of daily data -> with 3y formation / 1y trading, expect
    # several folds.
    px = _cointegrated_panel(n=1600)

    sel_cfg = pair_selection.PairSelectionConfig(max_pvalue=0.2, min_obs=100, max_pairs=10)
    backtest_cfg = strategy.BacktestConfig(initial_capital=100_000.0, trade_cost_per_unit=0.0)
    wf_cfg = walkforward.WalkForwardConfig(
        formation_years=3.0,
        trading_years=1.0,
        min_formation_obs=252,
        min_stable_folds=2,
        stability_lookback=3,
        zscore_window=30,
        fdr_alpha=0.10,
    )

    result = walkforward.run_walk_forward(px, sel_cfg, backtest_cfg, wf_cfg)

    assert len(result.fold_summary) > 0
    assert len(result.diagnostics) >= 2

    assert not result.portfolio_equity.empty
    assert not result.portfolio_equity.isna().any()
    assert np.isfinite(result.portfolio_equity.to_numpy()).all()

    # No wild discontinuities at fold boundaries (positions always start
    # flat at the beginning of a fold's own backtest, so day-over-day swings
    # should stay sane even across a fold seam).
    daily_pct_change = result.portfolio_equity.pct_change().dropna()
    assert daily_pct_change.abs().max() < 0.5

    # The genuinely cointegrated X/Y pair should show up as an FDR survivor
    # in most folds; the independent random-walk pairs (X/Z, Y/Z) should not.
    n_folds = result.diagnostics["fold_index"].nunique()
    xy_rows = result.fold_summary[(result.fold_summary["x"] == "X") & (result.fold_summary["y"] == "Y")]
    xy_fdr_hits = int(xy_rows["fdr_reject"].sum())
    assert xy_fdr_hits >= max(1, n_folds - 1)

    other_pairs = result.fold_summary[
        ~((result.fold_summary["x"] == "X") & (result.fold_summary["y"] == "Y"))
    ]
    if not other_pairs.empty:
        assert int(other_pairs["fdr_reject"].sum()) < xy_fdr_hits
