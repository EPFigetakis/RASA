"""
core.backtest_pairs

Multi-pair backtest harness.

This module ties together:
- core.panel        : loading prices from config
- core.pair_selection : scanning for candidate pairs (cointegration, half-life, etc.)
- core.pairs        : building spread & signals for each pair
- core.strategy     : backtesting each pair's PnL & metrics

Goal:
- Given a universe (from config/data_config.json),
- Find candidate pairs,
- Run a simple spread strategy backtest on each,
- Return a summary table of performance metrics per pair.

Typical usage (notebook):

    from core import backtest_pairs, pair_selection, strategy

    # 1) Configure pair selection
    sel_cfg = pair_selection.PairSelectionConfig(
        max_pvalue=0.05,
        min_obs=252,
        use_log_prices=False,
        add_intercept=False,
        max_pairs=20,
    )

    # 2) Configure backtest
    bt_cfg = strategy.BacktestConfig(
        initial_capital=100_000,
        notional_per_unit=1.0,
        trade_cost_per_unit=0.0,
    )

    # 3) Run universe backtest
    uni_res = backtest_pairs.run_universe_backtest_from_config(
        pair_sel_cfg=sel_cfg,
        backtest_cfg=bt_cfg,
    )

    print(uni_res.summary_table.head())

You can later pass regimes (from core.regimes) to make this regime-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, Any

import numpy as np
import pandas as pd

# Local imports (repo-style)
try:
    import core.panel as panel
    import core.pair_selection as pair_selection
    import core.pairs as pairs_mod
    import core.strategy as strategy
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import panel  # type: ignore
    import pair_selection  # type: ignore
    import pairs as pairs_mod  # type: ignore
    import strategy  # type: ignore


# ---------------------------------------------------------------------------
# Simple logging helper
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[backtest_pairs] {msg}")


# ---------------------------------------------------------------------------
# Dataclass to hold universe-level results
# ---------------------------------------------------------------------------

@dataclass
class UniverseBacktestResult:
    """
    Container for multi-pair backtest outputs.

    Attributes
    ----------
    summary_table : pd.DataFrame
        One row per pair with backtest and selection stats:
            - x, y
            - pvalue, half_life, n_obs, hedge_ratio, alpha
            - total_return, annual_return, sharpe, max_drawdown, n_trades
    pair_results : dict
        Mapping (x, y) -> pairs_mod.PairPipelineResult
        (spread, z-score, signals, etc.).
    bt_results : dict
        Mapping (x, y) -> strategy.BacktestResult
        (equity_curve, pnl, summary, etc.).
    px : pd.DataFrame
        Price panel used for all pairs.
    metadata : dict
        Extra info (configs, notes).
    """
    summary_table: pd.DataFrame
    pair_results: Dict[Tuple[str, str], pairs_mod.PairPipelineResult]
    bt_results: Dict[Tuple[str, str], strategy.BacktestResult]
    px: pd.DataFrame
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def backtest_pairs_for_universe(
    px: pd.DataFrame,
    pairs_df: pd.DataFrame,
    backtest_cfg: strategy.BacktestConfig,
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    zscore_window: int = 60,
    regimes: Optional[pd.Series] = None,
    allowed_regimes: Optional[Sequence[int]] = None,
) -> UniverseBacktestResult:
    """
    Run a spread strategy backtest for each pair listed in `pairs_df`.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price panel (index = date, columns = tickers).
    pairs_df : pd.DataFrame
        Output from core.pair_selection.find_candidate_pairs, with columns:
            - x, y, pvalue, hedge_ratio, alpha, half_life, n_obs
    backtest_cfg : strategy.BacktestConfig
        Configuration for the spread strategy backtest.
    hedge_cfg : pairs_mod.HedgeConfig or None
        Hedge configuration for building the spread. If None, defaults are used.
    signal_cfg : pairs_mod.SignalConfig or None
        Signal configuration for entry/exit rules. If None, defaults are used.
    zscore_window : int
        Rolling window for z-score computation in the pair pipeline.
    regimes : pd.Series, optional
        Regime labels indexed by date (optional, for regime-aware trading).
    allowed_regimes : sequence of int, optional
        Which regimes are allowed to trade. If None, trading is regime-neutral.

    Returns
    -------
    UniverseBacktestResult
    """
    if pairs_df.empty:
        log("No pairs provided; returning empty result.")
        return UniverseBacktestResult(
            summary_table=pd.DataFrame(
                columns=[
                    "x",
                    "y",
                    "pvalue",
                    "half_life",
                    "n_obs",
                    "hedge_ratio_sel",
                    "alpha_sel",
                    "total_return",
                    "annual_return",
                    "sharpe",
                    "max_drawdown",
                    "n_trades",
                ]
            ),
            pair_results={},
            bt_results={},
            px=px,
            metadata={},
        )

    pair_results: Dict[Tuple[str, str], pairs_mod.PairPipelineResult] = {}
    bt_results: Dict[Tuple[str, str], strategy.BacktestResult] = {}
    rows = []

    for idx, row in pairs_df.iterrows():
        x = str(row["x"])
        y = str(row["y"])
        key = (x, y)

        log(f"Backtesting pair {x} / {y} ({idx+1}/{len(pairs_df)}) …")

        spec = pairs_mod.PairSpec(x=x, y=y)

        # Build pair spread & signals
        try:
            pair_res = pairs_mod.build_pair_signals_from_panel(
                px=px,
                spec=spec,
                hedge_cfg=hedge_cfg,
                signal_cfg=signal_cfg,
                zscore_window=zscore_window,
            )
        except Exception as e:
            log(f"  [WARN] Failed to build pair {x}/{y}: {e}")
            continue

        pair_results[key] = pair_res

        # Run backtest
        try:
            bt = strategy.run_pair_strategy_from_pipeline(
                px=px,
                pair_result=pair_res,
                cfg=backtest_cfg,
                regimes=regimes,
                allowed_regimes=allowed_regimes,
            )
        except Exception as e:
            log(f"  [WARN] Failed backtest for {x}/{y}: {e}")
            continue

        bt_results[key] = bt

        # Collect summary row
        sel_pvalue = float(row.get("pvalue", np.nan))
        sel_half_life = float(row.get("half_life", np.nan))
        sel_n_obs = int(row.get("n_obs", np.nan)) if not np.isnan(row.get("n_obs", np.nan)) else np.nan
        sel_hedge_ratio = float(row.get("hedge_ratio", np.nan))
        sel_alpha = float(row.get("alpha", np.nan))

        summary = bt.summary

        rows.append(
            {
                "x": x,
                "y": y,
                "pvalue": sel_pvalue,
                "half_life": sel_half_life,
                "n_obs": sel_n_obs,
                "hedge_ratio_sel": sel_hedge_ratio,
                "alpha_sel": sel_alpha,
                "total_return": summary.get("total_return", np.nan),
                "annual_return": summary.get("annual_return", np.nan),
                "sharpe": summary.get("sharpe", np.nan),
                "max_drawdown": summary.get("max_drawdown", np.nan),
                "n_trades": summary.get("n_trades", np.nan),
            }
        )

    if not rows:
        log("All pairs failed in backtest; returning empty summary.")
        summary_table = pd.DataFrame(
            columns=[
                "x",
                "y",
                "pvalue",
                "half_life",
                "n_obs",
                "hedge_ratio_sel",
                "alpha_sel",
                "total_return",
                "annual_return",
                "sharpe",
                "max_drawdown",
                "n_trades",
            ]
        )
    else:
        summary_table = pd.DataFrame(rows)
        # Sort by Sharpe descending, then total_return
        summary_table = summary_table.sort_values(
            ["sharpe", "annual_return"],
            ascending=[False, False],
        ).reset_index(drop=True)

    metadata: Dict[str, Any] = {
        "backtest_config": backtest_cfg,
        "n_pairs_input": len(pairs_df),
        "n_pairs_success": len(summary_table),
    }

    return UniverseBacktestResult(
        summary_table=summary_table,
        pair_results=pair_results,
        bt_results=bt_results,
        px=px,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Config-based helper: full pipeline from config -> pairs -> backtests
# ---------------------------------------------------------------------------

def run_universe_backtest_from_config(
    pair_sel_cfg: pair_selection.PairSelectionConfig,
    backtest_cfg: strategy.BacktestConfig,
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    zscore_window: int = 60,
    regimes: Optional[pd.Series] = None,
    allowed_regimes: Optional[Sequence[int]] = None,
    return_method: str = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> UniverseBacktestResult:
    """
    End-to-end pipeline:

    - Load prices from config/data_config.json.
    - Run pair selection to find candidate pairs.
    - Backtest each candidate pair.
    - Return a universe-level result object.

    Parameters
    ----------
    pair_sel_cfg : pair_selection.PairSelectionConfig
        Configuration for pair scanning (p-value thresholds, etc.).
    backtest_cfg : strategy.BacktestConfig
        Configuration for the spread strategy backtest.
    hedge_cfg : pairs_mod.HedgeConfig or None
        Hedge configuration for pair signal generation.
    signal_cfg : pairs_mod.SignalConfig or None
        Signal configuration for pair signal generation.
    zscore_window : int
        Z-score window in pair pipeline.
    regimes : pd.Series, optional
        Regime labels indexed by date (for regime-aware trading).
    allowed_regimes : sequence of int, optional
        Regimes in which trading is allowed.
    return_method : {"simple", "log"}
        Return method used when loading returns (not critical for this pipeline).
    force_refresh : bool
        Force fresh download of prices.
    cache_key : str or None
        Explicit cache key for price parquet.

    Returns
    -------
    UniverseBacktestResult
    """
    # 1) Load price panel from config
    px, _rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )

    # 2) Run pair selection
    log("Running pair selection …")
    pairs_df = pair_selection.find_candidate_pairs(px, pair_sel_cfg)
    log(f"Pair selection finished; {len(pairs_df)} candidate pairs kept.")

    # 3) Backtest each candidate pair
    log("Running backtests for candidate pairs …")
    uni_res = backtest_pairs_for_universe(
        px=px,
        pairs_df=pairs_df,
        backtest_cfg=backtest_cfg,
        hedge_cfg=hedge_cfg,
        signal_cfg=signal_cfg,
        zscore_window=zscore_window,
        regimes=regimes,
        allowed_regimes=allowed_regimes,
    )

    return uni_res


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/backtest_pairs.py
    #
    # It will:
    #   - load prices from config/data_config.json
    #   - select candidate pairs
    #   - backtest them
    #   - print the top 10 strategies by Sharpe
    try:
        # 1) Default configs for a quick test
        sel_cfg = pair_selection.PairSelectionConfig(
            max_pvalue=0.05,
            min_obs=252,
            use_log_prices=False,
            add_intercept=False,
            max_pairs=20,
        )

        bt_cfg = strategy.BacktestConfig(
            initial_capital=100_000.0,
            notional_per_unit=1.0,
            trade_cost_per_unit=0.0,
        )

        uni_res = run_universe_backtest_from_config(
            pair_sel_cfg=sel_cfg,
            backtest_cfg=bt_cfg,
        )

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
    except Exception as e:
        print(f"[backtest_pairs] Error in CLI test: {e}")
    else:
        print("\n=== Universe backtest summary ===")
        print("Backtested pairs (top 10 by Sharpe):")
        print(uni_res.summary_table.head(10))
        print("\nMetadata:", uni_res.metadata)
