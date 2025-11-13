# --- path shim: run from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------

import argparse
import pandas as pd
from core import config
from core.data import load_prices
from core.kalman import kalman_spread
from core.pairs import rank_candidate_pairs
from core.strategy import StrategyParams
from core.backtest import backtest_pair
from core.utils import log

def parse_args():
    ap = argparse.ArgumentParser(description="Backtest a Kalman Z-score pairs strategy")
    ap.add_argument("--A", type=str, help="Ticker A (optional)")
    ap.add_argument("--B", type=str, help="Ticker B (optional)")
    ap.add_argument("--entry_z", type=float, default=2.0)
    ap.add_argument("--exit_z", type=float, default=0.5)
    ap.add_argument("--max_gross", type=float, default=50_000.0)
    ap.add_argument("--cvar_limit", type=float, default=1_000.0)
    ap.add_argument("--risk_window", type=int, default=60)
    ap.add_argument("--z_cap", type=float, default=3.5)
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Data
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
    )

    # 2) Choose pair
    if args.A and args.B:
        A, B = args.A.upper(), args.B.upper()
        if A not in prices.columns or B not in prices.columns:
            log("Supplied tickers not in price data/universe.")
            return
        log(f"Using supplied pair: {A} vs {B}")
    else:
        cand = rank_candidate_pairs(
            prices_df=prices,
            windows=config.WINDOWS,
            max_pairs=config.MAX_PAIRS,
            min_overlap_days=config.MIN_OVERLAP_DAYS,
            hl_lookback_days=config.HL_LOOKBACK_DAYS,
            min_half_life_days=config.MIN_HALF_LIFE_D,
            max_half_life_days=config.MAX_HALF_LIFE_D,
            pval_threshold=config.PVAL_THRESHOLD,
        )
        if cand.empty:
            log("No candidate pairs found.")
            return
        A, B = cand.iloc[0]["A"], cand.iloc[0]["B"]
        log(f"No pair provided; using top candidate: {A} vs {B}")

    # 3) Build Kalman spread
    ks = kalman_spread(prices[A], prices[B], roll_window=60, delta=1e-5, R=1e-3)
    kalman_path = config.DATA_DIR / f"kalman_{A}_{B}.csv"
    ks.to_csv(kalman_path)
    log(f"Saved Kalman series → {kalman_path}")

    # 4) Strategy params
    params = StrategyParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_gross_per_pair=args.max_gross,
        cvar95_limit_per_pair=args.cvar_limit,
        risk_window=args.risk_window,
        z_cap=args.z_cap,
    )

    # 5) Backtest
    trades_df, equity_df = backtest_pair(ks, params)

    # 6) Save & print summary
    trades_path = config.DATA_DIR / f"bt_{A}_{B}_trades.csv"
    equity_path = config.DATA_DIR / f"bt_{A}_{B}_equity.csv"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)

    log(f"Saved trades → {trades_path}")
    log(f"Saved equity → {equity_path}")

    # Console summary
    if not trades_df.empty:
        total_trades = len(trades_df)
        pnl_sum = trades_df["pnl_dollars"].sum()
        win_rate = (trades_df["pnl_dollars"] > 0).mean()
        avg_days = trades_df["days_held"].mean()

        log("=== Backtest summary ===")
        log(f"Pair: {A}/{B}")
        log(f"Trades: {total_trades}, Win rate: {win_rate:.1%}, "
            f"Avg hold: {avg_days:.1f}d, PnL: ${pnl_sum:,.0f}")
    else:
        log("No trades executed with current parameters.")

if __name__ == "__main__":
    main()
