# --- path shim: allow running this file from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------

import pandas as pd
from core import config
from core.data import load_prices
from core.utils import log
from core.pairs import rank_candidate_pairs
from core.kalman import kalman_spread


def main():
    # 1. Load price history
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
    )

    # 2. Get candidate pairs (reuse same criteria from prep flow)
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
        log("No passing pairs to inspect.")
        return

    # pick the first passing pair
    row0 = cand.iloc[0]
    A = row0["A"]
    B = row0["B"]
    log(f"Inspecting top pair: {A} vs {B}")

    # 3. Build dynamic spread with Kalman
    ks = kalman_spread(
        prices[A],
        prices[B],
        roll_window=60,
        delta=1e-5,
        R=1e-3,
    )

    # 4. Basic sanity info
    # Current z_resid (signal)
    curr_z = ks["z_resid"].iloc[-1]
    curr_beta = ks["beta"].iloc[-1]
    curr_resid = ks["resid"].iloc[-1]

    log("=== Snapshot ===")
    log(f"Latest beta (A vs B): {curr_beta:.4f}")
    log(f"Latest residual z-score: {curr_z:.2f}")
    log(f"Latest raw residual: {curr_resid:.4f}")

    # 5. Save full time series for plotting / analysis
    out_path = config.DATA_DIR / f"kalman_{A}_{B}.csv"
    ks.to_csv(out_path)
    log(f"Saved Kalman time series → {out_path}")

    # 6. Show tail in console
    with pd.option_context("display.width", 140, "display.max_columns", 10):
        print(ks.tail(10))


if __name__ == "__main__":
    main()
