# --- path shim: run this file from anywhere, no packaging needed ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------

import pandas as pd
from core import config
from core.data import load_prices
from core.utils import log, pct_change
from core.pairs import rank_candidate_pairs, filter_by_liquidity

def main():
    # 1) Data
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
    )

    # 2) Liquidity proxy
    rets = pct_change(prices).fillna(0.0)
    activity20 = rets.abs().rolling(20).sum().iloc[-1].dropna()

    # 3) Candidate ranking
    log("Ranking candidate pairs …")
    cand = rank_candidate_pairs(
        prices,
        windows=config.WINDOWS,
        max_pairs=config.MAX_PAIRS,
        min_half_life_days=config.MIN_HALF_LIFE_D,
        max_half_life_days=config.MAX_HALF_LIFE_D,
    )
    if cand.empty:
        log("No candidates found — expand universe or relax thresholds.")
        return

    # 4) Liquidity filter
    cand_liq = filter_by_liquidity(cand, activity20, drop_same_company=True)

    out_path = config.DATA_DIR / "pairs_candidates.csv"
    cand_liq.to_csv(out_path, index=False)
    log(f"Saved {len(cand_liq)} candidate pairs → {out_path}")

if __name__ == "__main__":
    main()
