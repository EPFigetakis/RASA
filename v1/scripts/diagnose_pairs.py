# --- path shim to run from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------

from core import config
from core.data import load_prices
from core.pairs import diagnose_universe
from core.utils import log, pct_change
import pandas as pd

def main():
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
    )
    diag = diagnose_universe(
        prices,
        windows=config.WINDOWS,
        min_overlap_days=config.MIN_OVERLAP_DAYS,
        min_half_life_days=config.MIN_HALF_LIFE_D,
        max_half_life_days=config.MAX_HALF_LIFE_D,
        pval_threshold=config.PVAL_THRESHOLD,
    )

    out_path = config.DATA_DIR / "pairs_diagnostics.csv"
    diag.to_csv(out_path, index=False)

    counts = diag["reason"].value_counts(dropna=False)
    log("=== Pair diagnostics by reason ===")
    for k, v in counts.items():
        log(f"{k:>14}: {v}")

    # show top-20 near passes (lowest med p, then half-life)
    near = diag.sort_values(["pval_med","half_life_d"], ascending=[True, True]).head(20)
    log("=== Top 20 (diagnostic preview) ===")
    with pd.option_context("display.max_columns", 20, "display.width", 120):
        print(near)

    log(f"Saved diagnostics → {out_path}")

if __name__ == "__main__":
    main()
