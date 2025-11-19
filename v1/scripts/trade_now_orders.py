# --- path shim: run from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from core import config
from core.data import load_prices
from core.pairs import rank_candidate_pairs, ou_half_life
from core.kalman import kalman_spread
from core.strategy import StrategyParams, compute_target_gross, target_shares, trade_cost_dollars
from core.utils import log

def parse_args():
    ap = argparse.ArgumentParser(description="RASA: generate clean 'trade now' order tickets from latest prices")
    # universe scan
    ap.add_argument("--top_k", type=int, default=25, help="Evaluate top K ranked pairs")
    ap.add_argument("--force_refresh", action="store_true", help="Force price re-download")
    # signal bands & risk
    ap.add_argument("--entry_z", type=float, default=2.0)
    ap.add_argument("--exit_z", type=float, default=0.5)
    ap.add_argument("--hard_stop_z", type=float, default=6.0)  # a bit wider for live entries
    ap.add_argument("--time_stop_days", type=int, default=30)
    ap.add_argument("--risk_window", type=int, default=60)
    # Kalman
    ap.add_argument("--roll_window", type=int, default=60)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--R", type=float, default=1e-3)
    # sizing
    ap.add_argument("--max_gross", type=float, default=50_000.0)
    ap.add_argument("--cvar_limit", type=float, default=1_000.0)
    ap.add_argument("--z_cap", type=float, default=3.5)
    # order formatting
    ap.add_argument("--order_type", choices=["market", "limit"], default="market")
    ap.add_argument("--limit_buffer_bps", type=float, default=10.0, help="Limit buffer in bps from last price")
    ap.add_argument("--tif", choices=["DAY", "GTC"], default="DAY")
    ap.add_argument("--round_lot", type=int, default=1, help="Round share qty to multiple (e.g., 1 or 5 or 10)")
    return ap.parse_args()

def _est_hold_days(z_now: float, exit_z: float, hl_days: float, cap_days: int) -> int:
    if not np.isfinite(z_now) or not np.isfinite(hl_days) or hl_days <= 0 or exit_z <= 0:
        return cap_days
    az = abs(z_now)
    if az <= exit_z:
        return 0
    t = hl_days * np.log(az / exit_z) / np.log(2.0)
    if not np.isfinite(t) or t < 0:
        return cap_days
    return int(min(round(t), cap_days))

def _round_to_multiple(x: float, multiple: int) -> int:
    if multiple <= 1:
        return int(np.round(x))
    return int(np.round(x / multiple) * multiple)

def _limit_price(side_action: str, ref_price: float, buffer_bps: float) -> float:
    """
    side_action: 'BUY' or 'SELL' for the leg
    BUY  -> limit ABOVE last price by buffer
    SELL -> limit BELOW last price by buffer
    """
    if buffer_bps <= 0:
        return ref_price
    bump = ref_price * (buffer_bps / 1e4)
    return ref_price + bump if side_action == "BUY" else ref_price - bump

def main():
    args = parse_args()

    # 1) Load prices
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
        force_refresh=bool(args.force_refresh)
    )

    #
