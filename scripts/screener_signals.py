# --- path shim: run from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from core import config
from core.data import load_prices
from core.pairs import rank_candidate_pairs, ou_half_life
from core.kalman import kalman_spread
from core.strategy import StrategyParams, compute_target_gross, target_shares, trade_cost_dollars
from core.utils import log

def parse_args():
    ap = argparse.ArgumentParser(description="RASA screener: immediate trade suggestions with exits/hold-time")
    ap.add_argument("--top_k", type=int, default=25, help="Evaluate top K ranked pairs")
    ap.add_argument("--entry_z", type=float, default=2.0)
    ap.add_argument("--exit_z", type=float, default=0.5)
    ap.add_argument("--hard_stop_z", type=float, default=4.0, help="Hard stop |z| threshold")
    ap.add_argument("--time_stop_days", type=int, default=20, help="Max holding days")
    ap.add_argument("--risk_window", type=int, default=60)
    ap.add_argument("--roll_window", type=int, default=60, help="Kalman residual z roll window")
    ap.add_argument("--delta", type=float, default=1e-5, help="Kalman state drift")
    ap.add_argument("--R", type=float, default=1e-3, help="Kalman obs noise variance")
    ap.add_argument("--max_gross", type=float, default=50_000.0)
    ap.add_argument("--cvar_limit", type=float, default=1_000.0)
    ap.add_argument("--z_cap", type=float, default=3.5)
    ap.add_argument("--force_refresh", action="store_true", help="Force price re-download")
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

def main():
    args = parse_args()

    # 1) Load latest prices (optionally force refresh)
    prices = load_prices(
        config.UNIVERSE,
        start=config.START_DATE,
        end=config.END_DATE,
        data_dir=config.DATA_DIR,
        force_refresh=bool(args.force_refresh)
    )

    # 2) Rank candidates
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
        log("No passing pairs under current filters.")
        return

    # 3) Strategy params (for sizing/costs)
    params = StrategyParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        hard_stop_z=args.hard_stop_z,
        max_holding_days=args.time_stop_days,
        max_gross_per_pair=args.max_gross,
        cvar95_limit_per_pair=args.cvar_limit,
        risk_window=args.risk_window,
        z_cap=args.z_cap,
    )

    rows = []
    take_rows = []
    n_eval = min(args.top_k, len(cand))

    for i in range(n_eval):
        A, B = cand.iloc[i][["A","B"]]
        ks = kalman_spread(
            prices[A], prices[B],
            roll_window=args.roll_window,
            delta=float(args.delta),
            R=float(args.R)
        ).dropna()
        if ks.empty:
            continue

        # Latest stats
        z = float(ks["z_resid"].iloc[-1])
        beta = float(ks["beta"].iloc[-1])
        pa = float(ks["price_a"].iloc[-1])
        pb = float(ks["price_b"].iloc[-1])

        # Residual sigma for sizing (fallbacks if needed)
        sig = float(ks["resid"].rolling(args.risk_window).std(ddof=1).iloc[-1])
        if not np.isfinite(sig) or sig == 0:
            w = max(args.risk_window, 2*args.roll_window, 30)
            sig = float(ks["resid"].tail(w).std(ddof=1)) if len(ks) >= 10 else float(ks["resid"].std(ddof=1))

        # Half-life on residuals (OU AR(1) approximation)
        hl = ou_half_life(ks["resid"], min_len=max(30, args.risk_window//2))
        if not np.isfinite(hl):
            hl = np.nan

        # Side + sizing
        action = "HOLD"
        side = 0
        gross = 0.0
        nA = nB = 0.0
        nA_round = nB_round = 0

        if np.isfinite(z):
            az = abs(z)
            if az >= params.entry_z:
                action = "ENTER_SHORT_A_LONG_betaB" if z > 0 else "ENTER_LONG_A_SHORT_betaB"
                side = -1 if z > 0 else +1
                gross = compute_target_gross(z, sig, params)
                nA, nB = target_shares(pa, pb, beta, side, gross)
                # rounded shares for a practical ticket
                nA_round = int(np.round(nA))
                nB_round = int(np.round(nB))
            elif az <= params.exit_z:
                action = "EXIT_IF_OPEN"

        # Costs (open only, for a sense of scale)
        open_cost = trade_cost_dollars(pa, pb, nA_round or nA, nB_round or nB, params) if gross > 0 else 0.0

        # Exit plan + hold-time estimate
        est_hold = _est_hold_days(z, params.exit_z, hl if np.isfinite(hl) else args.time_stop_days, args.time_stop_days)
        exit_plan = f"Exit at |z|≤{params.exit_z:.2f}, hard stop |z|≥{params.hard_stop_z:.2f}, time stop {args.time_stop_days}d"
        est_pnl_exit = 0.0
        if gross > 0 and np.isfinite(z) and np.isfinite(sig):
            est_pnl_exit = max(0.0, (abs(z) - params.exit_z)) * sig * gross - 2 * open_cost  # rough round-trip costs

        r = {
            "rank": i+1, "A": A, "B": B,
            "z": z, "beta": beta, "sigma_resid": sig, "hl_resid_d": hl,
            "price_a": pa, "price_b": pb,
            "side": ("LONG_A/SHORT_betaB" if side==+1 else "SHORT_A/LONG_betaB") if gross>0 else "",
            "gross_target": gross,
            "shares_A": nA, "shares_B": nB,
            "shares_A_round": nA_round, "shares_B_round": nB_round,
            "open_cost_est": open_cost,
            "est_hold_days": est_hold,
            "est_pnl_to_exit": est_pnl_exit,
            "exit_plan": exit_plan,
            "action": action
        }
        rows.append(r)
        if action.startswith("ENTER"):
            take_rows.append(r)

    df = pd.DataFrame(rows).sort_values(by="z", key=lambda s: s.abs(), ascending=False)
    nowtag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_all = Path(config.DATA_DIR) / f"signals_all_{nowtag}.csv"
    df.to_csv(out_all, index=False)

    # Console summary
    log(f"Evaluated top {n_eval} pairs; {len(take_rows)} meet entry criteria (|z| ≥ {params.entry_z}).")
    if take_rows:
        df_take = pd.DataFrame(take_rows).sort_values(by="z", key=lambda s: s.abs(), ascending=False)
        out_take = Path(config.DATA_DIR) / f"signals_take_{nowtag}.csv"
        df_take.to_csv(out_take, index=False)

        # Compact view for terminal
        cols = ["rank","A","B","side","z","beta","sigma_resid","hl_resid_d",
                "gross_target","shares_A_round","shares_B_round",
                "est_hold_days","est_pnl_to_exit","exit_plan"]
        with pd.option_context("display.width", 160, "display.max_columns", None, "display.float_format", "{:,.4f}".format):
            print(df_take[cols].head(20))
        log(f"Saved TAKE signals → {out_take}")
    else:
        log("No immediate entries at current thresholds.")

    log(f"Saved ALL signals → {out_all}")

if __name__ == "__main__":
    main()
