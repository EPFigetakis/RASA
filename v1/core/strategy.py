from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

CVAR95_NORMAL_MULT = 2.062  # ≈ E[loss | loss > VaR_95] / sigma under Normal

@dataclass
class StrategyParams:
    # signal bands
    entry_z: float = 2.0
    exit_z: float = 0.5
    hard_stop_z: float = 4.0
    # holding
    max_holding_days: int = 20
    # sizing & risk
    max_gross_per_pair: float = 50_000.0     # $ max gross (|long|+|short|) per pair
    cvar95_limit_per_pair: float = 1_000.0   # $ one-day CVaR limit per pair
    risk_window: int = 60                    # days to estimate residual sigma
    z_cap: float = 3.5                       # saturation level for linear sizing
    # costs (bps per leg, applied on open and close)
    commission_bps_per_leg: float = 1.0
    slippage_bps_per_leg: float = 1.0

    def total_bps_per_leg(self) -> float:
        return float(self.commission_bps_per_leg + self.slippage_bps_per_leg)


def compute_target_gross(
    z: float,
    resid_sigma: float,
    params: StrategyParams
) -> float:
    """
    Convert z-score into target gross $ subject to a per-pair CVaR limit.
    Linear sizing from entry_z -> z_cap, then clamp.
    """
    az = abs(float(z))
    if np.isnan(az) or az < params.entry_z:
        return 0.0

    # linear ramp from entry_z up to z_cap
    if params.z_cap <= params.entry_z:
        size_factor = 1.0
    else:
        size_factor = min(1.0, (az - params.entry_z) / (params.z_cap - params.entry_z))

    raw_gross = params.max_gross_per_pair * size_factor

    # CVaR guardrail (Normal approx): CVaR95 ≈ c * sigma * gross_unit
    # Treat residual sigma as unit-return scale of the position; conservative.
    sigma = max(1e-8, float(resid_sigma))
    cvar_est = CVAR95_NORMAL_MULT * sigma * raw_gross
    if cvar_est > params.cvar95_limit_per_pair:
        scale = params.cvar95_limit_per_pair / cvar_est
        raw_gross *= scale

    return float(raw_gross)


def target_shares(
    price_a: float,
    price_b: float,
    beta: float,
    side: int,
    gross: float
) -> tuple[float, float]:
    """
    Convert target gross $ and side into beta-scaled share counts (nA, nB).

      side = +1  => long A / short beta*B
      side = -1  => short A / long  beta*B

    We allocate dollars with |B| leg ≈ |beta| times A leg to reflect the hedge elasticity:
      $A = gross / (1 + |beta|),  $B = |beta| * $A

    Returns:
      nA, nB  (shares; may be negative for shorts)
    """
    if gross <= 0 or price_a <= 0 or price_b <= 0 or not np.isfinite(beta):
        return 0.0, 0.0

    b = abs(float(beta))
    a_dollars = gross / (1.0 + b)
    b_dollars = b * a_dollars

    if side > 0:
        # long A, short B
        nA = + a_dollars / price_a
        nB = - b_dollars / price_b
    else:
        # short A, long B
        nA = - a_dollars / price_a
        nB = + b_dollars / price_b

    return float(nA), float(nB)


def trade_cost_dollars(
    price_a: float,
    price_b: float,
    nA: float,
    nB: float,
    params: StrategyParams
) -> float:
    """
    Round-turn cost per trade *leg*, applied at order time (open or close).
    cost = (|nA|*PA + |nB|*PB) * bps_per_leg / 1e4
    """
    notional = abs(nA) * price_a + abs(nB) * price_b
    return notional * (params.total_bps_per_leg() / 1e4)
