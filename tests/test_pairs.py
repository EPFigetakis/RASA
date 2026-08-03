# tests/test_pairs.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import pairs


def test_estimate_hedge_ratio_recovers_known_beta_no_intercept():
    x = np.linspace(50, 60, 100)
    y = 2.0 * x  # exact, no noise, no intercept
    df_pair = pd.DataFrame({"X": x, "Y": y})

    cfg = pairs.HedgeConfig(use_log_prices=False, add_intercept=False)
    alpha, beta = pairs.estimate_hedge_ratio(df_pair, cfg)

    assert alpha == 0.0
    assert np.isclose(beta, 2.0)


def test_estimate_hedge_ratio_recovers_known_alpha_and_beta():
    x = np.linspace(50, 60, 100)
    y = 5.0 + 1.5 * x  # exact linear relationship
    df_pair = pd.DataFrame({"X": x, "Y": y})

    cfg = pairs.HedgeConfig(use_log_prices=False, add_intercept=True)
    alpha, beta = pairs.estimate_hedge_ratio(df_pair, cfg)

    assert np.isclose(alpha, 5.0)
    assert np.isclose(beta, 1.5)


def test_compute_spread_is_zero_for_perfect_fit():
    x = np.linspace(50, 60, 100)
    y = 5.0 + 1.5 * x
    df_pair = pd.DataFrame({"X": x, "Y": y})
    cfg = pairs.HedgeConfig(use_log_prices=False, add_intercept=True)

    spread = pairs.compute_spread(df_pair, alpha=5.0, beta=1.5, cfg=cfg)
    assert np.allclose(spread.to_numpy(), 0.0, atol=1e-8)


def test_compute_spread_zscore_is_standardized_over_window():
    rng = np.random.default_rng(42)
    spread = pd.Series(rng.normal(size=300))
    z = pairs.compute_spread_zscore(spread, window=50, min_periods=50)

    tail = z.dropna()
    assert len(tail) > 0
    # Rolling z-scores should be roughly standardized, not exploding.
    assert tail.abs().median() < 3


def test_generate_pair_signals_state_machine():
    # flat -> entry (z > entry_z) -> hold through noise -> exit (|z| < exit_z)
    z = pd.Series([0.0, 2.5, 1.0, 0.6, 0.3, -0.1, np.nan, 3.0])
    cfg = pairs.SignalConfig(entry_z=2.0, exit_z=0.5, max_position=1)

    signals = pairs.generate_pair_signals(z, cfg)

    assert signals.iloc[0] == 0.0  # flat, below entry
    assert signals.iloc[1] == -1.0  # entered short spread (z > entry_z)
    assert signals.iloc[2] == -1.0  # held despite z dropping (still above exit_z)
    assert signals.iloc[3] == -1.0  # still held (0.6 >= exit_z)
    assert signals.iloc[4] == 0.0  # exited (0.3 < exit_z)
    assert signals.iloc[5] == 0.0  # stays flat
    assert signals.iloc[6] == 0.0  # NaN keeps previous (flat) position
    assert signals.iloc[7] == -1.0  # new entry from flat


def test_extract_pair_prices_raises_on_missing_ticker():
    px = pd.DataFrame(
        {"AAA": [1.0, 2.0]},
        index=pd.date_range("2021-01-01", periods=2),
    )
    spec = pairs.PairSpec(x="AAA", y="MISSING")
    try:
        pairs.extract_pair_prices(px, spec)
        assert False, "expected KeyError for missing ticker"
    except KeyError:
        pass
