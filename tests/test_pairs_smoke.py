import pandas as pd
from core.pairs import build_spread, ou_half_life

def test_spread_and_half_life_smoke():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    x = pd.Series(100.0, index=idx)
    y = x * 1.01
    spr, beta, z = build_spread(x, y)
    hl = ou_half_life(spr)
    assert pd.isna(beta) is False
    assert pd.isna(hl) is False
