# tests/test_cache_freshness.py
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import pair_selection, prices


def _panel(start="2024-01-01", periods=30, cols=("AAA", "BBB"), seed=0):
    idx = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(seed)
    data = 100 + np.cumsum(rng.normal(size=(periods, len(cols))), axis=0)
    return pd.DataFrame(data, index=idx, columns=list(cols))


# ---------------------------- freshness maths ----------------------------

def test_latest_expected_bar_is_previous_weekday():
    assert prices.latest_expected_bar(date(2024, 3, 6)) == date(2024, 3, 5)   # Wed -> Tue
    assert prices.latest_expected_bar(date(2024, 3, 4)) == date(2024, 3, 1)   # Mon -> Fri
    assert prices.latest_expected_bar(date(2024, 3, 9)) == date(2024, 3, 8)   # Sat -> Fri


def test_business_days_behind_and_staleness():
    today = date(2024, 3, 6)  # expected bar = Tue 3/5
    assert prices.business_days_behind(date(2024, 3, 5), today) == 0
    assert prices.business_days_behind(date(2024, 3, 4), today) == 1
    assert prices.business_days_behind(date(2024, 3, 1), today) == 2
    assert not prices.is_stale(date(2024, 3, 1), today, stale_after_business_days=2)
    assert prices.is_stale(date(2024, 2, 29), today, stale_after_business_days=2)
    # A last bar in the "future" is never behind.
    assert prices.business_days_behind(date(2024, 3, 20), today) == 0


def test_panel_fingerprint_changes_with_any_value_or_column():
    a = _panel()
    assert prices.panel_fingerprint(a) == prices.panel_fingerprint(a.copy())
    b = a.copy()
    b.iloc[3, 0] += 0.01
    assert prices.panel_fingerprint(a) != prices.panel_fingerprint(b)
    assert prices.panel_fingerprint(a) != prices.panel_fingerprint(a[["AAA"]])


def test_get_cache_status(tmp_path):
    px = _panel(start="2024-01-01", periods=10)
    key = prices.build_cache_key(["AAA", "BBB"], "2024-01-01", None)
    status = prices.get_cache_status(["AAA", "BBB"], "2024-01-01", None, data_dir=tmp_path)
    assert status["exists"] is False

    path = tmp_path / f"{key}.parquet"
    px.to_parquet(path)
    prices._write_cache_metadata(path, ["AAA", "BBB"], "2024-01-01", None, "1d", px)

    status = prices.get_cache_status(
        ["AAA", "BBB"], "2024-01-01", None, data_dir=tmp_path, today=date(2024, 3, 6)
    )
    assert status["exists"] and status["is_stale"]
    assert status["last_bar_date"] == px.index.max().date().isoformat()

    # A fixed end date can never get new data -> never "stale".
    fixed = prices.get_cache_status(
        ["AAA", "BBB"], "2024-01-01", "2024-01-12", data_dir=tmp_path, cache_key=key,
        today=date(2024, 3, 6),
    )
    assert fixed["fixed_end"] and not fixed["is_stale"]


# ---------------------------- incremental update ----------------------------

def test_update_cached_prices_appends_only_new_bars(monkeypatch):
    full = _panel(periods=40)
    cached = full.iloc[:30]
    monkeypatch.setattr(prices, "_download_close_panel", lambda t, s, e: full.loc[s:])

    out = prices.update_cached_prices(cached, ["AAA", "BBB"], "2024-01-01", None)
    assert out is not None
    pd.testing.assert_frame_equal(out, full)


def test_update_cached_prices_detects_restated_history(monkeypatch):
    full = _panel(periods=40)
    cached = full.iloc[:30]
    restated = full * 0.97  # e.g. a dividend re-based the adjusted history
    monkeypatch.setattr(prices, "_download_close_panel", lambda t, s, e: restated.loc[s:])
    assert prices.update_cached_prices(cached, ["AAA", "BBB"], "2024-01-01", None) is None


def test_load_prices_updates_stale_cache_incrementally(tmp_path, monkeypatch):
    full = _panel(start="2024-01-01", periods=40)
    first = full.iloc[:30]
    calls = []

    def fake_download(tickers, start, end):
        calls.append(start)
        return full.loc[start:] if start >= "2024-01-01" else full

    monkeypatch.setattr(prices, "_download_close_panel", fake_download)
    tick = ["AAA", "BBB"]

    # Seed the cache with the first 30 bars.
    key = prices.build_cache_key(tick, "2024-01-01", None)
    path = tmp_path / f"{key}.parquet"
    first.to_parquet(path)
    prices._write_cache_metadata(path, tick, "2024-01-01", None, "1d", first)

    # Without update_if_stale the old data is served (and nothing is downloaded).
    stale = prices.load_prices(tick, "2024-01-01", data_dir=tmp_path)
    assert len(stale) == 30 and calls == []

    updated = prices.load_prices(tick, "2024-01-01", data_dir=tmp_path, update_if_stale=True)
    assert len(updated) == 40
    assert len(calls) == 1  # one short incremental fetch, not a full download
    meta = prices.read_cache_metadata(path)
    assert meta["last_bar_date"] == full.index.max().date().isoformat()
    assert meta["panel_fingerprint"] == prices.panel_fingerprint(full)
    # Reloading is served from the refreshed cache.
    assert len(prices.load_prices(tick, "2024-01-01", data_dir=tmp_path)) == 40


def test_load_prices_keeps_old_cache_when_update_fails(tmp_path, monkeypatch):
    first = _panel(periods=30)
    tick = ["AAA", "BBB"]
    key = prices.build_cache_key(tick, "2024-01-01", None)
    path = tmp_path / f"{key}.parquet"
    first.to_parquet(path)
    prices._write_cache_metadata(path, tick, "2024-01-01", None, "1d", first)

    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(prices, "_download_close_panel", boom)
    out = prices.load_prices(tick, "2024-01-01", data_dir=tmp_path, update_if_stale=True)
    assert len(out) == 30


# ---------------------------- pair-cache provenance ----------------------------

def _cointegrated_px(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = 100 + np.cumsum(rng.normal(size=n))
    y = 2 * x + rng.normal(scale=0.5, size=n)
    z = 50 + np.cumsum(rng.normal(size=n))
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=idx)


@pytest.fixture
def pair_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(pair_selection.paths, "cache_dir", lambda: tmp_path)
    return pair_selection.PairSelectionConfig(max_pvalue=0.2, min_obs=100, max_pairs=10)


def test_pair_cache_ok_when_prices_and_settings_match(pair_cfg):
    px = _cointegrated_px()
    pair_selection.find_candidate_pairs(px, pair_cfg)
    assert pair_selection.check_candidate_pairs_cache(px, pair_cfg).status == "ok"
    df = pair_selection.load_candidate_pairs(pair_cfg, px=px)
    assert not df.empty


def test_pair_cache_mismatch_on_universe_or_settings_blocks_load(pair_cfg):
    from dataclasses import replace

    px = _cointegrated_px()
    pair_selection.find_candidate_pairs(px, pair_cfg)

    other_universe = px[["X", "Y"]]
    check = pair_selection.check_candidate_pairs_cache(other_universe, pair_cfg)
    assert check.status == "mismatch" and "Universe" in check.reasons[0]
    with pytest.raises(pair_selection.StaleCacheError):
        pair_selection.load_candidate_pairs(pair_cfg, px=other_universe)

    changed = replace(pair_cfg, max_pvalue=0.01)
    check = pair_selection.check_candidate_pairs_cache(px, changed)
    assert check.status == "mismatch" and "max_pvalue" in check.reasons[0]
    # Non-strict mode loads anyway (and only warns).
    assert not pair_selection.load_candidate_pairs(changed, px=px, strict=False).empty


def test_pair_cache_flags_changed_prices_but_still_loads(pair_cfg):
    px = _cointegrated_px()
    pair_selection.find_candidate_pairs(px, pair_cfg)
    newer = pd.concat([px, _cointegrated_px(n=410).iloc[400:]])  # extra bars appended
    check = pair_selection.check_candidate_pairs_cache(newer, pair_cfg)
    assert check.status == "data_changed" and check.usable
    assert not pair_selection.load_candidate_pairs(pair_cfg, px=newer).empty


def test_legacy_pair_csv_without_metadata_is_reported(pair_cfg):
    px = _cointegrated_px()
    pair_selection.find_candidate_pairs(px, pair_cfg)
    pair_selection.get_pairs_metadata_path(pair_cfg).unlink()
    assert pair_selection.check_candidate_pairs_cache(px, pair_cfg).status == "no_metadata"


def test_empty_result_overwrites_previous_saved_selection(pair_cfg):
    from dataclasses import replace

    px = _cointegrated_px()
    first = pair_selection.find_candidate_pairs(px, pair_cfg)
    assert not first.empty
    impossible = replace(pair_cfg, max_pvalue=-1.0)
    assert pair_selection.find_candidate_pairs(px, impossible).empty
    # The stale non-empty table must be gone, not silently reloadable.
    assert pair_selection.load_candidate_pairs(impossible).empty
