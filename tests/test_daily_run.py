# tests/test_daily_run.py
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from core import daily_run, orders, risk
from core.broker_paper import PaperBroker, PaperBrokerConfig
from core.orders import OrderStatus

TODAY = date(2026, 11, 11)          # expected last bar: Tue 2026-11-10


def _px(n=300, extra_days=0, jump_y=1.0):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-11-10", periods=n)
    x = 100 + np.cumsum(rng.normal(size=n))
    noise = rng.normal(size=n)
    noise[-3:] += 9                              # spread far above its mean -> SHORT_SPREAD
    y = 2 * x + noise
    px = pd.DataFrame({"X": x, "Y": y, "Z": 80 + np.cumsum(rng.normal(size=n))}, index=idx)
    for i in range(extra_days):                  # same prices again on following days
        row = px.iloc[-1].copy()
        row["Y"] *= jump_y
        px.loc[px.index[-1] + pd.offsets.BDay(1)] = row
    return px


def _pairs():
    return pd.DataFrame([{"x": "X", "y": "Y", "pvalue": 0.001, "hedge_ratio": 2.0, "alpha": 0.0,
                          "half_life": 5.0, "n_obs": 300}])


def _cfgs(**risk_kw):
    run_cfg = daily_run.RunConfig(top_n=5, notional_per_pair=5_000.0)
    return run_cfg, risk.RiskConfig(**risk_kw)


@pytest.fixture
def env(tmp_path):
    broker = PaperBroker(tmp_path / "p.db", PaperBrokerConfig(
        commission_per_share=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0))
    return broker, tmp_path / "runs"


def _run(env, px=None, today=TODAY, risk_cfg=None, **kw):
    broker, rdir = env
    run_cfg, default_risk = _cfgs()
    return daily_run.run_daily(px if px is not None else _px(), _pairs(), run_cfg,
                               risk_cfg or default_risk, broker, today=today, record_dir=rdir, **kw)


# ------------------------------- basic run -------------------------------

def test_run_proposes_a_ticket_without_touching_the_account(env):
    broker, rdir = env
    rec = _run(env)
    assert rec.status == daily_run.STATUS_OK and rec.n_ideas == 1 and rec.data_age_business_days == 0
    [t] = rec.ticket_objects()
    assert t.status == OrderStatus.PROPOSED and t.pair == "X/Y"
    assert {l.symbol: l.side for l in t.legs} == {"X": orders.OrderSide.BUY, "Y": orders.OrderSide.SELL}
    assert all(abs(l.quantity - round(l.quantity)) < 1e-9 for l in t.legs)     # whole shares
    assert broker.get_positions() == {} and broker.list_tickets() == []        # nothing submitted
    assert broker.last_bar() == "2026-11-10"                                    # bar processed
    assert (rdir / f"{rec.run_id}.json").exists()
    assert daily_run.load_run_record(rec.run_id, rdir).to_dict() == rec.to_dict()


def test_run_is_idempotent(env):
    a, b = _run(env), _run(env)
    assert a.run_id == b.run_id
    assert [t.ticket_id for t in a.ticket_objects()] == [t.ticket_id for t in b.ticket_objects()]
    assert len(list((env[1]).glob("R*.json"))) == 1


def test_stale_data_aborts_before_touching_broker(env):
    broker, _ = env
    rec = _run(env, today=date(2026, 11, 20))
    assert rec.status == daily_run.STATUS_STALE and rec.tickets == []
    assert broker.last_bar() is None and any("aborted" in w for w in rec.warnings)


def test_dry_run_writes_nothing(env):
    broker, rdir = env
    rec = _run(env, dry_run=True)
    assert rec.n_ideas == 1 and rec.tickets
    assert broker.last_bar() is None and not rdir.exists()


def test_live_environment_is_refused(env):
    with pytest.raises(daily_run.RunError, match="paper-only"):
        _run(env, risk_cfg=risk.RiskConfig(environment="live"))


# ------------------------------- approval flow -------------------------------

def test_approve_submits_then_rerun_does_not_repropose(env):
    broker, rdir = env
    rec = _run(env)
    [tid] = [t.ticket_id for t in rec.ticket_objects()]
    done = daily_run.approve_and_submit(rec, broker, approved_by="tester", record_dir=rdir)
    assert [t.status for t in done] == [OrderStatus.SUBMITTED]
    assert "approved by tester" in broker.get_ticket(tid).history[1].reason
    assert daily_run.load_run_record(rec.run_id, rdir).submitted_ticket_ids == [tid]

    again = _run(env)                                    # same day, ticket still working
    assert again.tickets == [] and any("already with broker" in w for w in again.warnings)
    assert again.status == daily_run.STATUS_NO_ACTION
    assert len(broker.list_tickets()) == 1


def test_next_day_run_fills_the_ticket_and_stays_flat_on_orders(env):
    broker, rdir = env
    rec = _run(env)
    daily_run.approve_and_submit(rec, broker, record_dir=rdir)
    nxt = _run(env, px=_px(extra_days=1), today=date(2026, 11, 12))
    assert nxt.fills_processed == 2
    pos = broker.get_positions()
    assert pos["X"] > 0 > pos["Y"]
    # holdings now match the target, so the same idea is not proposed again
    assert [t for t in nxt.ticket_objects() if t.purpose == "idea"] == []


def test_approve_refused_when_broker_moved_on(env):
    broker, rdir = env
    rec = _run(env)
    _run(env, px=_px(extra_days=1), today=date(2026, 11, 12))      # advances the broker bar
    with pytest.raises(daily_run.RunError, match="re-run"):
        daily_run.approve_and_submit(rec, broker, record_dir=rdir)
    assert broker.list_tickets() == []


def test_approve_refused_when_positions_changed(env):
    broker, rdir = env
    rec = _run(env)
    broker.portfolio.apply_fill("X", 1.0, 100.0)                   # something else traded
    with pytest.raises(daily_run.RunError, match="positions changed"):
        daily_run.approve_and_submit(rec, broker, record_dir=rdir)


def test_rejected_tickets_cannot_be_approved(env):
    broker, rdir = env
    tight = risk.RiskConfig(max_pair_notional=100.0)
    rec = _run(env, risk_cfg=tight)
    [t] = rec.ticket_objects()
    assert t.status == OrderStatus.REJECTED
    assert daily_run.approve_and_submit(rec, broker, record_dir=rdir) == []
    with pytest.raises(daily_run.RunError, match="not eligible"):
        daily_run.approve_and_submit(rec, broker, [t.ticket_id], record_dir=rdir)
    with pytest.raises(daily_run.RunError, match="unknown"):
        daily_run.approve_and_submit(rec, broker, ["Tbogus"], record_dir=rdir)


def test_auto_approval_config_and_submit_flag(env):
    broker, rdir = env
    auto = risk.RiskConfig(require_manual_approval=False)
    rec = _run(env, risk_cfg=auto, submit=True)
    assert rec.submitted_ticket_ids and broker.list_tickets()[0].status == OrderStatus.SUBMITTED
    # Without --submit, approved tickets are only proposed.
    broker2 = PaperBroker(rdir.parent / "p2.db")
    rec2 = daily_run.run_daily(_px(), _pairs(), daily_run.RunConfig(notional_per_pair=5000.0), auto,
                               broker2, today=TODAY, record_dir=rdir)
    assert rec2.ticket_objects()[0].status == OrderStatus.APPROVED and broker2.list_tickets() == []


# ------------------------------- kill switch / exits -------------------------------

def test_kill_switch_status_after_loss(env):
    broker, rdir = env
    rec = _run(env)
    daily_run.approve_and_submit(rec, broker, record_dir=rdir)
    _run(env, px=_px(extra_days=1), today=date(2026, 11, 12))      # day 2: ticket fills
    # Day 3 the short leg moves against us a lot; limits are tiny so the switch trips.
    harsh = risk.RiskConfig(max_drawdown_pct=0.0001, max_daily_loss_pct=0.0001)
    nxt = _run(env, px=_px(extra_days=2, jump_y=1.10), today=date(2026, 11, 13), risk_cfg=harsh)
    assert nxt.status == daily_run.STATUS_KILL and nxt.kill_switch_reasons


def test_held_pair_that_goes_flat_gets_a_close_out_ticket(env):
    broker, rdir = env
    rec = _run(env)
    daily_run.approve_and_submit(rec, broker, record_dir=rdir)
    # Spread collapses back to its mean on the next bars -> signal exits.
    px = _px(extra_days=1)
    px.iloc[-1, px.columns.get_loc("Y")] = 2 * px.iloc[-1]["X"]
    run_cfg, risk_cfg = _cfgs()
    out = daily_run.run_daily(px, _pairs(), run_cfg, risk_cfg, broker, today=date(2026, 11, 12),
                              record_dir=rdir)
    closes = [t for t in out.ticket_objects() if t.purpose == "close_out"]
    assert closes and {l.symbol for l in closes[0].legs} == {"X", "Y"}
    assert closes[0].status == OrderStatus.PROPOSED and out.risk_report["decisions"][0]["reduces_risk"]


# ------------------------------- CLI -------------------------------

def test_cli_exit_codes(env, monkeypatch, tmp_path):
    import run_daily_proposals as cli
    from core import paths

    monkeypatch.setattr(paths, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)

    def no_pairs(update_prices=True):
        raise daily_run.RunError("No verified saved pair selection found.")

    monkeypatch.setattr(daily_run, "load_inputs", no_pairs)
    assert cli.main(["--db", str(tmp_path / "x.db"), "run"]) == 2

    monkeypatch.setattr(daily_run, "load_inputs", lambda update_prices=True: (_px(), _pairs(), []))
    monkeypatch.setattr(daily_run.prices, "business_days_behind", lambda *a, **k: 10)
    assert cli.main(["--db", str(tmp_path / "x.db"), "run"]) == 2          # stale data

    monkeypatch.setattr(daily_run.prices, "business_days_behind", lambda *a, **k: 0)
    assert cli.main(["--db", str(tmp_path / "x.db"), "run"]) == 0
    assert cli.main(["--db", str(tmp_path / "x.db"), "list"]) == 0
    assert cli.main(["--db", str(tmp_path / "x.db"), "approve", "Rnope", "--all"]) == 2
