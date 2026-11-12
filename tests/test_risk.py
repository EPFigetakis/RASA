# tests/test_risk.py
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import orders, risk
from core.orders import OrderStatus
from core.trade_proposals import TradeIdea

AS_OF = pd.Timestamp("2026-11-10")


def _idea(x="X", y="Y", qty_x=20.0, qty_y=-10.0, px=50.0, py=100.0, position=-1):
    return TradeIdea(
        x=x, y=y, as_of=AS_OF, side="SHORT_SPREAD" if position < 0 else "LONG_SPREAD",
        position=position, zscore=2.5, spread=1.0, beta=2.0, alpha=0.0,
        price_x=px, price_y=py, qty_x=qty_x, qty_y=qty_y, notional_target=2000.0,
        bars_in_position=1,
    )


def _cfg(**kw):
    base = dict(capital_base=100_000, max_pair_notional=10_000, max_symbol_exposure_pct=0.10,
                max_gross_exposure_pct=1.0, max_net_exposure_pct=0.10, max_concurrent_pairs=10)
    return risk.RiskConfig(**{**base, **kw})


def _eval(ideas, cfg=None, current=None, **kw):
    plan = orders.plan_orders(ideas, current or {}, **kw.pop("plan_kw", {}))
    return plan, risk.evaluate_plan(plan, cfg or _cfg(), **kw)


def _failed(report, i=0):
    return {c.name for c in report.decisions[i].failures}


# ------------------------------- config -------------------------------

def test_default_config_file_loads_and_is_valid():
    cfg = risk.load_risk_config()
    assert cfg.environment == "research" and cfg.require_manual_approval
    assert cfg.config_hash == risk.load_risk_config().config_hash


def test_config_rejects_unknown_keys_and_bad_values(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_gross_exposure": 1.0}))      # typo of a real key
    with pytest.raises(ValueError, match="Unknown keys"):
        risk.load_risk_config(p)
    with pytest.raises(FileNotFoundError):
        risk.load_risk_config(tmp_path / "missing.json")
    for bad in (dict(environment="prod"), dict(capital_base=0), dict(max_concurrent_pairs=0),
                dict(allowed_order_types=("STOP",)), dict(allowed_order_types=())):
        with pytest.raises(ValueError):
            risk.RiskConfig(**bad)


def test_config_hash_changes_with_limits():
    assert _cfg().config_hash != _cfg(max_pair_notional=5_000).config_hash


# ------------------------------- passing / per-ticket limits -------------------------------

def test_clean_ticket_passes_every_check():
    # gross = 20*50 + 10*100 = 2000
    plan, rep = _eval([_idea()])
    d = rep.decisions[0]
    assert d.approved and not d.reduces_risk and not d.failures
    assert {c.name for c in d.checks} >= {"gross_exposure", "net_exposure", "pair_notional", "symbol_exposure", "min_price"}
    assert rep.after.gross == pytest.approx(2000) and rep.before.gross == 0


def test_pair_notional_and_symbol_limits():
    big = _idea(qty_x=300.0, qty_y=-150.0)          # 15k + 15k = 30k gross
    _, rep = _eval([big])
    assert {"pair_notional", "symbol_exposure"} <= _failed(rep)
    assert "pair_notional" in rep.decisions[0].reason


def test_min_price_and_whole_shares_and_order_type():
    _, rep = _eval([_idea(px=2.0)], _cfg(min_price=5.0))
    assert "min_price" in _failed(rep)

    _, rep = _eval([_idea(qty_x=20.5)], _cfg(allow_fractional_shares=False))
    assert "whole_shares" in _failed(rep)
    _, rep = _eval([_idea(qty_x=20.5)], _cfg(allow_fractional_shares=True))
    assert rep.decisions[0].approved

    _, rep = _eval([_idea()], _cfg(allowed_order_types=("LIMIT",)))
    assert "order_type_allowed" in _failed(rep)


def test_net_and_gross_exposure_limits():
    _, rep = _eval([_idea()], _cfg(max_net_exposure_pct=0.0))          # balanced: net = 20*50 - 10*100 = 0
    assert rep.decisions[0].approved
    unbalanced = _idea(qty_x=40.0, qty_y=-10.0)                        # net +1000
    _, rep = _eval([unbalanced], _cfg(max_net_exposure_pct=0.005))     # limit $500
    assert "net_exposure" in _failed(rep)
    _, rep = _eval([_idea()], _cfg(max_gross_exposure_pct=0.01))       # limit $1000 < 2000
    assert "gross_exposure" in _failed(rep)


# ------------------------------- portfolio-level, greedy -------------------------------

def test_concurrent_pairs_limit_is_greedy_in_plan_order():
    ideas = [_idea("A", "B"), _idea("C", "D"), _idea("E", "F")]
    plan, rep = _eval(ideas, _cfg(max_concurrent_pairs=2))
    assert [d.approved for d in rep.decisions] == [True, True, False]
    assert "concurrent_pairs" in _failed(rep, 2)
    assert rep.after.n_pairs == 2


def test_existing_open_pairs_count_toward_limit():
    _, rep = _eval([_idea()], _cfg(max_concurrent_pairs=3), account=risk.AccountState(equity=1e5, open_pairs=3))
    assert "concurrent_pairs" in _failed(rep)


def test_rebalancing_an_open_pair_is_not_a_new_pair():
    _, rep = _eval([_idea()], _cfg(max_concurrent_pairs=1), current={"X": 10.0, "Y": -5.0},
                   account=risk.AccountState(equity=1e5, open_pairs=1))
    assert rep.decisions[0].approved


def test_rejected_ticket_does_not_consume_capacity():
    ideas = [_idea("A", "B", qty_x=300.0, qty_y=-150.0), _idea("C", "D")]
    _, rep = _eval(ideas, _cfg())
    assert [d.approved for d in rep.decisions] == [False, True]
    assert rep.after.gross == pytest.approx(2000)


def test_gross_limit_fills_in_order_then_blocks():
    ideas = [_idea("A", "B"), _idea("C", "D"), _idea("E", "F")]       # 2000 gross each
    _, rep = _eval(ideas, _cfg(max_gross_exposure_pct=0.045))         # limit $4500
    assert [d.approved for d in rep.decisions] == [True, True, False]


# ------------------------------- data age, kill switch, reducing -------------------------------

def test_stale_data_blocks_new_risk():
    _, rep = _eval([_idea()], data_age_business_days=5)
    assert "data_fresh" in _failed(rep)
    _, rep = _eval([_idea()], data_age_business_days=1)
    assert rep.decisions[0].approved


def test_kill_switch_blocks_new_risk_but_allows_closing():
    acct = risk.AccountState(equity=97_000, day_start_equity=100_000, peak_equity=100_000)
    assert risk.kill_switch_status(_cfg(), acct)                      # 3% daily loss > 2%
    _, rep = _eval([_idea()], account=acct)
    assert rep.kill_switch and "kill_switch" in _failed(rep)

    plan = orders.plan_orders([], {"OLD": 10.0}, managed_symbols=["OLD"], close_out_prices={"OLD": 20.0})
    rep = risk.evaluate_plan(plan, _cfg(), account=acct, data_age_business_days=30)
    d = rep.decisions[0]
    assert d.approved and d.reduces_risk                              # even with stale data + kill switch


def test_drawdown_kill_switch():
    acct = risk.AccountState(equity=85_000, peak_equity=100_000)
    assert "drawdown" in risk.kill_switch_status(_cfg(), acct)[0]
    assert risk.kill_switch_status(_cfg(), risk.AccountState(equity=99_000, peak_equity=100_000)) == []
    assert risk.kill_switch_status(_cfg(), None) == []


def test_shrinking_a_position_counts_as_risk_reducing():
    # hold 40 X / -20 Y, target 20 / -10 -> sell 20 X, buy 10 Y
    plan, rep = _eval([_idea()], current={"X": 40.0, "Y": -20.0},
                      account=risk.AccountState(equity=97_000, day_start_equity=100_000))
    assert rep.decisions[0].reduces_risk and rep.decisions[0].approved


def test_unpriced_held_symbol_fails_closed():
    _, rep = _eval([_idea()], current={"MYSTERY": 10.0})
    assert "price_available" in _failed(rep)
    _, rep = _eval([_idea()], current={"MYSTERY": 10.0}, prices={"MYSTERY": 10.0})
    assert rep.decisions[0].approved
    assert rep.before.gross == pytest.approx(100)


# ------------------------------- conflicts, report, apply -------------------------------

def test_planner_conflicts_become_rejections():
    plan, rep = _eval([_idea("X", "Y"), _idea("X", "Z")])
    assert plan.tickets == [] and rep.decisions == []
    assert {u.subject for u in rep.unticketed} == {"X/Y", "X/Z"}
    assert all(u.reason.startswith("rejected by risk") for u in rep.unticketed)


def test_report_is_json_serializable_and_records_config():
    cfg = _cfg()
    _, rep = _eval([_idea()], cfg, account=risk.AccountState(equity=1e5))
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["config_hash"] == cfg.config_hash and d["environment"] == "research"
    assert d["decisions"][0]["checks"][0]["name"]


def test_apply_decisions_rejects_failed_and_respects_manual_approval():
    ideas = [_idea("A", "B"), _idea("C", "D", qty_x=300.0, qty_y=-150.0)]
    plan, rep = _eval(ideas)
    manual = risk.apply_decisions(rep, plan.tickets, _cfg(require_manual_approval=True))
    assert manual[0].status == OrderStatus.PROPOSED and "awaiting manual approval" in manual[0].notes[-1]
    assert manual[1].status == OrderStatus.REJECTED
    assert "pair_notional" in manual[1].history[-1].reason

    auto = risk.apply_decisions(rep, plan.tickets, _cfg(require_manual_approval=False))
    assert auto[0].status == OrderStatus.APPROVED and auto[1].status == OrderStatus.REJECTED
    # rejected tickets are terminal
    with pytest.raises(orders.InvalidTransition):
        auto[1].approve()
