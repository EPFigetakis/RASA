# tests/test_monitoring.py
import json
import sys
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

from core import daily_run, monitoring, orders, risk
from core.broker_paper import PaperBroker, PaperBrokerConfig
from core.orders import OrderLeg, OrderSide, OrderTicket, OrderType

TODAY = date(2026, 11, 11)
LAST_BAR = date(2026, 11, 10)
PX = {"X": 50.0, "Y": 100.0}
CFG = risk.RiskConfig()


def _broker(tmp_path, **kw):
    base = dict(commission_per_share=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0)
    return PaperBroker(tmp_path / "p.db", PaperBrokerConfig(**{**base, **kw}))


def _ticket(legs=None):
    legs = legs or [OrderLeg("X", OrderSide.BUY, 20.0, reference_price=50.0),
                    OrderLeg("Y", OrderSide.SELL, 10.0, reference_price=100.0)]
    return OrderTicket.create(legs, "2026-11-10", idea_id="I", pair="X/Y").approve()


def _snap(broker, runs=(), price_last_bar=LAST_BAR, today=TODAY, cfg=CFG):
    return monitoring.build_snapshot(broker, cfg, list(runs), price_last_bar, today)


def _codes(snap):
    return {a.code for a in snap.alerts}


def _run_record(status="ok", started="2026-11-11T08:00:00+00:00", as_of="2026-11-10", tickets=(), decisions=(), unticketed=()):
    return daily_run.RunRecord(
        run_id="Rtest", status=status, as_of=as_of, started_at=started, environment="paper",
        data_age_business_days=0, tickets=[t.to_dict() for t in tickets],
        risk_report={"decisions": list(decisions), "unticketed": list(unticketed)})


# ------------------------------- healthy / basic -------------------------------

def test_fresh_account_has_only_a_no_runs_warning(tmp_path):
    snap = _snap(_broker(tmp_path))
    assert snap.equity == 100_000 and snap.drawdown == 0 and snap.positions.empty
    assert _codes(snap) == {"no_runs"} and monitoring.exit_code(snap.alerts) == 1


def test_healthy_account_with_a_recent_run_is_clean(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    snap = _snap(b, [_run_record()])
    assert snap.alerts == [] and monitoring.exit_code(snap.alerts) == 0
    assert snap.broker_lag_bdays == 0 and snap.price_age_bdays == 0
    assert snap.gross == pytest.approx(2000) and snap.gross_utilization == pytest.approx(2000 / 100_000)
    assert set(snap.positions["symbol"]) == {"X", "Y"}
    assert snap.positions.set_index("symbol").loc["X", "unrealized_pnl"] == pytest.approx(0)


def test_pnl_and_drawdown_from_equity_history(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    b.process_bar({"X": 50.0, "Y": 110.0}, "2026-11-11")              # short Y loses 100
    snap = _snap(b, [_run_record()], price_last_bar=date(2026, 11, 11), today=date(2026, 11, 12))
    assert snap.equity == pytest.approx(99_900) and snap.daily_pnl == pytest.approx(-100)
    assert snap.total_pnl == pytest.approx(-100)
    assert snap.peak_equity == 100_000 and snap.drawdown == pytest.approx(0.001)


# ------------------------------- alerts -------------------------------

def test_stale_prices_and_missing_price_data(tmp_path):
    b = _broker(tmp_path)
    s = _snap(b, [_run_record()], price_last_bar=date(2026, 11, 2))
    assert "stale_prices" in _codes(s) and monitoring.exit_code(s.alerts) == 2
    assert "no_price_data" in _codes(_snap(b, [_run_record()], price_last_bar=None))


def test_run_overdue_aborted_and_kill_statuses(tmp_path):
    b = _broker(tmp_path)
    assert "run_overdue" in _codes(_snap(b, [_run_record(started="2026-11-05T08:00:00+00:00")]))
    assert "run_aborted_stale" in _codes(_snap(b, [_run_record(status=daily_run.STATUS_STALE)]))
    assert "run_kill_switch" in _codes(_snap(b, [_run_record(status=daily_run.STATUS_KILL)]))
    unt = [{"subject": "X/Z", "reason": "rejected by risk: symbol overlap", "idea_id": None}]
    assert "unticketed" in _codes(_snap(b, [_run_record(unticketed=unt)]))


def test_broker_behind_price_data(tmp_path):
    b = _broker(tmp_path)
    b.process_bar(PX, "2026-11-04")
    assert "broker_behind" in _codes(_snap(b, [_run_record()]))


def test_kill_switch_and_drawdown_warning(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    b.process_bar({"X": 50.0, "Y": 110.0}, "2026-11-11")              # -0.1% equity
    tight = risk.RiskConfig(max_drawdown_pct=0.0005, max_daily_loss_pct=0.5)
    s = _snap(b, [_run_record()], price_last_bar=date(2026, 11, 11), today=date(2026, 11, 12), cfg=tight)
    assert "kill_switch" in _codes(s)
    near = risk.RiskConfig(max_drawdown_pct=0.0012, max_daily_loss_pct=0.5)
    s = _snap(b, [_run_record()], price_last_bar=date(2026, 11, 11), today=date(2026, 11, 12), cfg=near)
    assert "drawdown_near_limit" in _codes(s) and "kill_switch" not in _codes(s)


def test_exposure_limit_alerts(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    near = risk.RiskConfig(max_gross_exposure_pct=0.0225)             # 2000 / 2250 = 89%
    assert "gross_near_limit" in _codes(_snap(b, [_run_record()], cfg=near))
    over = risk.RiskConfig(max_gross_exposure_pct=0.015)
    assert "gross_over_limit" in _codes(_snap(b, [_run_record()], cfg=over))


def test_one_legged_pair_is_critical_until_closed(tmp_path):
    legs = [OrderLeg("X", OrderSide.BUY, 10.0, reference_price=50.0),
            OrderLeg("Y", OrderSide.SELL, 5.0, OrderType.LIMIT, limit_price=150.0)]
    b = _broker(tmp_path)
    b.submit_ticket(_ticket(legs)); b.process_bar(PX, "2026-11-10")   # Y never fills; DAY expires
    s = _snap(b, [_run_record()])
    assert "one_legged" in _codes(s) and monitoring.exit_code(s.alerts) == 2
    close = OrderTicket.create([OrderLeg("X", OrderSide.SELL, 10.0, reference_price=50.0)],
                               "2026-11-11", purpose="close_out").approve()
    b.submit_ticket(close); b.process_bar(PX, "2026-11-11")
    assert "one_legged" not in _codes(_snap(b, [_run_record()], price_last_bar=date(2026, 11, 11), today=date(2026, 11, 12)))


def test_reconciliation_break_detected(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    b.portfolio.apply_fill("X", 3.0, 50.0)                            # position changed with no ticket
    s = _snap(b, [_run_record()])
    assert "reconciliation_break" in _codes(s)
    assert [(x.symbol, x.difference) for x in s.reconciliation_breaks] == [("X", 3.0)]


def test_stuck_working_ticket(tmp_path):
    b = _broker(tmp_path)
    b.process_bar(PX, "2026-11-17")
    snap = _snap(b, [_run_record()], price_last_bar=date(2026, 11, 17), today=date(2026, 11, 18))
    leg = [OrderLeg("X", OrderSide.BUY, 10.0, OrderType.LIMIT, limit_price=1.0,
                    time_in_force=orders.TimeInForce.GTC)]
    t = (OrderTicket.create(leg, "2026-11-10", at="2026-11-10T20:00:00+00:00")
         .approve(at="2026-11-10T20:01:00+00:00").submit(at="2026-11-10T20:02:00+00:00"))
    snap.working_tickets = [t]                                         # submitted 5 business days ago
    alerts = monitoring.compute_alerts(snap, CFG, [t])
    assert "stuck_ticket" in {a.code for a in alerts}
    snap.working_tickets = []                                          # nothing working any more
    assert "stuck_ticket" not in {a.code for a in monitoring.compute_alerts(snap, CFG, [])}


def test_pending_proposals_detected_and_escalate_with_age(tmp_path):
    b = _broker(tmp_path)
    t = OrderTicket.create([OrderLeg("X", OrderSide.BUY, 1.0, reference_price=50.0)], "2026-11-10",
                           idea_id="I", pair="X/Y")
    dec = [{"ticket_id": t.ticket_id, "pair": "X/Y", "approved": True, "reduces_risk": False, "checks": []}]
    rec = _run_record(tickets=[t], decisions=dec)
    s = _snap(b, [rec])
    assert [x.ticket_id for x in s.pending_proposals] == [t.ticket_id]
    assert [a.severity for a in s.alerts if a.code == "pending_proposals"] == ["info"]
    old = _snap(b, [rec], today=date(2026, 11, 16))
    assert [a.severity for a in old.alerts if a.code == "pending_proposals"] == ["warning"]
    b.submit_ticket(t.approve())                                       # once submitted, no longer pending
    assert _snap(b, [rec]).pending_proposals == []


def test_alerts_sorted_by_severity_and_exit_codes():
    A = monitoring.Alert
    assert monitoring.exit_code([]) == 0
    assert monitoring.exit_code([A("info", "a", "")]) == 0
    assert monitoring.exit_code([A("info", "a", ""), A("warning", "b", "")]) == 1
    assert monitoring.exit_code([A("warning", "b", ""), A("critical", "c", "")]) == 2


def test_expected_positions_from_tickets(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10")
    assert monitoring.expected_positions_from_tickets(b.list_tickets()) == {"X": 20.0, "Y": -10.0}


# ------------------------------- CLI & UI -------------------------------

def test_check_health_cli_exit_codes_and_json(tmp_path, monkeypatch, capsys):
    import check_health
    from core import prices

    b = _broker(tmp_path)
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10"); b.close()
    monkeypatch.setattr(prices, "get_cache_status_from_config", lambda: {"last_bar_date": "2026-11-02"})
    monkeypatch.setattr(daily_run, "list_run_records", lambda: [])
    assert check_health.main(["--db", str(tmp_path / "p.db"), "--json"]) == 2     # stale prices
    out = json.loads(capsys.readouterr().out)
    assert out["exit_code"] == 2 and any(a["code"] == "stale_prices" for a in out["alerts"])
    assert check_health.main(["--db", str(tmp_path / "missing.db")]) == 3


def test_monitor_view_renders_headless(tmp_path, monkeypatch):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    # The view opens state_dir()/rasa_paper.db
    b = PaperBroker(tmp_path / "rasa_paper.db", PaperBrokerConfig(
        commission_per_share=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0))
    b.submit_ticket(_ticket()); b.process_bar(PX, "2026-11-10"); b.close()

    from core import paths, prices
    monkeypatch.setattr(paths, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(prices, "get_cache_status_from_config", lambda: {"last_bar_date": "2026-11-10"})
    monkeypatch.setenv("RUNNING_TESTS", "1")

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "import app\n"
        "app.run_paper_monitor_ui()\n"
    )
    at = streamlit_testing.AppTest.from_string(script)
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert any("Paper trading monitor" in m.value for m in at.markdown)
    assert any(m.label == "Equity" for m in at.metric), ([m.label for m in at.metric], [e.value for e in at.error], [w.value for w in at.warning])
