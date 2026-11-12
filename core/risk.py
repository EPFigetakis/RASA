"""
core.risk

Pre-trade risk gates for proposed order tickets.

Every ticket from core.orders.plan_orders gets a RiskDecision: approved or rejected,
with the result of every check recorded (name, pass/fail, human-readable detail) so a
rejection is always explainable. Limits come from config/trading_config.json.

How it evaluates
- Tickets are evaluated in the order given (plan order = strongest signal first) against
  a running *projected* portfolio: current holdings plus every ticket accepted so far plus
  the ticket under test. A ticket that would breach a portfolio limit is rejected and does
  not consume capacity, so later, smaller tickets can still fit.
- Risk-reducing tickets (closing or shrinking positions, never growing any symbol or the
  gross exposure) skip exposure / pair-count / price / data-age / kill-switch checks:
  the system must always be able to get out. Order-type validity still applies.
- Ideas the planner could not ticket (symbol overlap, missing close-out price) are
  carried into the report as rejections. Overlapping pairs are never netted automatically.
- A daily-loss or drawdown breach (from the AccountState you pass) trips a kill switch
  that rejects every risk-increasing ticket.

Not covered (no data in the system yet): sector/correlation concentration, borrow
availability, liquidity/volume. min_price is a crude penny-stock guard, not a liquidity check.

Typical usage:

    cfg = risk.load_risk_config()
    plan = orders.plan_orders(batch.ideas, positions)
    report = risk.evaluate_plan(plan, cfg, prices=last_prices,
                                account=risk.AccountState(equity=100_000, ...),
                                data_age_business_days=batch.data_age_business_days)
    tickets = risk.apply_decisions(report, plan.tickets, cfg)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    import core.orders as orders
    import core.paths as paths
except ModuleNotFoundError:  # run from inside core/
    import orders  # type: ignore
    import paths  # type: ignore

ENVIRONMENTS = ("research", "paper", "live")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """
    Hard limits. Percentages are fractions of `capital_base` (0.10 = 10%).

    environment               research | paper | live (recorded in reports; live must be
                              chosen explicitly and still requires manual approval below
                              unless that is deliberately switched off)
    capital_base              dollars the limits are measured against
    max_gross_exposure_pct    total |long| + |short| dollars after trading
    max_net_exposure_pct      |long - short| dollars after trading
    max_concurrent_pairs      open pairs, including new ones in this batch
    max_symbol_exposure_pct   one symbol's |position| dollars
    max_pair_notional         gross dollars (both legs) a single pair may hold after trading
    min_price                 reject legs priced below this
    max_data_age_business_days  reject risk-increasing tickets on older prices
    allowed_order_types       subset of MARKET / LIMIT
    allow_fractional_shares   if False, risk-increasing legs must be whole shares
    max_daily_loss_pct        kill switch: loss vs day-start equity
    max_drawdown_pct          kill switch: drawdown vs peak equity
    require_manual_approval   passing tickets stay PROPOSED for a human to approve
    """

    environment: str = "research"
    capital_base: float = 100_000.0
    max_gross_exposure_pct: float = 1.0
    max_net_exposure_pct: float = 0.10
    max_concurrent_pairs: int = 10
    max_symbol_exposure_pct: float = 0.10
    max_pair_notional: float = 10_000.0
    min_price: float = 5.0
    max_data_age_business_days: int = 2
    allowed_order_types: tuple = ("MARKET", "LIMIT")
    allow_fractional_shares: bool = False
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    require_manual_approval: bool = True

    def __post_init__(self) -> None:
        if self.environment not in ENVIRONMENTS:
            raise ValueError(f"environment must be one of {ENVIRONMENTS}, got {self.environment!r}")
        positive = ("capital_base", "max_gross_exposure_pct", "max_symbol_exposure_pct",
                    "max_pair_notional", "max_daily_loss_pct", "max_drawdown_pct")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("max_net_exposure_pct", "min_price", "max_data_age_business_days"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_concurrent_pairs < 1:
            raise ValueError("max_concurrent_pairs must be >= 1")
        valid_types = {t.value for t in orders.OrderType}
        object.__setattr__(self, "allowed_order_types", tuple(str(t).upper() for t in self.allowed_order_types))
        if not self.allowed_order_types or not set(self.allowed_order_types) <= valid_types:
            raise ValueError(f"allowed_order_types must be a non-empty subset of {sorted(valid_types)}")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def default_config_path() -> Path:
    return paths.PROJECT_ROOT / "config" / "trading_config.json"


def load_risk_config(path: Optional[Path] = None) -> RiskConfig:
    """Load and validate config/trading_config.json. Unknown keys are an error (typos matter here)."""
    path = Path(path) if path else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Risk config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(RiskConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown keys in {path.name}: {unknown}")
    if "allowed_order_types" in raw:
        raw["allowed_order_types"] = tuple(raw["allowed_order_types"])
    return RiskConfig(**raw)


# ---------------------------------------------------------------------------
# Inputs & outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountState:
    """Account-level numbers needed by the kill-switch and pair-count checks."""

    equity: float
    day_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None
    open_pairs: int = 0


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RiskDecision:
    ticket_id: str
    pair: Optional[str]
    approved: bool
    reduces_risk: bool
    checks: List[CheckResult]

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def reason(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures) or "all checks passed"


@dataclass
class PortfolioSnapshot:
    gross: float
    net: float
    n_pairs: int


@dataclass
class RiskReport:
    decisions: List[RiskDecision]
    unticketed: List[orders.PlanIssue]
    before: PortfolioSnapshot
    after: PortfolioSnapshot             # after accepted tickets only
    kill_switch: bool
    kill_switch_reasons: List[str]
    config_hash: str
    environment: str
    as_of: str

    def decision_for(self, ticket_id: str) -> RiskDecision:
        return next(d for d in self.decisions if d.ticket_id == ticket_id)

    @property
    def approved_ids(self) -> List[str]:
        return [d.ticket_id for d in self.decisions if d.approved]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "environment": self.environment,
            "config_hash": self.config_hash,
            "kill_switch": self.kill_switch,
            "kill_switch_reasons": self.kill_switch_reasons,
            "before": asdict(self.before),
            "after": asdict(self.after),
            "decisions": [
                {
                    "ticket_id": d.ticket_id, "pair": d.pair, "approved": d.approved,
                    "reduces_risk": d.reduces_risk,
                    "checks": [asdict(c) for c in d.checks],
                }
                for d in self.decisions
            ],
            "unticketed": [asdict(u) for u in self.unticketed],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def kill_switch_status(cfg: RiskConfig, account: Optional[AccountState]) -> List[str]:
    """Reasons the kill switch is tripped (empty list = not tripped)."""
    reasons: List[str] = []
    if account is None:
        return reasons
    if account.day_start_equity and account.day_start_equity > 0:
        loss = (account.day_start_equity - account.equity) / account.day_start_equity
        if loss > cfg.max_daily_loss_pct:
            reasons.append(f"daily loss {loss:.2%} exceeds {cfg.max_daily_loss_pct:.2%}")
    if account.peak_equity and account.peak_equity > 0:
        dd = (account.peak_equity - account.equity) / account.peak_equity
        if dd > cfg.max_drawdown_pct:
            reasons.append(f"drawdown {dd:.2%} exceeds {cfg.max_drawdown_pct:.2%}")
    return reasons


def _exposures(positions: Mapping[str, float], prices: Mapping[str, float]) -> tuple:
    """(gross, net, symbols_without_price) in dollars."""
    gross = net = 0.0
    missing: List[str] = []
    for sym, qty in positions.items():
        if abs(qty) <= orders.QTY_TOL:
            continue
        px = prices.get(sym)
        if px is None:
            missing.append(sym)
            continue
        gross += abs(qty * px)
        net += qty * px
    return gross, net, missing


def _apply_ticket(positions: Mapping[str, float], ticket: orders.OrderTicket) -> Dict[str, float]:
    out = dict(positions)
    for leg in ticket.legs:
        out[leg.symbol] = out.get(leg.symbol, 0.0) + leg.signed_quantity
    return {k: v for k, v in out.items() if abs(v) > orders.QTY_TOL}


def evaluate_plan(
    plan: orders.OrderPlan,
    cfg: RiskConfig,
    prices: Optional[Mapping[str, float]] = None,
    account: Optional[AccountState] = None,
    data_age_business_days: Optional[int] = None,
) -> RiskReport:
    """
    Run every risk check on every ticket in `plan`.

    `prices` supplies marks for held symbols that are not legs of a ticket; leg
    reference prices are used for ticket symbols. A held symbol with no price
    makes risk-increasing tickets fail closed ("price_available").
    """
    current = dict(plan.current_positions)
    marks: Dict[str, float] = {k: float(v) for k, v in (prices or {}).items()}
    for t in plan.tickets:
        for leg in t.legs:
            if leg.reference_price is not None:
                marks.setdefault(leg.symbol, leg.reference_price)

    capital = cfg.capital_base
    ks_reasons = kill_switch_status(cfg, account)
    open_pairs = account.open_pairs if account else 0

    g0, n0, _ = _exposures(current, marks)
    before = PortfolioSnapshot(g0, n0, open_pairs)

    accepted = dict(current)
    accepted_new_pairs = 0
    decisions: List[RiskDecision] = []

    for t in plan.tickets:
        checks: List[CheckResult] = []

        def add(name: str, passed: bool, detail: str = "") -> None:
            checks.append(CheckResult(name, bool(passed), "" if passed else detail))

        # --- always-on validity ---
        bad_types = sorted({l.order_type.value for l in t.legs} - set(cfg.allowed_order_types))
        add("order_type_allowed", not bad_types, f"{', '.join(bad_types)} not allowed")

        projected = _apply_ticket(accepted, t)
        gross_before, _, _ = _exposures(accepted, marks)
        gross_after, net_after, missing = _exposures(projected, marks)
        symbol_grew = any(abs(projected.get(l.symbol, 0.0)) > abs(accepted.get(l.symbol, 0.0)) + orders.QTY_TOL
                          for l in t.legs)
        reduces = (not symbol_grew) and not missing and gross_after <= gross_before + 1e-6

        if not reduces:
            # --- kill switch ---
            add("kill_switch", not ks_reasons, "; ".join(ks_reasons))

            # --- data freshness ---
            if data_age_business_days is not None:
                add("data_fresh", data_age_business_days <= cfg.max_data_age_business_days,
                    f"prices are {data_age_business_days} business days old "
                    f"(max {cfg.max_data_age_business_days})")

            # --- per-leg ---
            low = [l.symbol for l in t.legs
                   if l.reference_price is not None and l.reference_price < cfg.min_price]
            add("min_price", not low, f"{', '.join(low)} priced below {cfg.min_price}")
            if not cfg.allow_fractional_shares:
                frac = [l.symbol for l in t.legs if abs(l.quantity - round(l.quantity)) > 1e-9]
                add("whole_shares", not frac, f"fractional quantity for {', '.join(frac)}")

            # --- pricing completeness ---
            add("price_available", not missing,
                f"no price to value held position(s): {', '.join(sorted(missing))}")

            if not missing:
                # --- per-symbol ---
                limit_sym = cfg.max_symbol_exposure_pct * capital
                over = [f"{l.symbol} ${abs(projected.get(l.symbol, 0.0)) * marks[l.symbol]:,.0f}"
                        for l in t.legs
                        if abs(projected.get(l.symbol, 0.0)) * marks[l.symbol] > limit_sym + 1e-6]
                add("symbol_exposure", not over, f"{', '.join(over)} exceeds ${limit_sym:,.0f}")

                # --- per-pair ---
                if t.purpose == "idea":
                    pair_gross = sum(abs(projected.get(l.symbol, 0.0)) * marks[l.symbol] for l in t.legs)
                    add("pair_notional", pair_gross <= cfg.max_pair_notional + 1e-6,
                        f"pair holds ${pair_gross:,.0f} gross after trade (max ${cfg.max_pair_notional:,.0f})")

                    opens_new = any(abs(accepted.get(l.symbol, 0.0)) <= orders.QTY_TOL for l in t.legs)
                    n_pairs = open_pairs + accepted_new_pairs + (1 if opens_new else 0)
                    add("concurrent_pairs", n_pairs <= cfg.max_concurrent_pairs,
                        f"would be {n_pairs} open pairs (max {cfg.max_concurrent_pairs})")

                # --- portfolio ---
                add("gross_exposure", gross_after <= cfg.max_gross_exposure_pct * capital + 1e-6,
                    f"gross ${gross_after:,.0f} exceeds ${cfg.max_gross_exposure_pct * capital:,.0f}")
                add("net_exposure", abs(net_after) <= cfg.max_net_exposure_pct * capital + 1e-6,
                    f"|net| ${abs(net_after):,.0f} exceeds ${cfg.max_net_exposure_pct * capital:,.0f}")

        approved = all(c.passed for c in checks)
        decisions.append(RiskDecision(t.ticket_id, t.pair, approved, reduces, checks))
        if approved:
            if t.purpose == "idea" and not reduces and any(
                abs(accepted.get(l.symbol, 0.0)) <= orders.QTY_TOL for l in t.legs
            ):
                accepted_new_pairs += 1
            accepted = projected

    g1, n1, _ = _exposures(accepted, marks)
    after = PortfolioSnapshot(g1, n1, open_pairs + accepted_new_pairs)

    unticketed = [
        orders.PlanIssue(c.subject, f"rejected by risk: {c.reason}", c.idea_id)
        for c in plan.conflicts
    ]

    return RiskReport(
        decisions=decisions, unticketed=unticketed, before=before, after=after,
        kill_switch=bool(ks_reasons), kill_switch_reasons=ks_reasons,
        config_hash=cfg.config_hash, environment=cfg.environment, as_of=plan.as_of,
    )


def apply_decisions(
    report: RiskReport,
    tickets: Sequence[orders.OrderTicket],
    cfg: RiskConfig,
    at: Optional[str] = None,
) -> List[orders.OrderTicket]:
    """
    Move tickets along according to their risk decisions.

    - Failed -> REJECTED, with the failing checks as the reason.
    - Passed and `cfg.require_manual_approval` -> stays PROPOSED, with a note
      recording that risk passed (a human still has to approve).
    - Passed and manual approval off -> APPROVED automatically.
    """
    out: List[orders.OrderTicket] = []
    for t in tickets:
        d = report.decision_for(t.ticket_id)
        if not d.approved:
            out.append(t.reject(d.reason, at))
        elif cfg.require_manual_approval:
            out.append(replace(t, notes=t.notes + (f"risk passed ({report.config_hash}); awaiting manual approval",)))
        else:
            out.append(t.approve(f"risk passed ({report.config_hash})", at))
    return out
