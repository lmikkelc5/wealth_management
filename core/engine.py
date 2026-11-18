"""Portfolio simulation engine: one client, day by day, at closing prices.

Conventions (all trades happen at the day's close, no commissions, fractional shares allowed):
  start day   initial capital is deposited and invested to the first plan's targets
  each day    1. interest on cash (annual rate / 252)
              2. distributions on the ex-date: reinvested in the same holding, or kept as cash
              3. first trading day of a quarter: advisory fee (annual % / 4 of prior close value)
              4. first trading day of a month: contribution, invested at target weights
              5. first trading day of a month (from withdrawal_start): withdrawal, paid from cash
                 first, then by selling (pro-rata or overweight-first)
              6. rebalance to targets if: a new plan version takes effect today, or the
                 rebalance rule fires (annual / quarterly / drift beyond the band)
Prices are split-adjusted closes, so splits never need handling here (share counts are in
today's share units).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH = "CASH"
EPS = 1e-9


class SimError(Exception):
    """Simulation can't run (bad plan, missing data)."""


class NotStartedYet(SimError):
    """Start date is after the latest available close: nothing to simulate yet."""


@dataclass
class PlanVersion:
    effective_date: pd.Timestamp
    weights: dict[str, float]          # fractions; anything not allocated (or under "CASH") is cash
    note: str = ""


@dataclass
class SimConfig:
    start_date: pd.Timestamp
    initial_capital: float
    plans: list[PlanVersion]
    monthly_contribution: float = 0.0
    contribution_end: pd.Timestamp | None = None
    monthly_withdrawal: float = 0.0
    withdrawal_start: pd.Timestamp | None = None
    withdrawal_growth: float = 0.0     # annual step-up of the withdrawal, e.g. 0.03
    annual_fee: float = 0.0            # e.g. 0.01 for 1%/yr, charged quarterly
    dividend_policy: str = "reinvest"  # "reinvest" | "cash"
    rebalance_rule: str = "annual"     # "none" | "annual" | "quarterly" | "threshold"
    rebalance_band: float = 0.05       # absolute drift that triggers a threshold rebalance
    withdrawal_rule: str = "overweight_first"  # "pro_rata" | "overweight_first"
    end_date: pd.Timestamp | None = None


@dataclass
class SimResult:
    daily: pd.DataFrame                 # value, cash, invested, deposit, withdrawal, withdrawal_requested,
                                        # fee, dividends, interest, flow
    holdings_value: pd.DataFrame        # date x ticker market value
    shares: pd.Series                   # final shares
    ledger: pd.DataFrame
    tickers: list[str]
    applied_plans: list[tuple[pd.Timestamp, PlanVersion]]
    pending_plans: list[PlanVersion]
    depleted_on: pd.Timestamp | None = None
    warnings: list[str] = field(default_factory=list)


def _targets(plan: PlanVersion, tickers: list[str]) -> np.ndarray:
    return np.array([max(float(plan.weights.get(t, 0.0)), 0.0) for t in tickers])


def validate_plan_weights(weights: dict[str, float]) -> None:
    if any(w < 0 for w in weights.values()):
        raise SimError("Weights can't be negative.")
    total = sum(weights.values())
    if total > 1 + 1e-6:
        raise SimError(f"Weights add up to {total:.1%}; they can't exceed 100%.")


def simulate(config: SimConfig, prices: pd.DataFrame, distributions: pd.DataFrame,
             cash_rate: pd.Series | None = None) -> SimResult:
    if not config.plans:
        raise SimError("Client has no plan.")
    if config.initial_capital <= 0:
        raise SimError("Starting capital must be positive.")
    plans = sorted(config.plans, key=lambda p: pd.Timestamp(p.effective_date))
    for p in plans:
        validate_plan_weights(p.weights)

    tickers = sorted({t for p in plans for t, w in p.weights.items() if t != CASH and w > 0})
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise SimError(f"No price data for: {', '.join(missing)}")

    start = pd.Timestamp(config.start_date)
    px = prices[tickers] if tickers else pd.DataFrame(index=prices.index)
    cal = px.index[px.notna().any(axis=1)] if tickers else prices.index
    cal = cal[cal >= start]
    if config.end_date is not None:
        cal = cal[cal <= pd.Timestamp(config.end_date)]
    if len(cal) == 0:
        raise NotStartedYet(f"No closing prices on or after {start.date()} yet. "
                            "The client will be invested at the next available close.")

    # Map plan versions to the trading day they take effect.
    initial = [p for p in plans if pd.Timestamp(p.effective_date) <= start]
    if not initial:
        raise SimError("The first plan must take effect on the client's start date.")
    schedule: dict[int, PlanVersion] = {0: initial[-1]}
    applied = [(cal[0], initial[-1])]
    pending = []
    for p in plans:
        eff = pd.Timestamp(p.effective_date)
        if eff <= start:
            continue
        pos = int(cal.searchsorted(eff))
        if pos >= len(cal):
            pending.append(p)
            continue
        schedule[pos] = p
        applied = [a for a in applied if a[0] != cal[pos]] + [(cal[pos], p)]

    P = px.reindex(cal).to_numpy(dtype=float) if tickers else np.zeros((len(cal), 0))
    D = (distributions.reindex(index=cal, columns=tickers).fillna(0.0).to_numpy(dtype=float)
         if tickers else np.zeros((len(cal), 0)))
    R = (cash_rate.reindex(cal).ffill().fillna(0.0).to_numpy(dtype=float)
         if cash_rate is not None else np.zeros(len(cal)))

    # Every ticker in a plan needs a price on the day that plan takes effect.
    for pos, plan in schedule.items():
        for j, t in enumerate(tickers):
            if plan.weights.get(t, 0) > 0 and not np.isfinite(P[pos, j]):
                first = prices[t].first_valid_index()
                when = f"its data starts {first.date()}" if first is not None else "no data at all"
                raise SimError(f"{t} has no price on {cal[pos].date()} ({when}). "
                               f"Pick a different holding or a later date.")

    k = len(tickers)
    shares = np.zeros(k)
    cash = 0.0
    ledger: list[dict] = []
    hv = np.zeros((len(cal), k))
    out = {c: np.zeros(len(cal)) for c in ("value", "cash", "deposit", "withdrawal", "withdrawal_requested",
                                            "fee", "dividends", "interest", "flow")}
    targets = _targets(schedule[0], tickers)
    w_start = pd.Timestamp(config.withdrawal_start) if config.withdrawal_start is not None else start
    c_end = pd.Timestamp(config.contribution_end) if config.contribution_end is not None else None
    depleted_on = None

    def log(d, typ, ticker="", shares_=np.nan, price=np.nan, amount=0.0, note=""):
        ledger.append({"date": d, "type": typ, "ticker": ticker, "shares": shares_, "price": price,
                       "amount": amount, "note": note})

    def invested_values(p):
        return np.where(shares > 0, shares * np.nan_to_num(p), 0.0)

    def rebalance(d, p, reason):
        nonlocal shares, cash
        V = cash + invested_values(p).sum()
        if V <= EPS:
            return
        new = np.zeros(k)
        mask = targets > 0
        new[mask] = targets[mask] * V / p[mask]
        for j in range(k):
            delta = new[j] - shares[j]
            amt = abs(delta) * np.nan_to_num(p[j])
            if amt > 0.01:
                log(d, "BUY" if delta > 0 else "SELL", tickers[j], abs(delta), p[j], amt, reason)
        shares = new
        cash = V - (shares * np.nan_to_num(p)).sum()

    def raise_cash(d, p, need, reason):
        """Sell holdings to raise `need` dollars of cash. Returns amount raised."""
        nonlocal shares, cash
        inv = invested_values(p)
        total = inv.sum()
        if need <= EPS or total <= EPS:
            return 0.0
        amount = min(need, total)
        if config.withdrawal_rule == "overweight_first":
            after = cash + total - need
            excess = np.maximum(inv - targets * max(after, 0.0), 0.0)
            if excess.sum() >= amount:
                sell = excess / excess.sum() * amount
            else:
                rest = inv - excess
                sell = excess + (rest / rest.sum() * (amount - excess.sum()) if rest.sum() > EPS else 0)
        else:
            sell = inv / total * amount
        sell = np.minimum(sell, inv)
        for j in range(k):
            if sell[j] > 0.01:
                q = sell[j] / p[j]
                log(d, "SELL", tickers[j], q, p[j], sell[j], reason)
                shares[j] -= q
        shares = np.where(shares < 1e-12, 0.0, shares)
        raised = sell.sum()
        cash += raised
        return raised

    for i, d in enumerate(cal):
        p = P[i]
        if i == 0:
            cash = float(config.initial_capital)
            out["deposit"][0] = cash
            out["flow"][0] = cash
            log(d, "DEPOSIT", amount=cash, note="Initial investment")
            log(d, "PLAN", note=f"Plan in effect: {_describe(schedule[0])}")
            rebalance(d, p, "initial investment")
        else:
            prev = cal[i - 1]
            new_month = d.month != prev.month or d.year != prev.year
            new_quarter = new_month and d.month in (1, 4, 7, 10)
            new_year = d.year != prev.year

            # 1. interest on cash
            if cash > 0 and R[i] > 0:
                interest = cash * R[i] / 252
                cash += interest
                out["interest"][i] = interest

            # 2. distributions (ex-date)
            pay = D[i] * shares
            if pay.sum() > 0:
                out["dividends"][i] = pay.sum()
                for j in np.nonzero(pay > 0)[0]:
                    log(d, "DIVIDEND", tickers[j], shares[j], D[i, j], pay[j],
                        "reinvested" if config.dividend_policy == "reinvest" else "to cash")
                    if config.dividend_policy == "reinvest" and np.isfinite(p[j]) and p[j] > 0:
                        q = pay[j] / p[j]
                        shares[j] += q
                        log(d, "BUY", tickers[j], q, p[j], pay[j], "dividend reinvestment")
                    else:
                        cash += pay[j]

            # 3. advisory fee
            if new_quarter and config.annual_fee > 0 and depleted_on is None:
                fee = config.annual_fee / 4 * out["value"][i - 1]
                if fee > cash:
                    raise_cash(d, p, fee - cash, "raise cash for fee")
                fee = min(fee, cash)
                cash -= fee
                out["fee"][i] = fee
                if fee > 0:
                    log(d, "FEE", amount=fee, note=f"{config.annual_fee:.2%}/yr, charged quarterly")

            # 4. contribution
            if (new_month and config.monthly_contribution > 0 and depleted_on is None
                    and (c_end is None or d <= c_end)):
                c = float(config.monthly_contribution)
                cash += c
                out["deposit"][i] += c
                out["flow"][i] += c
                log(d, "DEPOSIT", amount=c, note="Monthly contribution")
                for j in range(k):
                    if targets[j] > 0:
                        amt = c * targets[j]
                        q = amt / p[j]
                        shares[j] += q
                        cash -= amt
                        log(d, "BUY", tickers[j], q, p[j], amt, "invest contribution")

            # 5. withdrawal
            if new_month and config.monthly_withdrawal > 0 and d >= w_start:
                years = int((d - w_start).days // 365.25)
                req = float(config.monthly_withdrawal) * (1 + config.withdrawal_growth) ** years
                out["withdrawal_requested"][i] = req
                if depleted_on is None:
                    if req > cash:
                        raise_cash(d, p, req - cash, "raise cash for withdrawal")
                    paid = min(req, cash)
                    cash -= paid
                    out["withdrawal"][i] = paid
                    out["flow"][i] -= paid
                    note = "Monthly income" + ("" if paid >= req - 0.01 else f" (SHORT by ${req - paid:,.2f})")
                    log(d, "WITHDRAWAL", amount=paid, note=note)

            # 6. rebalance
            if i in schedule:
                targets = _targets(schedule[i], tickers)
                log(d, "PLAN", note=f"New plan: {_describe(schedule[i])}"
                    + (f" | {schedule[i].note}" if schedule[i].note else ""))
                rebalance(d, p, "plan change")
            elif depleted_on is None:
                rule = config.rebalance_rule
                if (rule == "annual" and new_year) or (rule == "quarterly" and new_quarter):
                    rebalance(d, p, f"{rule} rebalance")
                elif rule == "threshold":
                    inv = invested_values(p)
                    V = cash + inv.sum()
                    if V > EPS:
                        drift = max(np.abs(inv / V - targets).max(initial=0.0),
                                    abs(cash / V - (1 - targets.sum())))
                        if drift > config.rebalance_band:
                            rebalance(d, p, f"drift {drift:.1%} > band")

        hv[i] = invested_values(p)
        cash = 0.0 if abs(cash) < 1e-9 else cash
        out["cash"][i] = cash
        out["value"][i] = cash + hv[i].sum()
        if depleted_on is None and i > 0 and out["value"][i] < 0.01 and out["withdrawal_requested"][i] > 0:
            depleted_on = d
            log(d, "NOTE", note="Portfolio depleted: withdrawals can no longer be paid.")

    daily = pd.DataFrame(out, index=cal)
    daily["invested"] = daily["value"] - daily["cash"]
    daily.index.name = "date"

    # Monthly interest rows instead of one per day.
    led = pd.DataFrame(ledger, columns=["date", "type", "ticker", "shares", "price", "amount", "note"])
    monthly_int = daily["interest"].groupby(daily.index.to_period("M")).agg(["sum"])
    last_day = daily.index.to_series().groupby(daily.index.to_period("M")).max()
    int_rows = pd.DataFrame({"date": last_day.values, "type": "INTEREST", "ticker": "", "shares": np.nan,
                             "price": np.nan, "amount": monthly_int["sum"].values, "note": "Interest on cash"})
    int_rows = int_rows[int_rows["amount"] > 0.005]
    led = pd.concat([led, int_rows], ignore_index=True) if len(int_rows) else led
    # Stable sort keeps same-day events in the order they were executed; interest rows go last.
    led = led.sort_values("date", kind="stable").reset_index(drop=True)

    return SimResult(
        daily=daily,
        holdings_value=pd.DataFrame(hv, index=cal, columns=tickers),
        shares=pd.Series(shares, index=tickers),
        ledger=led,
        tickers=tickers,
        applied_plans=applied,
        pending_plans=pending,
        depleted_on=depleted_on,
    )


def _describe(plan: PlanVersion) -> str:
    parts = [f"{t} {w:.0%}" for t, w in sorted(plan.weights.items(), key=lambda kv: -kv[1]) if w > 0 and t != CASH]
    cash = 1 - sum(w for t, w in plan.weights.items() if t != CASH)
    if cash > 0.0005:
        parts.append(f"Cash {cash:.0%}")
    return ", ".join(parts) if parts else "all cash"
