"""Glue between stored clients, market data, the engine, and analytics."""
from __future__ import annotations

import pandas as pd

from analytics import metrics
from core import data, db
from core.engine import CASH, NotStartedYet, PlanVersion, SimConfig, SimError, simulate


def parse_weights_spec(spec: str) -> dict[str, float]:
    """'SPY:60, AGG:40' -> {'SPY': 0.6, 'AGG': 0.4}. A bare 'SPY' means 100%."""
    out: dict[str, float] = {}
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            t, w = part.split(":", 1)
            out[t.strip().upper()] = float(w) / 100.0
        else:
            out[part.upper()] = 1.0
    return out


def default_benchmark(risk_score: int, start_date=None) -> str:
    equity = int(round(max(0, min(100, risk_score)) / 10.0) * 10)
    bond = "AGG" if start_date is None or pd.Timestamp(start_date) >= pd.Timestamp("2003-10-01") else "VBMFX"
    if equity >= 100:
        return "SPY:100"
    if equity <= 0:
        return f"{bond}:100"
    return f"SPY:{equity}, {bond}:{100 - equity}"


def _ts(x):
    return pd.Timestamp(x) if x else None


def client_config(client: dict, versions: list[dict]) -> SimConfig:
    plans = [PlanVersion(pd.Timestamp(v["effective_date"]), {t: w / 100.0 for t, w in v["weights"].items()},
                         v.get("note") or "") for v in versions]
    return SimConfig(
        start_date=pd.Timestamp(client["start_date"]),
        initial_capital=float(client["initial_capital"]),
        plans=plans,
        monthly_contribution=float(client["monthly_contribution"] or 0),
        contribution_end=_ts(client.get("contribution_end")),
        monthly_withdrawal=float(client["monthly_withdrawal"] or 0),
        withdrawal_start=_ts(client.get("withdrawal_start")),
        withdrawal_growth=float(client["withdrawal_growth_pct"] or 0) / 100.0,
        annual_fee=float(client["advisory_fee_pct"] or 0) / 100.0,
        dividend_policy=client["dividend_policy"],
        rebalance_rule=client["rebalance_rule"],
        rebalance_band=float(client["rebalance_band_pct"] or 5) / 100.0,
        withdrawal_rule=client["withdrawal_rule"],
    )


def benchmark_config(cfg: SimConfig, weights: dict[str, float]) -> SimConfig:
    """Same deposits/withdrawals as the client, no advisory fee, rebalanced annually."""
    return SimConfig(
        start_date=cfg.start_date, initial_capital=cfg.initial_capital,
        plans=[PlanVersion(cfg.start_date, weights, "benchmark")],
        monthly_contribution=cfg.monthly_contribution, contribution_end=cfg.contribution_end,
        monthly_withdrawal=cfg.monthly_withdrawal, withdrawal_start=cfg.withdrawal_start,
        withdrawal_growth=cfg.withdrawal_growth, annual_fee=0.0, dividend_policy=cfg.dividend_policy,
        rebalance_rule="annual", withdrawal_rule="pro_rata",
    )


def client_tickers(client: dict, versions: list[dict]) -> list[str]:
    ts = {t for v in versions for t in v["weights"] if t != CASH}
    ts |= {t for t in parse_weights_spec(client.get("benchmark") or "") if t != CASH}
    return sorted(ts)


def run_client(client: dict, versions: list[dict]) -> dict:
    """Simulate a client and its benchmark. Assumes data.refresh() already ran for its tickers."""
    tickers = client_tickers(client, versions)
    mkt = data.load_market(tickers)
    out: dict = {"status": "ok", "warnings": [], "issues": mkt["issues"], "missing": mkt["missing"]}
    cfg = client_config(client, versions)
    try:
        res = simulate(cfg, mkt["prices"], mkt["distributions"], mkt["cash_rate"])
    except NotStartedYet as e:
        return {**out, "status": "pending", "message": str(e)}
    except SimError as e:
        return {**out, "status": "error", "message": str(e)}

    r = metrics.daily_returns(res.daily)
    bench_r, bench_res = None, None
    bw = parse_weights_spec(client.get("benchmark") or "")
    if bw:
        try:
            bench_res = simulate(benchmark_config(cfg, bw), mkt["prices"], mkt["distributions"], mkt["cash_rate"])
            bench_r = metrics.daily_returns(bench_res.daily)
        except SimError as e:
            out["warnings"].append(f"Benchmark skipped: {e}")

    for t in res.tickers:
        last = mkt["prices"][t].last_valid_index()
        raw_last = db.get_fetch_log([t]).get(t, {}).get("last_date")
        if raw_last and (res.daily.index[-1] - pd.Timestamp(raw_last)).days > 10:
            out["warnings"].append(f"{t}: data ends {raw_last}; its price is held flat after that.")
        elif last is None:
            out["warnings"].append(f"{t}: no usable prices.")

    out.update({
        "result": res,
        "returns": r,
        "bench": bench_res,
        "bench_returns": bench_r,
        "cash_rate": mkt["cash_rate"],
        "summary": metrics.summary(res.daily, r, mkt["cash_rate"], bench_r),
        "yearly": metrics.yearly_table(res.daily, r, bench_r),
    })
    return out
