"""Performance and risk statistics.   *** STARTER VERSION. This file is Mikkel's to own and rewrite. ***

Inputs come from the engine's `daily` DataFrame (index = trading dates) with columns:
    value, cash, deposit, withdrawal, withdrawal_requested, fee, dividends, interest, flow
`flow` is the external cash flow on that day (+deposit, -withdrawal; day 0 = initial capital).
Fees are NOT external flows: they're a cost, so returns here are net of fees.

The app calls: daily_returns, growth_index, drawdown_series, yearly_table, summary, hhi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(daily: pd.DataFrame) -> pd.Series:
    """Time-weighted daily returns. Flows happen at the close, so they didn't earn that day's return:
    r_t = (V_t - F_t) / V_{t-1} - 1."""
    v, f = daily["value"], daily["flow"]
    prev = v.shift(1)
    r = (v - f) / prev - 1
    r = r.iloc[1:]
    r = r[prev.iloc[1:] > 0.01]
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def growth_index(r: pd.Series, start_date=None) -> pd.Series:
    """Growth of $1 (TWR). Starts at 1.0 on start_date if given."""
    idx = (1 + r).cumprod()
    if start_date is not None:
        idx = pd.concat([pd.Series([1.0], index=[pd.Timestamp(start_date)]), idx])
    return idx


def total_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1) if len(r) else 0.0


def annualize(total: float, start, end) -> float | None:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    if days < 365:
        return None  # don't annualize less than a year
    return (1 + total) ** (365.25 / days) - 1


def volatility(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 2 else float("nan")


def drawdown_series(index: pd.Series) -> pd.Series:
    return index / index.cummax() - 1


def max_drawdown(index: pd.Series) -> dict:
    dd = drawdown_series(index)
    trough = dd.idxmin()
    peak = index.loc[:trough].idxmax()
    after = index.loc[trough:]
    recovered = after[after >= index.loc[peak]]
    return {"max_drawdown": float(dd.min()), "peak": peak, "trough": trough,
            "recovered": recovered.index[0] if len(recovered) else None}


def _excess(r: pd.Series, rf_annual: pd.Series | None) -> pd.Series:
    if rf_annual is None:
        return r
    return r - rf_annual.reindex(r.index).ffill().fillna(0.0) / TRADING_DAYS


def sharpe(r: pd.Series, rf_annual: pd.Series | None = None) -> float:
    ex = _excess(r, rf_annual)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and len(ex) > 2 else float("nan")


def sortino(r: pd.Series, rf_annual: pd.Series | None = None) -> float:
    ex = _excess(r, rf_annual)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and len(ex) > 2 else float("nan")


def beta(r: pd.Series, bench: pd.Series) -> float:
    both = pd.concat([r, bench], axis=1, join="inner").dropna()
    if len(both) < 20:
        return float("nan")
    var = both.iloc[:, 1].var()
    return float(both.iloc[:, 0].cov(both.iloc[:, 1]) / var) if var else float("nan")


def xirr(dates, amounts) -> float:
    """Annualized money-weighted return. amounts from the investor's view: money in < 0, money out > 0."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    amounts = np.asarray(amounts, dtype=float)
    if not (amounts > 0).any() or not (amounts < 0).any():
        return float("nan")
    t = (dates - dates.iloc[0]).dt.days.to_numpy() / 365.25

    def npv(rate):
        return float(np.sum(amounts / (1 + rate) ** t))

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def money_weighted_return(daily: pd.DataFrame) -> float:
    flows = daily.loc[daily["flow"] != 0, "flow"]
    dates = list(flows.index) + [daily.index[-1]]
    amounts = list(-flows.values) + [daily["value"].iloc[-1]]
    return xirr(dates, amounts)


def yearly_table(daily: pd.DataFrame, r: pd.Series, bench_r: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    years = sorted(set(daily.index.year))
    start = daily.index[0]
    today = pd.Timestamp.today()
    for y in years:
        d = daily[daily.index.year == y]
        if y == years[0]:
            start_val = daily["value"].iloc[0]
            flows = d.iloc[1:]          # day 0's deposit is the starting value, not a contribution
        else:
            start_val = daily.loc[daily.index.year < y, "value"].iloc[-1]
            flows = d
        contrib = flows["deposit"].sum()
        wd = flows["withdrawal"].sum()
        fees = flows["fee"].sum()
        income = flows["dividends"].sum() + flows["interest"].sum()
        end_val = d["value"].iloc[-1]
        price_gain = end_val - start_val - contrib + wd + fees - income
        label = str(y)
        if y == years[0] and (start.month, start.day) > (1, 7):
            label += f" (from {start:%b %d})"
        elif y == years[-1] and y == today.year:
            label += " YTD"
        row = {
            "Year": label,
            "Start value": start_val,
            "Contributions": contrib,
            "Withdrawals": wd,
            "Fees": fees,
            "Dividends + interest": income,
            "Price gain/loss": price_gain,
            "End value": end_val,
            "Return": total_return(r[r.index.year == y]),
        }
        if bench_r is not None:
            row["Benchmark"] = total_return(bench_r[bench_r.index.year == y])
            row["vs. benchmark"] = row["Return"] - row["Benchmark"]
        row["Income / withdrawals"] = income / wd if wd > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary(daily: pd.DataFrame, r: pd.Series, rf_annual: pd.Series | None = None,
            bench_r: pd.Series | None = None) -> dict:
    start, end = daily.index[0], daily.index[-1]
    idx = growth_index(r, start)
    tot = total_return(r)
    yt = yearly_table(daily, r)
    full_years = yt.iloc[1:] if len(yt) > 1 else yt
    out = {
        "start": start, "end": end,
        "start_value": float(daily["value"].iloc[0]),
        "end_value": float(daily["value"].iloc[-1]),
        "total_deposits": float(daily["deposit"].iloc[1:].sum()),
        "total_withdrawals": float(daily["withdrawal"].sum()),
        "total_fees": float(daily["fee"].sum()),
        "total_income": float(daily["dividends"].sum() + daily["interest"].sum()),
        "twr_total": tot,
        "twr_annual": annualize(tot, start, end),
        "mwr_annual": money_weighted_return(daily) if (end - start).days >= 365 else None,
        "volatility": volatility(r),
        "sharpe": sharpe(r, rf_annual),
        "sortino": sortino(r, rf_annual),
        **max_drawdown(idx),
        "worst_year": float(full_years["Return"].min()) if len(full_years) else None,
        "best_year": float(full_years["Return"].max()) if len(full_years) else None,
    }
    if bench_r is not None and len(bench_r):
        bt = total_return(bench_r)
        out.update({"bench_total": bt, "bench_annual": annualize(bt, start, end), "beta": beta(r, bench_r),
                    "bench_volatility": volatility(bench_r),
                    "bench_max_drawdown": float(drawdown_series(growth_index(bench_r, start)).min())})
    return out


def hhi(weights) -> float:
    """Herfindahl concentration: sum of squared weights (1 = everything in one thing)."""
    w = np.asarray(list(weights), dtype=float)
    w = w[w > 0]
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    return float((w ** 2).sum())


def effective_n(weights) -> float:
    h = hhi(weights)
    return 1 / h if h and h == h else float("nan")
