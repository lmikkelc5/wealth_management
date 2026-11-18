"""Diversification breakdowns.   *** STARTER VERSION. This file is Mikkel's to own and rewrite. ***

All functions take:
    holdings   {ticker: market value}  (positions only)
    cash       float
    securities {ticker: metadata dict from the securities table}

Funds are looked through using Yahoo's *current* fund data (sector weights, asset mix,
top-10 holdings), even when viewing a past date. Good enough for practice; a real version
would use dated holdings (e.g. SEC N-PORT filings).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

SECTOR_LABELS = {
    "realestate": "Real Estate", "consumer_cyclical": "Consumer Cyclical", "basic_materials": "Basic Materials",
    "consumer_defensive": "Consumer Defensive", "technology": "Technology",
    "communication_services": "Communication Services", "financial_services": "Financial Services",
    "utilities": "Utilities", "industrials": "Industrials", "energy": "Energy", "healthcare": "Healthcare",
}
FUND_ASSET_KEYS = {"stockPosition": "Equity", "bondPosition": "Fixed Income", "cashPosition": "Cash",
                   "preferredPosition": "Fixed Income", "convertiblePosition": "Fixed Income",
                   "otherPosition": "Other"}
NO_LOOKTHROUGH = {"Commodity", "Crypto", "Cash"}


def _is_fund(meta: dict | None) -> bool:
    return bool(meta) and meta.get("quote_type") in ("ETF", "MUTUALFUND")


def asset_split(meta: dict | None) -> dict[str, float]:
    if not meta:
        return {"Other": 1.0}
    ac = meta.get("asset_class") or "Other"
    la = meta.get("lookthrough_assets")
    if meta.get("user_edited") or ac in NO_LOOKTHROUGH or not la:
        return {ac: 1.0}
    totals: dict[str, float] = defaultdict(float)
    for k, v in la.items():
        totals[FUND_ASSET_KEYS.get(k, "Other")] += max(v or 0.0, 0.0)
    s = sum(totals.values())
    return {k: v / s for k, v in totals.items() if v > 0} if s > 0 else {ac: 1.0}


def sector_split(meta: dict | None) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for ac, w in asset_split(meta).items():
        if ac != "Equity":
            out["Mixed / unclassified" if ac == "Mixed" else ac] += w
            continue
        ls = None if (meta or {}).get("user_edited") else (meta or {}).get("lookthrough_sectors")
        if ls and sum(ls.values()) > 0:
            s = sum(ls.values())
            for k, v in ls.items():
                out[SECTOR_LABELS.get(k, k.replace("_", " ").title())] += w * v / s
        elif (meta or {}).get("sector"):
            out[meta["sector"]] += w
        else:
            out["Equity (unclassified)"] += w
    return dict(out)


def _table(values: dict[str, float], label: str) -> pd.DataFrame:
    df = pd.DataFrame({label: list(values.keys()), "Value": list(values.values())})
    df = df[df["Value"] > 0.005]
    total = df["Value"].sum()
    df["Weight"] = df["Value"] / total if total > 0 else 0.0
    return df.sort_values("Value", ascending=False).reset_index(drop=True)


def by_holding(holdings: dict, cash: float, securities: dict) -> pd.DataFrame:
    rows = [{"Ticker": t, "Name": (securities.get(t) or {}).get("name") or t,
             "Asset class": (securities.get(t) or {}).get("asset_class") or "Other", "Value": v}
            for t, v in holdings.items() if v > 0.005]
    if cash > 0.005:
        rows.append({"Ticker": "CASH", "Name": "Cash", "Asset class": "Cash", "Value": cash})
    df = pd.DataFrame(rows, columns=["Ticker", "Name", "Asset class", "Value"])
    total = df["Value"].sum()
    df["Weight"] = df["Value"] / total if total > 0 else 0.0
    return df.sort_values("Value", ascending=False).reset_index(drop=True)


def by_asset_class(holdings: dict, cash: float, securities: dict) -> pd.DataFrame:
    acc: dict[str, float] = defaultdict(float)
    for t, v in holdings.items():
        for ac, w in asset_split(securities.get(t)).items():
            acc[ac] += v * w
    acc["Cash"] += cash
    return _table(acc, "Asset class")


def by_sector(holdings: dict, cash: float, securities: dict) -> pd.DataFrame:
    acc: dict[str, float] = defaultdict(float)
    for t, v in holdings.items():
        for s, w in sector_split(securities.get(t)).items():
            acc[s] += v * w
    acc["Cash"] += cash
    return _table(acc, "Sector")


def by_industry(holdings: dict, cash: float, securities: dict) -> pd.DataFrame:
    """Single stocks by industry. Funds don't report industries, so they're grouped by fund category."""
    acc: dict[str, float] = defaultdict(float)
    for t, v in holdings.items():
        meta = securities.get(t) or {}
        if _is_fund(meta):
            acc[f"Fund: {meta.get('category') or meta.get('asset_class') or 'other'}"] += v
        else:
            acc[meta.get("industry") or meta.get("sector") or "Unclassified"] += v
    acc["Cash"] += cash
    return _table(acc, "Industry")


def underlying_stocks(holdings: dict, securities: dict, top_n: int = 15) -> tuple[pd.DataFrame, float]:
    """Combine direct stock positions with funds' top-10 holdings.
    Returns (table, share of total positions value that's visible at the single-stock level)."""
    acc: dict[str, dict] = {}
    total = sum(v for v in holdings.values() if v > 0)
    visible = 0.0
    for t, v in holdings.items():
        meta = securities.get(t) or {}
        if _is_fund(meta):
            for sym, name, w in meta.get("lookthrough_top") or []:
                e = acc.setdefault(sym, {"Name": name, "Value": 0.0, "Via": set()})
                e["Value"] += v * w
                e["Via"].add(t)
                visible += v * w
        elif meta.get("asset_class", "Equity") == "Equity":
            e = acc.setdefault(t, {"Name": meta.get("name") or t, "Value": 0.0, "Via": set()})
            e["Value"] += v
            e["Via"].add("direct")
            visible += v
    rows = [{"Symbol": s, "Name": e["Name"], "Value": e["Value"],
             "Weight (of portfolio)": e["Value"] / total if total else 0.0,
             "Held via": ", ".join(sorted(e["Via"]))} for s, e in acc.items()]
    df = pd.DataFrame(rows, columns=["Symbol", "Name", "Value", "Weight (of portfolio)", "Held via"])
    df = df.sort_values("Value", ascending=False).head(top_n).reset_index(drop=True)
    return df, (visible / total if total else 0.0)
