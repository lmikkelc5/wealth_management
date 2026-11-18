"""Fetching market data and security metadata, and loading it for the engine.

Pipeline:  source (Yahoo) --> raw_prices table (as received) --> analytics.cleaning --> engine

The fetchers store data exactly as received. Don't clean or fill anything here:
that's analytics/cleaning.py's job.

Set ADVISOR_DATA_SOURCE=synthetic to use fake, deterministic data (offline demo / tests).
"""
from __future__ import annotations

import os
import time
import zlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core import db
from analytics import cleaning

CASH = "CASH"
RATE_TICKER = "^IRX"          # 13-week T-bill yield (percent). Used for interest on cash.
STALE_AFTER = timedelta(hours=6)
RETRY_AFTER_FAIL = timedelta(minutes=15)

_last_failed_attempt: dict[str, datetime] = {}


class DataError(Exception):
    pass


# ====================================================================== sources

class YahooSource:
    name = "Yahoo Finance (yfinance)"

    def fetch_history(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        t = yf.Ticker(ticker)
        df = t.history(period="max", interval="1d", auto_adjust=False, actions=True)
        if df is None or df.empty:
            raise DataError(f"No price data returned for {ticker} (bad ticker, delisted, or rate-limited).")
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
            "close": df["Close"].astype(float),
            "adj_close": df["Adj Close"].astype(float) if "Adj Close" in df else df["Close"].astype(float),
            "dividend": df["Dividends"].astype(float) if "Dividends" in df else 0.0,
            "capital_gain": df["Capital Gains"].astype(float) if "Capital Gains" in df else 0.0,
            "split": df["Stock Splits"].astype(float) if "Stock Splits" in df else 0.0,
        })
        return _drop_unfinished_today(out)

    def fetch_metadata(self, ticker: str) -> dict:
        import yfinance as yf

        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        quote_type = info.get("quoteType")
        meta = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "quote_type": quote_type,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "category": info.get("category"),
            "lookthrough_sectors": None,
            "lookthrough_assets": None,
            "lookthrough_top": None,
        }
        if quote_type in ("ETF", "MUTUALFUND"):
            try:
                fd = t.funds_data
                sw = {k: float(v) for k, v in (fd.sector_weightings or {}).items() if v is not None}
                ac = {k: float(v) for k, v in (fd.asset_classes or {}).items() if v is not None}
                top = fd.top_holdings
                meta["lookthrough_sectors"] = sw or None
                meta["lookthrough_assets"] = ac or None
                if top is not None and not top.empty:
                    meta["lookthrough_top"] = [
                        [str(sym), str(row.get("Name", sym)), float(row.get("Holding Percent", 0) or 0)]
                        for sym, row in top.iterrows()
                    ]
                if not meta["category"]:
                    meta["category"] = (fd.fund_overview or {}).get("categoryName")
            except Exception:
                pass
        meta["asset_class"] = guess_asset_class(quote_type, meta["category"], meta["lookthrough_assets"])
        return meta


class SyntheticSource:
    """Deterministic fake data so the app and tests run offline. Not real prices."""

    name = "Synthetic demo data (NOT real prices)"
    BONDS = {"AGG", "BND", "TLT", "IEF", "SHY", "LQD", "TIP", "VBMFX", "BNDX", "VGIT", "SCHZ"}
    INCEPTION = {"BND": "2007-04-10", "BNDX": "2013-06-04"}
    SECTORS = ["Technology", "Healthcare", "Financial Services", "Industrials", "Consumer Cyclical",
               "Energy", "Consumer Defensive", "Utilities", "Communication Services"]

    def _end(self) -> pd.Timestamp:
        end = os.environ.get("ADVISOR_SYNTHETIC_END")
        return pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    def fetch_history(self, ticker: str) -> pd.DataFrame:
        if ticker.startswith("ZZ"):
            raise DataError(f"No price data returned for {ticker} (synthetic: unknown ticker).")
        start = pd.Timestamp(self.INCEPTION.get(ticker, "2000-01-03"))
        dates = pd.bdate_range(start, self._end())
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        n = len(dates)
        if ticker == RATE_TICKER:
            rate = np.clip(2.0 + np.cumsum(rng.normal(0, 0.03, n)), 0.0, 6.0)
            return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": rate, "adj_close": rate,
                                 "dividend": 0.0, "capital_gain": 0.0, "split": 0.0})
        is_bond = ticker in self.BONDS
        mu, vol, yld = (0.04, 0.05, 0.035) if is_bond else ((0.06, 0.16, 0.0) if ticker == "GLD" else (0.08, 0.18, 0.018))
        rets = rng.normal(mu / 252 - 0.5 * (vol ** 2) / 252, vol / np.sqrt(252), n)
        close = 50 * np.exp(np.cumsum(rets))
        div = np.zeros(n)
        if yld > 0:
            month = dates.month
            new_month = np.r_[True, month[1:] != month[:-1]]
            pay = new_month & ((month % 3 == 0) | is_bond)
            per_year = 12 if is_bond else 4
            div[pay] = close[pay] * yld / per_year
            div[0] = 0.0
        tr = np.cumprod(np.r_[1.0, (close[1:] + div[1:]) / close[:-1]])
        adj = tr / tr[-1] * close[-1]
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close, "adj_close": adj,
                             "dividend": div, "capital_gain": 0.0, "split": 0.0})

    def fetch_metadata(self, ticker: str) -> dict:
        idx = zlib.crc32(ticker.encode()) % len(self.SECTORS)
        if ticker in self.BONDS:
            return {"ticker": ticker, "name": f"{ticker} Bond Fund (synthetic)", "quote_type": "ETF",
                    "asset_class": "Fixed Income", "sector": None, "industry": None,
                    "category": "Intermediate Core Bond",
                    "lookthrough_sectors": None, "lookthrough_assets": {"bondPosition": 0.98, "cashPosition": 0.02},
                    "lookthrough_top": None}
        if ticker in ("SPY", "VTI", "VOO", "IVV", "EFA", "VXUS", "QQQ"):
            return {"ticker": ticker, "name": f"{ticker} Equity ETF (synthetic)", "quote_type": "ETF",
                    "asset_class": "Equity", "sector": None, "industry": None, "category": "Large Blend",
                    "lookthrough_sectors": {"technology": 0.3, "healthcare": 0.12, "financial_services": 0.13,
                                            "industrials": 0.09, "consumer_cyclical": 0.1, "energy": 0.04,
                                            "consumer_defensive": 0.06, "utilities": 0.03,
                                            "communication_services": 0.09, "realestate": 0.02,
                                            "basic_materials": 0.02},
                    "lookthrough_assets": {"stockPosition": 0.995, "cashPosition": 0.005},
                    "lookthrough_top": [["AAPL", "Apple Inc", 0.07], ["MSFT", "Microsoft Corp", 0.065],
                                        ["NVDA", "NVIDIA Corp", 0.06]]}
        if ticker == "GLD":
            return {"ticker": ticker, "name": "Gold Trust (synthetic)", "quote_type": "ETF",
                    "asset_class": "Commodity", "sector": None, "industry": None,
                    "category": "Commodities Focused", "lookthrough_sectors": None,
                    "lookthrough_assets": {"otherPosition": 1.0}, "lookthrough_top": None}
        return {"ticker": ticker, "name": f"{ticker} Corp (synthetic)", "quote_type": "EQUITY",
                "asset_class": "Equity", "sector": self.SECTORS[idx], "industry": f"{self.SECTORS[idx]} - General",
                "category": None, "lookthrough_sectors": None, "lookthrough_assets": None, "lookthrough_top": None}


def get_source():
    if os.environ.get("ADVISOR_DATA_SOURCE", "").lower() == "synthetic":
        return SyntheticSource()
    return YahooSource()


def _drop_unfinished_today(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo includes today's live price as a row during market hours. Drop it until the close is final."""
    ny = datetime.now(ZoneInfo("America/New_York"))
    if len(df) and df["date"].iloc[-1] == ny.strftime("%Y-%m-%d") and (ny.hour, ny.minute) < (16, 30):
        return df.iloc[:-1]
    return df


FUND_ASSET_KEYS = {"stockPosition": "Equity", "bondPosition": "Fixed Income", "cashPosition": "Cash",
                   "preferredPosition": "Fixed Income", "convertiblePosition": "Fixed Income",
                   "otherPosition": "Other"}


def guess_asset_class(quote_type: str | None, category: str | None, assets: dict | None) -> str:
    cat = (category or "").lower()
    if quote_type == "EQUITY":
        return "Equity"
    if quote_type == "CRYPTOCURRENCY":
        return "Crypto"
    if quote_type == "MONEYMARKET" or "money market" in cat:
        return "Cash"
    if any(w in cat for w in ("commodit", "precious", "gold")):
        return "Commodity"
    if any(w in cat for w in ("bond", "treasury", "government", "muni", "credit", "inflation", "fixed income",
                              "ultrashort", "bank loan")):
        return "Fixed Income"
    if any(w in cat for w in ("allocation", "target-date", "target date", "retirement income")):
        return "Mixed"
    if assets:
        totals: dict[str, float] = {}
        for k, v in assets.items():
            totals[FUND_ASSET_KEYS.get(k, "Other")] = totals.get(FUND_ASSET_KEYS.get(k, "Other"), 0) + v
        best = max(totals, key=totals.get)
        if totals[best] >= 0.7:
            return best
        return "Mixed"
    if quote_type in ("ETF", "MUTUALFUND"):
        return "Equity"
    return "Other"


# ====================================================================== refresh

def _is_stale(log: dict | None) -> bool:
    if not log or log.get("status") != "ok" or not log.get("fetched_at"):
        return True
    return datetime.now() - datetime.fromisoformat(log["fetched_at"]) > STALE_AFTER


def needs_refresh(tickers: list[str]) -> list[str]:
    """Tickers whose prices are stale/missing (or that have no metadata), skipping recent failures."""
    tickers = sorted({t for t in tickers if t and t != CASH} | {RATE_TICKER})
    logs = db.get_fetch_log(tickers)
    known = db.get_securities(tickers)
    now = datetime.now()
    out = []
    for t in tickers:
        failed_at = _last_failed_attempt.get(t)
        if failed_at and now - failed_at < RETRY_AFTER_FAIL:
            continue
        if _is_stale(logs.get(t)) or (t != RATE_TICKER and t not in known):
            out.append(t)
    return out


def refresh(tickers: list[str], force: bool = False) -> dict[str, str]:
    """Fetch any tickers whose cached data is missing or stale. Returns {ticker: status message}."""
    source = get_source()
    tickers = sorted({t for t in tickers if t and t != CASH} | {RATE_TICKER})
    logs = db.get_fetch_log(tickers)
    known = db.get_securities(tickers)
    results = {}
    for t in tickers:
        needs_prices = force or _is_stale(logs.get(t))
        failed_at = _last_failed_attempt.get(t)
        if needs_prices and not force and failed_at and datetime.now() - failed_at < RETRY_AFTER_FAIL:
            results[t] = "skipped (recent failure, will retry later)"
            continue
        if needs_prices:
            try:
                raw = source.fetch_history(t)
                db.replace_raw_prices(t, raw)
                db.log_fetch(t, "ok", first_date=raw["date"].iloc[0], last_date=raw["date"].iloc[-1], rows=len(raw))
                _last_failed_attempt.pop(t, None)
                results[t] = f"updated through {raw['date'].iloc[-1]}"
            except Exception as e:  # network errors, rate limits, bad tickers
                _last_failed_attempt[t] = datetime.now()
                db.log_fetch(t, "error", message=str(e)[:300])
                results[t] = f"error: {str(e)[:200]}"
                continue
            if len(tickers) > 5 and isinstance(source, YahooSource):
                time.sleep(0.3)  # be gentle with Yahoo
        if t != RATE_TICKER and (t not in known or force):
            try:
                db.upsert_security(source.fetch_metadata(t))
            except Exception:
                db.upsert_security({"ticker": t, "name": t, "quote_type": None, "asset_class": "Other",
                                    "sector": None, "industry": None, "category": None,
                                    "lookthrough_sectors": None, "lookthrough_assets": None,
                                    "lookthrough_top": None})
    return results


def data_stamp(tickers: list[str]) -> str:
    """Changes whenever any of these tickers is re-fetched (used as a cache key)."""
    logs = db.get_fetch_log(sorted(set(tickers)))
    return "|".join(f"{t}:{logs.get(t, {}).get('fetched_at')}" for t in sorted(set(tickers)))


# ====================================================================== load for engine

def load_market(tickers: list[str]) -> dict:
    """Clean, aligned data for the engine.

    Returns dict with:
      prices        DataFrame  date x ticker, split-adjusted close (NaN before a ticker's first day)
      distributions DataFrame  date x ticker, cash paid per share on the ex-date (0 elsewhere)
      cash_rate     Series     annual interest rate on cash (decimal) by date
      issues        dict       {ticker: [validation messages]}
      missing       list       tickers with no usable data
    """
    tickers = sorted({t for t in tickers if t and t != CASH})
    raw = db.read_raw_prices(tickers + [RATE_TICKER])
    closes, dists, issues, missing = {}, {}, {}, []
    for t in tickers:
        r = raw[raw["ticker"] == t]
        if r.empty:
            missing.append(t)
            continue
        issues[t] = cleaning.validate_prices(r)
        c = cleaning.clean_prices(r)
        closes[t] = c["close"]
        dists[t] = c["distribution"]
    prices = pd.DataFrame(closes).sort_index()
    distributions = pd.DataFrame(dists).reindex(prices.index).fillna(0.0)
    # Forward-fill gaps after each ticker's first price (holidays / missing rows). ffill never fills
    # before a ticker's first price, so pre-inception dates stay NaN.
    prices = prices.ffill()

    rate_raw = raw[raw["ticker"] == RATE_TICKER]
    if not rate_raw.empty:
        rc = cleaning.clean_prices(rate_raw)["close"] / 100.0
        cash_rate = rc.reindex(prices.index.union(rc.index)).ffill().reindex(prices.index).fillna(0.0)
    else:
        cash_rate = pd.Series(0.0, index=prices.index)
    return {"prices": prices, "distributions": distributions, "cash_rate": cash_rate,
            "issues": issues, "missing": missing}
