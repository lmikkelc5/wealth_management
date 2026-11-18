"""SQLite storage: clients, plan history, security metadata, and raw price data.

Everything lives in one file, data/advisor.db (override with the ADVISOR_DB
environment variable). Delete that file to start over.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT NOT NULL,
    risk_score           INTEGER NOT NULL DEFAULT 50,
    goal                 TEXT,
    horizon_years        INTEGER,
    account_type         TEXT,
    start_date           TEXT NOT NULL,
    initial_capital      REAL NOT NULL,
    monthly_contribution REAL NOT NULL DEFAULT 0,
    contribution_end     TEXT,
    monthly_withdrawal   REAL NOT NULL DEFAULT 0,
    withdrawal_start     TEXT,
    withdrawal_growth_pct REAL NOT NULL DEFAULT 0,
    advisory_fee_pct     REAL NOT NULL DEFAULT 0,
    dividend_policy      TEXT NOT NULL DEFAULT 'reinvest',
    rebalance_rule       TEXT NOT NULL DEFAULT 'annual',
    rebalance_band_pct   REAL NOT NULL DEFAULT 5,
    withdrawal_rule      TEXT NOT NULL DEFAULT 'overweight_first',
    benchmark            TEXT,
    notes                TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_versions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id      INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    effective_date TEXT NOT NULL,
    note           TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_weights (
    plan_version_id INTEGER NOT NULL REFERENCES plan_versions(id) ON DELETE CASCADE,
    ticker          TEXT NOT NULL,
    weight          REAL NOT NULL,          -- percent, e.g. 60 for 60%
    PRIMARY KEY (plan_version_id, ticker)
);

-- Security master. Filled from Yahoo; asset_class / sector / name can be edited in the app.
CREATE TABLE IF NOT EXISTS securities (
    ticker              TEXT PRIMARY KEY,
    name                TEXT,
    quote_type          TEXT,
    asset_class         TEXT,
    sector              TEXT,
    industry            TEXT,
    category            TEXT,
    lookthrough_sectors TEXT,   -- JSON {sector: weight}
    lookthrough_assets  TEXT,   -- JSON {stockPosition: w, bondPosition: w, ...}
    lookthrough_top     TEXT,   -- JSON [[symbol, name, weight], ...]
    meta_fetched_at     TEXT,
    user_edited         INTEGER NOT NULL DEFAULT 0
);

-- Raw daily data exactly as received from the source. Cleaning happens in analytics/cleaning.py.
-- close = split-adjusted close (NOT dividend-adjusted); adj_close = split + dividend adjusted.
CREATE TABLE IF NOT EXISTS raw_prices (
    ticker       TEXT NOT NULL,
    date         TEXT NOT NULL,
    close        REAL,
    adj_close    REAL,
    dividend     REAL,
    capital_gain REAL,
    split        REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    ticker     TEXT PRIMARY KEY,
    fetched_at TEXT,
    first_date TEXT,
    last_date  TEXT,
    rows       INTEGER,
    status     TEXT,
    message    TEXT
);
"""

CLIENT_FIELDS = [
    "name", "risk_score", "goal", "horizon_years", "account_type", "start_date",
    "initial_capital", "monthly_contribution", "contribution_end", "monthly_withdrawal",
    "withdrawal_start", "withdrawal_growth_pct", "advisory_fee_pct", "dividend_policy",
    "rebalance_rule", "rebalance_band_pct", "withdrawal_rule", "benchmark", "notes",
]


def db_path() -> Path:
    return Path(os.environ.get("ADVISOR_DB", PROJECT_ROOT / "data" / "advisor.db"))


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- clients

def list_clients() -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM clients ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def get_client(client_id: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    return dict(row) if row else None


def create_client(data: dict, initial_weights: dict[str, float], plan_note: str = "Initial plan") -> int:
    vals = {k: data.get(k) for k in CLIENT_FIELDS}
    with connect() as con:
        cur = con.execute(
            f"INSERT INTO clients ({', '.join(CLIENT_FIELDS)}, created_at) "
            f"VALUES ({', '.join('?' for _ in CLIENT_FIELDS)}, ?)",
            [vals[k] for k in CLIENT_FIELDS] + [_now()],
        )
        client_id = cur.lastrowid
        _insert_plan(con, client_id, vals["start_date"], initial_weights, plan_note)
    return client_id


def update_client(client_id: int, data: dict) -> None:
    fields = [k for k in CLIENT_FIELDS if k in data]
    if not fields:
        return
    with connect() as con:
        con.execute(
            f"UPDATE clients SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
            [data[k] for k in fields] + [client_id],
        )
        if "start_date" in data:
            # Keep the initial plan pinned to the start date.
            first = con.execute(
                "SELECT id FROM plan_versions WHERE client_id = ? ORDER BY effective_date, id LIMIT 1",
                (client_id,),
            ).fetchone()
            if first:
                con.execute("UPDATE plan_versions SET effective_date = ? WHERE id = ?",
                            (data["start_date"], first["id"]))


def delete_client(client_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM clients WHERE id = ?", (client_id,))


# ---------------------------------------------------------------- plans

def _insert_plan(con, client_id: int, effective_date: str, weights: dict[str, float], note: str) -> int:
    cur = con.execute(
        "INSERT INTO plan_versions (client_id, effective_date, note, created_at) VALUES (?, ?, ?, ?)",
        (client_id, effective_date, note, _now()),
    )
    pv_id = cur.lastrowid
    con.executemany(
        "INSERT INTO plan_weights (plan_version_id, ticker, weight) VALUES (?, ?, ?)",
        [(pv_id, t.upper().strip(), float(w)) for t, w in weights.items() if float(w) != 0],
    )
    return pv_id


def add_plan_version(client_id: int, effective_date: str, weights: dict[str, float], note: str) -> int:
    with connect() as con:
        return _insert_plan(con, client_id, effective_date, weights, note)


def delete_plan_version(plan_version_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM plan_versions WHERE id = ?", (plan_version_id,))


def list_plan_versions(client_id: int) -> list[dict]:
    """Plan versions sorted by effective date, each with a `weights` dict (percent)."""
    with connect() as con:
        versions = [dict(r) for r in con.execute(
            "SELECT * FROM plan_versions WHERE client_id = ? ORDER BY effective_date, id", (client_id,)
        ).fetchall()]
        for v in versions:
            rows = con.execute(
                "SELECT ticker, weight FROM plan_weights WHERE plan_version_id = ? ORDER BY weight DESC",
                (v["id"],),
            ).fetchall()
            v["weights"] = {r["ticker"]: r["weight"] for r in rows}
    return versions


# ---------------------------------------------------------------- securities

def get_securities(tickers: list[str] | None = None) -> dict[str, dict]:
    with connect() as con:
        if tickers is None:
            rows = con.execute("SELECT * FROM securities ORDER BY ticker").fetchall()
        else:
            if not tickers:
                return {}
            rows = con.execute(
                f"SELECT * FROM securities WHERE ticker IN ({', '.join('?' for _ in tickers)})", tickers
            ).fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        for k in ("lookthrough_sectors", "lookthrough_assets", "lookthrough_top"):
            d[k] = json.loads(d[k]) if d[k] else None
        out[d["ticker"]] = d
    return out


def upsert_security(meta: dict, keep_user_edits: bool = True) -> None:
    """Insert/update metadata fetched from a source. User-edited name/asset_class/sector are kept."""
    existing = get_securities([meta["ticker"]]).get(meta["ticker"])
    row = dict(meta)
    for k in ("lookthrough_sectors", "lookthrough_assets", "lookthrough_top"):
        row[k] = json.dumps(row[k]) if row.get(k) is not None else None
    row["meta_fetched_at"] = _now()
    if existing and keep_user_edits and existing.get("user_edited"):
        for k in ("name", "asset_class", "sector", "industry"):
            row[k] = existing[k]
        row["user_edited"] = 1
    else:
        row["user_edited"] = 0
    cols = ["ticker", "name", "quote_type", "asset_class", "sector", "industry", "category",
            "lookthrough_sectors", "lookthrough_assets", "lookthrough_top", "meta_fetched_at", "user_edited"]
    with connect() as con:
        con.execute(
            f"INSERT OR REPLACE INTO securities ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            [row.get(c) for c in cols],
        )


def edit_security(ticker: str, name: str, asset_class: str, sector: str, industry: str | None) -> None:
    with connect() as con:
        con.execute(
            "UPDATE securities SET name = ?, asset_class = ?, sector = ?, industry = ?, user_edited = 1 "
            "WHERE ticker = ?",
            (name, asset_class, sector, industry, ticker),
        )


# ---------------------------------------------------------------- raw prices

def replace_raw_prices(ticker: str, df: pd.DataFrame) -> None:
    """Replace all stored rows for a ticker (full-history refetch keeps split adjustments consistent)."""
    cols = ["date", "close", "adj_close", "dividend", "capital_gain", "split"]
    records = [(ticker, *row) for row in df[cols].itertuples(index=False, name=None)]
    with connect() as con:
        con.execute("DELETE FROM raw_prices WHERE ticker = ?", (ticker,))
        con.executemany(
            "INSERT INTO raw_prices (ticker, date, close, adj_close, dividend, capital_gain, split) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )


def read_raw_prices(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close", "dividend", "capital_gain", "split"])
    with connect() as con:
        return pd.read_sql_query(
            f"SELECT * FROM raw_prices WHERE ticker IN ({', '.join('?' for _ in tickers)}) ORDER BY ticker, date",
            con, params=tickers,
        )


def log_fetch(ticker: str, status: str, message: str = "", first_date=None, last_date=None, rows=0) -> None:
    with connect() as con:
        if status == "ok":
            con.execute(
                "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticker, _now(), first_date, last_date, rows, status, message),
            )
        else:
            # Keep the last good fetch info, just record the error.
            prev = con.execute("SELECT * FROM fetch_log WHERE ticker = ?", (ticker,)).fetchone()
            if prev and prev["status"] == "ok":
                con.execute("UPDATE fetch_log SET message = ? WHERE ticker = ?",
                            (f"Last refresh failed: {message}", ticker))
            else:
                con.execute(
                    "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ticker, _now(), None, None, 0, status, message),
                )


def get_fetch_log(tickers: list[str] | None = None) -> dict[str, dict]:
    with connect() as con:
        if tickers is None:
            rows = con.execute("SELECT * FROM fetch_log").fetchall()
        else:
            if not tickers:
                return {}
            rows = con.execute(
                f"SELECT * FROM fetch_log WHERE ticker IN ({', '.join('?' for _ in tickers)})", tickers
            ).fetchall()
    return {r["ticker"]: dict(r) for r in rows}
