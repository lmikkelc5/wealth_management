# Advisor Sandbox: notes for Claude Code

Mikkel's practice tool for investment advising (simulated clients, plans, backtests and daily
forward updates, retirement withdrawals, diversification dashboard). Python + Streamlit + SQLite.
See README.md for how it works and the known simplifications.

## Division of labor

Mikkel owns data cleaning and the stats. Claude Code owns everything else: fetching, storage,
the engine's mechanics, the UI, and deployment.

- Never modify anything in `analytics/`. That's Mikkel's code. Call its functions. If a needed function
  doesn't exist yet, add a stub there that raises NotImplementedError and tell him.
- Store fetched data exactly as received (`raw_prices`). No forward-filling, dropping outliers or adjusting
  in `core/data.py`. Cleaning happens in `analytics/cleaning.py`.
- The engine reads only cleaned data (via `data.load_market`, which calls `analytics.cleaning`), never raw rows.
- Charts and tables display numbers from `analytics/`. Don't recompute metrics in `app.py`.
- Engine changes must keep `tests/test_engine.py` passing. Mikkel adds Excel-verified cases there. If one
  fails, the engine is wrong until proven otherwise.

## Commands

- Run: `streamlit run app.py` (or `run.bat` / `run.sh`)
- Tests: `python -m pytest` (includes headless app smoke tests on synthetic data)
- Offline demo: set `ADVISOR_DATA_SOURCE=synthetic`; separate DB: set `ADVISOR_DB=path/to.db`

## Layout

- `core/db.py`: SQLite schema + CRUD (clients, plan_versions, plan_weights, securities, raw_prices, fetch_log)
- `core/data.py`: Yahoo/synthetic sources, refresh (full-history refetch when >6h old), `load_market`
- `core/engine.py`: daily simulation; conventions are documented at the top of the file
- `core/service.py`: client → SimConfig, benchmark, run_client
- `analytics/`: cleaning, metrics, exposure (Mikkel's)
- `app.py`: Streamlit UI (pages: All clients, New client, Client, Securities & data)

## Ideas queued (not built yet)

Step-through "time machine" mode (hide data after the sim date), blind mode (random start, generic
labels), Monte Carlo for retirees, tax layer (account types, lots, RMDs, harvesting), Tiingo as a
second source, dated fund holdings from SEC N-PORT, client PDF reports, user accounts.
