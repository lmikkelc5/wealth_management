# Advisor Sandbox

Practice investment advising on simulated clients. Create clients with different risk scores,
goals, capital and start dates (2005, today, anything since 1990), invest them in a plan, then
watch them year by year or day by day. Retirees can draw monthly income, which is paid from
dividends and interest first, then by selling holdings.

Runs on your own computer. Uses free Yahoo Finance data through `yfinance`. No API keys needed.

## Start it

**Windows:** install Python 3.11+ from python.org (tick "Add python.exe to PATH"), then
double-click `run.bat`. The first run installs packages (1–2 minutes). The app then opens at
http://localhost:8501. Keep the black window open while you use it.

**Mac / Linux / Raspberry Pi:** `./run.sh`

On first launch, click **Load example clients** to see a 2005 retiree and a saver who starts today.

## How it works

- **Daily updates need no background job.** Each client is re-simulated from its start date
  whenever you open it. Prices refresh from Yahoo when they're more than 6 hours old, so a client
  started "today" gains a day of history after every market close. Leave it alone for a month and it
  catches up when you next open it.
- **Plan changes have effective dates.** On the Plan tab, change the allocation from any date. Pick a
  past date (say 2009-03-09) to practice a client review at that point in history. Every change keeps
  your written rationale, and changes show as markers on the performance chart.
- **Benchmark:** the same deposits and withdrawals, no advisory fee, rebalanced once a year. If you
  leave it blank, the benchmark is set from the risk score (risk 60 → SPY 60 / AGG 40).

## Simplifications (know these before trusting a number)

- All trades happen at the closing price. No commissions, spreads or taxes; fractional shares allowed.
- Dividends are credited on the ex-date, not the pay date.
- Yahoo's `Close` is already adjusted for splits. Share counts and old prices are therefore in today's
  share units; dollar amounts are unaffected.
- Fund capital-gain distributions are ignored for now. Yahoo sometimes includes them in dividends and
  sometimes doesn't. The Data page flags tickers where this matters.
- Survivorship bias: Yahoo mostly has tickers that still exist today. For 2005 starts, stick to
  ETFs and funds that were trading then: SPY, IVV, VTI, QQQ, IWM, EFA, EEM, VNQ, AGG, TLT, IEF,
  SHY, LQD, TIP, GLD.
- The fund breakdowns (sectors, asset mix, top 10 holdings) use Yahoo's *current* fund data, even when
  you look at a past date.
- Cash earns the 13-week T-bill yield (`^IRX`). The advisory fee is charged each quarter on the prior
  close. Withdrawals happen on the first trading day of each month and step up on each anniversary.

## Who owns what

| Folder | Owner | What's in it |
|---|---|---|
| `analytics/` | **Mikkel** | `cleaning.py` (raw → clean data, checks), `metrics.py` (returns, risk), `exposure.py` (diversification). These are working **starter versions** to replace with your own. |
| `core/` | Claude Code | `data.py` (fetch + store raw), `engine.py` (portfolio simulation), `service.py` (glue), `db.py` (SQLite) |
| `app.py` | Claude Code | The Streamlit interface |
| `tests/` | both | Engine tests are small cases checked by hand. Add your Excel-verified cases to `tests/test_engine.py`. |

Data flow: Yahoo → `raw_prices` table (exactly as received) → `analytics/cleaning.py` → engine → `analytics/metrics.py` → app.

## Useful commands

```bash
python -m pytest                     # run all tests (engine, metrics, app smoke tests)
ADVISOR_DATA_SOURCE=synthetic ./run.sh   # demo mode with fake prices (offline)
```

Everything is stored in `data/advisor.db` (SQLite). Delete it to start over. You can also open it
in DB Browser for SQLite or pandas (`pd.read_sql`) to work on the raw data directly.
