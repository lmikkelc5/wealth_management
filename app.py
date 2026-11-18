"""Advisor Sandbox: practice investment advising on simulated clients.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import cleaning, exposure, metrics
from core import data, db, service
from core.engine import CASH

st.set_page_config(page_title="Advisor Sandbox", page_icon="📈", layout="wide")
db.init_db()

# ----------------------------------------------------------------------------- constants

BLUE = "#2a78d6"            # the client's portfolio
BENCH = "#8a8984"           # benchmark (neutral, recessive)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CASH_COLOR = "#c3c2b7"
OTHER_COLOR = "#8a8984"

PAGES = ["All clients", "New client", "Client", "Securities & data"]
GOALS = ["Retirement income", "Growth", "Balanced growth", "Capital preservation", "Education", "Other"]
ACCOUNT_TYPES = ["Taxable", "Traditional IRA", "Roth IRA", "401(k)", "Trust", "Other"]
REBALANCE = {"annual": "Annually (first trading day of the year)", "quarterly": "Quarterly",
             "threshold": "When any weight drifts past the band", "none": "Never"}
WITHDRAW_RULE = {"overweight_first": "Sell overweight holdings first", "pro_rata": "Sell proportionally"}
DIVIDENDS = {"reinvest": "Reinvest in the same holding", "cash": "Pay to cash (income)"}
ASSET_CLASSES = ["Equity", "Fixed Income", "Cash", "Commodity", "Real Estate", "Mixed", "Crypto", "Other"]
SINGLE_STOCK_LIMIT = 0.20      # flag a single stock above this weight
SECTOR_LIMIT = 0.30            # flag an equity sector above this weight
PRE_2005 = "SPY, IVV, VTI, QQQ, IWM, EFA, EEM, VNQ, AGG, TLT, IEF, SHY, LQD, TIP, GLD (from Nov 2004)"


# ----------------------------------------------------------------------------- helpers

def money(x, dec=0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"-${abs(x):,.{dec}f}" if x < 0 else f"${x:,.{dec}f}"


def pct(x, sign=False, dec=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{dec}%}" if sign else f"{x:.{dec}%}"


def _norm(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def money_short(x) -> str:
    if x is not None and abs(x) >= 1e6:
        return f"{'-' if x < 0 else ''}${abs(x) / 1e6:,.2f}M"
    return money(x)


def styled(df: pd.DataFrame, money_cols=(), pct_cols=(), signed_cols=()):
    fmt = {c: money for c in money_cols if c in df}
    fmt.update({c: pct for c in pct_cols if c in df})
    fmt.update({c: (lambda v: pct(v, sign=True)) for c in signed_cols if c in df})
    return df.style.format(fmt, na_rep="—")


def go_to(page: str, client_id: int | None = None) -> None:
    st.session_state["_goto"] = (page, client_id)


def weights_text(w: dict) -> str:
    parts = [f"{t} {v:g}%" for t, v in sorted(w.items(), key=lambda kv: -kv[1])]
    rest = 100 - sum(v for t, v in w.items())
    if rest > 0.05:
        parts.append(f"(+{rest:g}% cash)")
    return ", ".join(parts)


def parse_weights(df: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    weights, errors = {}, []
    for _, row in df.iterrows():
        t = str(row.get("Ticker") or "").strip().upper()
        w = row.get("Weight %")
        if not t or t == "NONE" or w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        w = float(w)
        if w < 0:
            errors.append(f"{t}: weight can't be negative.")
        if w == 0:
            continue
        weights[t] = weights.get(t, 0) + w
    total = sum(weights.values())
    if not weights:
        errors.append("Add at least one holding.")
    if total > 100.0001:
        errors.append(f"Weights add up to {total:g}%. They can't exceed 100% (anything under 100% is cash).")
    return weights, errors


def weights_editor(key: str, weights: dict | None):
    rows = [{"Ticker": t, "Weight %": float(w)} for t, w in (weights or {}).items()]
    df = pd.DataFrame(rows or [{"Ticker": "", "Weight %": None}], columns=["Ticker", "Weight %"])
    return st.data_editor(
        df, key=key, num_rows="dynamic", hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo format (e.g. BRK-B). Use CASH for cash."),
            "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=0.5),
        },
    )


@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(client_json: str, versions_json: str, stamp: str) -> dict:
    return service.run_client(json.loads(client_json), json.loads(versions_json))


def run(client: dict, versions: list[dict]) -> dict:
    tickers = service.client_tickers(client, versions)
    stamp = data.data_stamp(tickers + [data.RATE_TICKER])
    return cached_run(json.dumps(client, sort_keys=True, default=str),
                      json.dumps(versions, sort_keys=True, default=str), stamp)


def ensure_data(tickers: list[str], force: bool = False) -> None:
    todo = sorted(set(tickers) | {data.RATE_TICKER}) if force else data.needs_refresh(tickers)
    if not todo:
        return
    with st.spinner(f"Fetching market data: {', '.join(t for t in todo)} …"):
        res = data.refresh(todo, force=force)
    for t, msg in res.items():
        if msg.startswith("error") and t != data.RATE_TICKER:
            st.warning(f"Couldn't refresh {t}: {msg[7:]}")


def plotly_layout(fig: go.Figure, height=360, yfmt=None, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if legend else None,
        showlegend=legend, barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def hbar(df: pd.DataFrame, label: str, value: str = "Weight", targets: dict | None = None, height=None):
    df = df.sort_values(value)
    show_text = not targets      # with target markers, the table beside the chart carries the numbers
    fig = go.Figure(go.Bar(
        x=df[value], y=df[label], orientation="h", marker_color=BLUE, name="Current",
        text=[pct(v) for v in df[value]] if show_text else None, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    if targets:
        tx = [targets.get(lbl, 0.0) for lbl in df[label]]
        fig.add_trace(go.Scatter(x=tx, y=df[label], mode="markers", name="Target",
                                 marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#33322f")),
                                 hovertemplate="Target %{y}: %{x:.1%}<extra></extra>"))
    plotly_layout(fig, height=height or 60 + 30 * len(df), yfmt=None, legend=bool(targets))
    fig.update_layout(hovermode="closest", bargap=0.35)
    top = max([float(df[value].max()) if len(df) else 0.0] + list((targets or {}).values()))
    fig.update_xaxes(tickformat=".0%", range=[0, max(top, 0.01) * 1.18], showgrid=True,
                     gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=False)
    return fig


# ----------------------------------------------------------------------------- navigation

if "_goto" in st.session_state:
    _page, _cid = st.session_state.pop("_goto")
    st.session_state["nav"] = _page
    if _cid is not None:
        st.session_state["client_pick"] = _cid
st.session_state.setdefault("nav", "All clients")

clients = db.list_clients()
client_ids = [c["id"] for c in clients]
client_names = {c["id"]: c["name"] for c in clients}
if st.session_state.get("client_pick") not in client_ids:
    st.session_state["client_pick"] = client_ids[0] if client_ids else None

with st.sidebar:
    st.markdown("## 📈 Advisor Sandbox")
    st.radio("Go to", PAGES, key="nav")
    if client_ids:
        st.selectbox("Client", client_ids, key="client_pick", format_func=lambda i: client_names.get(i, i),
                     on_change=go_to, args=("Client",))
    st.divider()
    src = data.get_source()
    st.caption(f"Data: {src.name}. Prices refresh automatically when older than "
               f"{int(data.STALE_AFTER.total_seconds() // 3600)} hours.")
    if st.button("Refresh market data now"):
        all_t = sorted({t for c in clients for t in service.client_tickers(c, db.list_plan_versions(c["id"]))})
        ensure_data(all_t, force=True)
        st.toast("Market data refreshed.")

if isinstance(data.get_source(), data.SyntheticSource):
    st.warning("Demo mode: prices are synthetic, not real. Unset ADVISOR_DATA_SOURCE to use Yahoo data.", icon="🧪")


# ----------------------------------------------------------------------------- client form fields

def client_fields(prefix: str, c: dict) -> dict:
    """Renders the client settings widgets (inside a form) and returns their values."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profile**")
        name = st.text_input("Client name", value=c.get("name", ""), key=f"{prefix}name")
        risk = st.slider("Risk score (0 = very conservative, 100 = very aggressive)", 0, 100,
                         int(c.get("risk_score", 50)), key=f"{prefix}risk")
        goal = st.selectbox("Goal", GOALS, index=GOALS.index(c["goal"]) if c.get("goal") in GOALS else 0,
                            key=f"{prefix}goal")
        horizon = st.number_input("Time horizon (years)", 1, 70, int(c.get("horizon_years") or 20),
                                  key=f"{prefix}horizon")
        acct = st.selectbox("Account type", ACCOUNT_TYPES,
                            index=ACCOUNT_TYPES.index(c["account_type"]) if c.get("account_type") in ACCOUNT_TYPES else 0,
                            key=f"{prefix}acct")
        notes = st.text_area("Notes", value=c.get("notes") or "", key=f"{prefix}notes", height=100)
    with col2:
        st.markdown("**Money in / out**")
        start = st.date_input("Start date (invested at that day's close)",
                              value=pd.Timestamp(c["start_date"]).date() if c.get("start_date") else date.today(),
                              min_value=date(1990, 1, 1), max_value=date.today(), key=f"{prefix}start")
        capital = st.number_input("Starting capital ($)", 1.0, 1e10, float(c.get("initial_capital") or 500_000),
                                  step=10_000.0, format="%.0f", key=f"{prefix}capital")
        contrib = st.number_input("Monthly contribution ($)", 0.0, 1e9, float(c.get("monthly_contribution") or 0),
                                  step=100.0, format="%.0f", key=f"{prefix}contrib")
        c_end = st.date_input("Contributions stop on (optional)",
                              value=pd.Timestamp(c["contribution_end"]).date() if c.get("contribution_end") else None,
                              min_value=date(1990, 1, 1), max_value=date(2100, 1, 1), key=f"{prefix}cend")
        wd = st.number_input("Monthly withdrawal / income need ($)", 0.0, 1e9,
                             float(c.get("monthly_withdrawal") or 0), step=100.0, format="%.0f", key=f"{prefix}wd")
        w_start = st.date_input("Withdrawals start on (optional, default = start date)",
                                value=pd.Timestamp(c["withdrawal_start"]).date() if c.get("withdrawal_start") else None,
                                min_value=date(1990, 1, 1), max_value=date(2100, 1, 1), key=f"{prefix}wstart")
        growth = st.number_input("Raise withdrawal each year by (%)", 0.0, 15.0,
                                 float(c.get("withdrawal_growth_pct") or 0), step=0.5, key=f"{prefix}growth")
    with col3:
        st.markdown("**Management**")
        fee = st.number_input("Advisory fee (% per year, billed quarterly)", 0.0, 5.0,
                              float(c.get("advisory_fee_pct") if c.get("advisory_fee_pct") is not None else 1.0),
                              step=0.05, key=f"{prefix}fee")
        divs = st.selectbox("Dividends", list(DIVIDENDS), format_func=DIVIDENDS.get,
                            index=list(DIVIDENDS).index(c.get("dividend_policy", "reinvest")), key=f"{prefix}divs")
        reb = st.selectbox("Rebalancing", list(REBALANCE), format_func=REBALANCE.get,
                           index=list(REBALANCE).index(c.get("rebalance_rule", "annual")), key=f"{prefix}reb")
        band = st.number_input("Drift band (percentage points, threshold rule only)", 0.5, 50.0,
                               float(c.get("rebalance_band_pct") or 5), step=0.5, key=f"{prefix}band")
        wrule = st.selectbox("To fund withdrawals/fees when cash runs out", list(WITHDRAW_RULE),
                             format_func=WITHDRAW_RULE.get,
                             index=list(WITHDRAW_RULE).index(c.get("withdrawal_rule", "overweight_first")),
                             key=f"{prefix}wrule")
        bench = st.text_input("Benchmark (e.g. SPY:60, AGG:40). Blank = based on risk score",
                              value=c.get("benchmark") or "", key=f"{prefix}bench")
    return {
        "name": name.strip(), "risk_score": int(risk), "goal": goal, "horizon_years": int(horizon),
        "account_type": acct, "start_date": start.isoformat(), "initial_capital": float(capital),
        "monthly_contribution": float(contrib), "contribution_end": c_end.isoformat() if c_end else None,
        "monthly_withdrawal": float(wd), "withdrawal_start": w_start.isoformat() if w_start else None,
        "withdrawal_growth_pct": float(growth), "advisory_fee_pct": float(fee), "dividend_policy": divs,
        "rebalance_rule": reb, "rebalance_band_pct": float(band), "withdrawal_rule": wrule,
        "benchmark": bench.strip() or service.default_benchmark(int(risk), start), "notes": notes.strip(),
    }


def check_client(c: dict) -> list[str]:
    errs = []
    if not c["name"]:
        errs.append("Give the client a name.")
    try:
        service.parse_weights_spec(c["benchmark"])
    except ValueError:
        errs.append("Benchmark should look like 'SPY:60, AGG:40'.")
    return errs


def try_run(client: dict, versions: list[dict]) -> dict:
    ensure_data(service.client_tickers(client, versions))
    return service.run_client(client, versions)


# ----------------------------------------------------------------------------- examples

def load_examples() -> None:
    db.create_client({
        "name": "Example: Retiree from 2005", "risk_score": 40, "goal": "Retirement income", "horizon_years": 25,
        "account_type": "Traditional IRA", "start_date": "2005-01-03", "initial_capital": 1_000_000,
        "monthly_contribution": 0, "monthly_withdrawal": 4_000, "withdrawal_growth_pct": 2.5,
        "advisory_fee_pct": 1.0, "dividend_policy": "cash", "rebalance_rule": "annual", "rebalance_band_pct": 5,
        "withdrawal_rule": "overweight_first", "benchmark": "SPY:40, AGG:60",
        "notes": "Retired at 65 in January 2005. Needs $4,000/month, raised 2.5% a year.",
    }, {"SPY": 35, "EFA": 10, "AGG": 45, "GLD": 5, "CASH": 5}, "Initial plan: income-focused 50/45/5")
    db.create_client({
        "name": "Example: Young saver (starts today)", "risk_score": 80, "goal": "Growth", "horizon_years": 35,
        "account_type": "Roth IRA", "start_date": date.today().isoformat(), "initial_capital": 25_000,
        "monthly_contribution": 500, "monthly_withdrawal": 0, "withdrawal_growth_pct": 0,
        "advisory_fee_pct": 0.5, "dividend_policy": "reinvest", "rebalance_rule": "threshold",
        "rebalance_band_pct": 5, "withdrawal_rule": "pro_rata", "benchmark": "SPY:80, AGG:20",
        "notes": "28, steady income, adds $500/month.",
    }, {"VTI": 60, "VXUS": 25, "BND": 15}, "Initial plan: global 85/15")
    go_to("All clients")


# ----------------------------------------------------------------------------- page: all clients

def page_clients():
    st.title("Clients")
    if not clients:
        st.info("No clients yet. Create one, or load two example clients to see how it works: "
                "a retiree invested in January 2005, and a saver who starts today.")
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("➕ New client", on_click=go_to, args=("New client",), type="primary")
        c2.button("Load example clients", on_click=load_examples)
        return

    versions = {c["id"]: db.list_plan_versions(c["id"]) for c in clients}
    ensure_data(sorted({t for c in clients for t in service.client_tickers(c, versions[c["id"]])}))

    widths = [3, 1.3, 1.5, 1.2, 1.2, 1.2, 1.3, 0.8]
    head = st.columns(widths)
    for col, label in zip(head, ["Client", "Started", "Value", "Annualized", "vs. bench", "Max drawdown",
                                 "Income covers", ""]):
        col.caption(label)
    for c in clients:
        out = run(c, versions[c["id"]])
        cols = st.columns(widths, vertical_alignment="center")
        cols[0].markdown(f"**{c['name']}**  \n<span style='opacity:.65'>{c['goal']} · risk {c['risk_score']}"
                         f"</span>", unsafe_allow_html=True)
        cols[1].write(pd.Timestamp(c["start_date"]).strftime("%b %d, %Y"))
        if out["status"] == "ok":
            s, res = out["summary"], out["result"]
            ann = s["twr_annual"] if s["twr_annual"] is not None else s["twr_total"]
            cols[2].write(money(s["end_value"]))
            cols[3].write(pct(ann) + ("" if s["twr_annual"] is not None else " (total)"))
            if "bench_total" in s:
                b = s["bench_annual"] if s["twr_annual"] is not None else s["bench_total"]
                cols[4].write(pct(ann - b, sign=True))
            else:
                cols[4].write("—")
            cols[5].write(pct(s["max_drawdown"]))
            cover = _trailing_cover(res.daily)
            cols[6].write("—" if cover is None else ("🔴 DEPLETED" if res.depleted_on is not None else pct(cover, dec=0)))
        else:
            cols[2].write("⏳ invests at next close" if out["status"] == "pending" else "⚠️ error")
            for col in cols[3:7]:
                col.write("")
        cols[7].button("Open", key=f"open_{c['id']}", on_click=go_to, args=("Client", c["id"]))
    st.caption("Annualized = time-weighted, net of fees. Income covers = dividends + interest over the last "
               "12 months as a share of withdrawals.")


def _trailing_cover(daily: pd.DataFrame) -> float | None:
    last = daily.index[-1]
    d = daily[daily.index > last - pd.DateOffset(years=1)]
    wd = d["withdrawal_requested"].sum()
    if wd <= 0:
        return None
    return float((d["dividends"].sum() + d["interest"].sum()) / wd)


# ----------------------------------------------------------------------------- page: new client

def page_new_client():
    st.title("New client")
    st.caption(f"Holdings that were trading before 2005: {PRE_2005}. For older start dates, mutual funds "
               "like VFINX and VBMFX go back further.")
    with st.form("new_client"):
        c = client_fields("nc_", {})
        st.markdown("**Initial plan** (anything under 100% is held as cash)")
        left, right = st.columns([1, 1])
        with left:
            edited = weights_editor("nc_weights", {"SPY": 50.0, "AGG": 45.0, "CASH": 5.0})
        with right:
            plan_note = st.text_area("Why this plan? (saved with the plan as your rationale)", key="nc_note",
                                     height=150)
        submitted = st.form_submit_button("Create client and invest", type="primary")
    if not submitted:
        return
    weights, errs = parse_weights(edited)
    errs += check_client(c)
    if errs:
        for e in errs:
            st.error(e)
        return
    versions = [{"effective_date": c["start_date"], "weights": weights, "note": plan_note or "Initial plan"}]
    out = try_run(c, versions)
    if out["status"] == "error":
        st.error(out["message"])
        return
    cid = db.create_client(c, weights, plan_note or "Initial plan")
    go_to("Client", cid)
    st.rerun()


# ----------------------------------------------------------------------------- page: client

def page_client():
    cid = st.session_state.get("client_pick")
    if cid is None:
        st.info("No clients yet.")
        st.button("➕ New client", on_click=go_to, args=("New client",))
        return
    client = db.get_client(cid)
    versions = db.list_plan_versions(cid)
    ensure_data(service.client_tickers(client, versions))
    out = run(client, versions)

    st.title(client["name"])
    st.caption(" · ".join([client["goal"] or "", f"Risk score {client['risk_score']}", client["account_type"] or "",
                           f"{client['horizon_years']}-year horizon",
                           f"Started {pd.Timestamp(client['start_date']):%b %d, %Y} with {money(client['initial_capital'])}",
                           f"Benchmark: {client['benchmark']}"]))
    if out["status"] == "pending":
        st.info(f"⏳ {out['message']}")
    elif out["status"] == "error":
        st.error(f"Can't simulate this client: {out['message']} Fix it in the Plan or Profile tab.")
    for w in out.get("warnings", []):
        st.warning(w)

    ok = out["status"] == "ok"
    if ok:
        kpis(out)
    tabs = st.tabs(["Performance", "Yearly", "Diversification", "Plan", "Activity", "Profile"])
    with tabs[0]:
        if ok:
            performance_tab(out, client)
        else:
            st.caption("Nothing to show yet.")
    with tabs[1]:
        if ok:
            yearly_tab(out)
        else:
            st.caption("Nothing to show yet.")
    with tabs[2]:
        if ok:
            diversification_tab(out, client)
        else:
            st.caption("Nothing to show yet.")
    with tabs[3]:
        plan_tab(out, client, versions)
    with tabs[4]:
        if ok:
            activity_tab(out, client)
        else:
            st.caption("Nothing to show yet.")
    with tabs[5]:
        profile_tab(client, versions)


def kpis(out: dict):
    res, s = out["result"], out["summary"]
    daily = res.daily
    end = daily.index[-1]
    val = daily["value"].iloc[-1]
    cols = st.columns(6)
    if len(daily) > 1:
        chg = val - daily["value"].iloc[-2] - daily["flow"].iloc[-1]
        base = daily["value"].iloc[-2]
        cols[0].metric(f"Value, {end:%b %d} close", money_short(val), help=f"{money(val, 2)}. Change is vs. the "
                       "previous close, excluding deposits/withdrawals.",
                       delta=f"{money(chg)} ({pct(chg / base if base else 0, sign=True, dec=2)})")
    else:
        cols[0].metric(f"Value, {end:%b %d} close", money_short(val), help=money(val, 2))
    bench_delta = f"{pct(s['twr_total'] - s['bench_total'], sign=True)} vs benchmark" if "bench_total" in s else None
    cols[1].metric("Total return", pct(s["twr_total"]), delta=bench_delta,
                   help="Time-weighted (ignores the timing of deposits/withdrawals), net of advisory fees.")
    cols[2].metric("Annualized", pct(s["twr_annual"]) if s["twr_annual"] is not None else "< 1 yr",
                   delta=(f"{pct(s['twr_annual'] - s['bench_annual'], sign=True)} vs benchmark"
                          if s.get("bench_annual") is not None and s["twr_annual"] is not None else None))
    cols[3].metric("Money-weighted (IRR)", pct(s["mwr_annual"]) if s["mwr_annual"] is not None else "< 1 yr",
                   help="The client's actual experience, including the timing of deposits and withdrawals.")
    cols[4].metric("Max drawdown", pct(s["max_drawdown"]),
                   help=f"Peak {s['peak']:%b %d, %Y} → trough {s['trough']:%b %d, %Y}; "
                        + (f"recovered {s['recovered']:%b %d, %Y}." if s["recovered"] is not None else "not yet recovered."))
    cover = _trailing_cover(daily)
    if cover is not None:
        cols[5].metric("Income coverage (12 mo)", pct(cover, dec=0),
                       help="Dividends + interest over the last 12 months ÷ withdrawals requested. "
                            "Under 100% means the rest came from selling holdings.")
    else:
        last12 = daily[daily.index > end - pd.DateOffset(years=1)]
        cols[5].metric("Income (12 mo)", money(last12["dividends"].sum() + last12["interest"].sum()))
    if res.depleted_on is not None:
        st.error(f"🔴 This portfolio ran out of money on {res.depleted_on:%B %d, %Y}.")
    st.caption(f"Since start: deposits {money(s['total_deposits'])} · withdrawals {money(s['total_withdrawals'])} "
               f"· fees {money(s['total_fees'])} · dividends + interest {money(s['total_income'])}")


def performance_tab(out: dict, client: dict):
    res, r = out["result"], out["returns"]
    bench, br = out["bench"], out["bench_returns"]
    start = res.daily.index[0]
    view = st.radio("Show", ["Account value", "Growth of $1 (time-weighted)"], horizontal=True,
                    key=f"perf_view_{client['id']}", label_visibility="collapsed")
    fig = go.Figure()
    if view == "Account value":
        y, yb, fmt = res.daily["value"], (bench.daily["value"] if bench else None), "$,.0f"
    else:
        y, yb, fmt = metrics.growth_index(r, start), (metrics.growth_index(br, start) if br is not None else None), "$.2f"
    if yb is not None:
        fig.add_trace(go.Scatter(x=yb.index, y=yb, name=f"Benchmark ({client['benchmark']})",
                                 line=dict(color=BENCH, width=2, dash="dot"),
                                 hovertemplate="%{y:" + fmt + "}"))
    fig.add_trace(go.Scatter(x=y.index, y=y, name="Portfolio", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:" + fmt + "}"))
    changes = [(d, p) for d, p in res.applied_plans[1:]]
    if changes:
        xs = [d for d, _ in changes]
        fig.add_trace(go.Scatter(x=xs, y=[y.asof(d) for d in xs], mode="markers", name="Plan change",
                                 marker=dict(size=10, color="#33322f", symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 text=[(p.note or "plan change")[:80] for _, p in changes],
                                 hovertemplate="Plan change: %{text}<extra></extra>"))
    st.plotly_chart(plotly_layout(fig, 380, fmt), key=f"perf_{client['id']}")
    if view == "Account value" and (res.daily["deposit"].iloc[1:].sum() > 0 or res.daily["withdrawal"].sum() > 0):
        st.caption("Account value includes deposits and withdrawals, so it isn't a pure return measure. "
                   "Switch to Growth of $1 to compare investment results.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Drawdown** (time-weighted, from previous peak)")
        f2 = go.Figure()
        if br is not None:
            dd_b = metrics.drawdown_series(metrics.growth_index(br, start))
            f2.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name="Benchmark", line=dict(color=BENCH, width=2, dash="dot"),
                                    hovertemplate="%{y:.1%}"))
        dd = metrics.drawdown_series(metrics.growth_index(r, start))
        f2.add_trace(go.Scatter(x=dd.index, y=dd, name="Portfolio", line=dict(color=BLUE, width=2),
                                fill="tozeroy", fillcolor="rgba(42,120,214,0.12)", hovertemplate="%{y:.1%}"))
        st.plotly_chart(plotly_layout(f2, 300, ".0%"), key=f"dd_{client['id']}")
    with c2:
        st.markdown("**Allocation over time** (share of account)")
        hv = res.holdings_value.copy()
        hv[CASH] = res.daily["cash"]
        weekly = hv.resample("W-FRI").last().dropna(how="all")
        tickers = [t for t in res.tickers]
        if len(tickers) > 7:
            keep = list(hv[tickers].mean().sort_values(ascending=False).index[:7])
            weekly["Other"] = weekly[[t for t in tickers if t not in keep]].sum(axis=1)
            tickers = sorted(keep) + ["Other"]
        f3 = go.Figure()
        for i, t in enumerate(tickers + [CASH]):
            if t not in weekly or weekly[t].abs().sum() == 0:
                continue
            color = CASH_COLOR if t == CASH else (OTHER_COLOR if t == "Other" else SERIES[i % len(SERIES)])
            f3.add_trace(go.Scatter(x=weekly.index, y=weekly[t], name=t, stackgroup="one", groupnorm="fraction",
                                    line=dict(width=0.5, color="white"), fillcolor=color,
                                    hovertemplate="%{y:.1%}"))
        st.plotly_chart(plotly_layout(f3, 300, ".0%"), key=f"alloc_{client['id']}")

    st.markdown("**Risk & return**")
    s = out["summary"]
    bs = metrics.summary(bench.daily, br, out["cash_rate"]) if bench is not None and br is not None else None

    def col(x):
        if x is None:
            return {k: "—" for k in range(9)}
        return [pct(x["twr_annual"]) if x["twr_annual"] is not None else pct(x["twr_total"]) + " (total)",
                pct(x["volatility"]), f"{x['sharpe']:.2f}", f"{x['sortino']:.2f}", pct(x["max_drawdown"]),
                f"{x['peak']:%b %Y} → {x['trough']:%b %Y}",
                f"{x['recovered']:%b %Y}" if x["recovered"] is not None else "not yet",
                pct(x["best_year"]) if x["best_year"] is not None else "—",
                pct(x["worst_year"]) if x["worst_year"] is not None else "—"]
    rows = ["Annualized return", "Volatility (annualized)", "Sharpe ratio", "Sortino ratio", "Max drawdown",
            "Drawdown period", "Recovered", "Best full year", "Worst full year"]
    tbl = pd.DataFrame({"": rows, "Portfolio": col(s)})
    if bs is not None:
        tbl["Benchmark"] = col(bs)
    st.dataframe(tbl, hide_index=True)
    if "beta" in s:
        st.caption(f"Beta vs benchmark: {s['beta']:.2f}. Sharpe/Sortino use the 13-week T-bill yield as the "
                   f"risk-free rate.")


def yearly_tab(out: dict):
    yt = out["yearly"].copy()
    has_bench = "Benchmark" in yt
    fig = go.Figure()
    if has_bench:
        fig.add_trace(go.Bar(x=yt["Year"], y=yt["Benchmark"], name="Benchmark", marker_color=BENCH,
                             hovertemplate="%{y:.1%}"))
    fig.add_trace(go.Bar(x=yt["Year"], y=yt["Return"], name="Portfolio", marker_color=BLUE,
                         hovertemplate="%{y:.1%}"))
    plotly_layout(fig, 320, ".0%")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, key="yearly_chart")
    money_cols = ["Start value", "Contributions", "Withdrawals", "Fees", "Dividends + interest",
                  "Price gain/loss", "End value"]
    st.dataframe(styled(yt, money_cols, ["Return", "Benchmark", "Income / withdrawals"], ["vs. benchmark"]),
                 hide_index=True, height=38 + 35 * len(yt))
    st.caption("End value = start + contributions − withdrawals − fees + dividends/interest + price gain/loss. "
               "Return is time-weighted and net of fees; partial years aren't annualized.")


def diversification_tab(out: dict, client: dict):
    res = out["result"]
    idx = res.daily.index
    asof = st.date_input("As of", value=idx[-1].date(), min_value=idx[0].date(), max_value=idx[-1].date(),
                         key=f"asof_{client['id']}")
    d = idx[idx <= pd.Timestamp(asof)][-1]
    holdings = {t: float(v) for t, v in res.holdings_value.loc[d].items() if v > 0.005}
    cash = float(res.daily.loc[d, "cash"])
    sec = db.get_securities(res.tickers)
    plan = [p for when, p in res.applied_plans if when <= d][-1]
    targets = {t: w for t, w in plan.weights.items() if t != CASH and w > 0}
    targets[CASH] = max(0.0, 1 - sum(targets.values()))

    bh = exposure.by_holding(holdings, cash, sec)
    by_sector = exposure.by_sector(holdings, cash, sec)
    stock_w = {t: v for t, v in holdings.items() if not exposure._is_fund(sec.get(t))
               and (sec.get(t) or {}).get("asset_class", "Equity") == "Equity"}
    total = sum(holdings.values()) + cash

    m = st.columns(4)
    m[0].metric("Effective # of holdings", f"{metrics.effective_n(bh['Value']):.1f}",
                help="1 / Σ(weight²). Ten equal holdings = 10; one big position drags it toward 1.")
    top = bh.iloc[0] if len(bh) else None
    m[1].metric(f"Largest position: {top['Ticker']}" if top is not None else "Largest position",
                pct(top["Weight"]) if top is not None else "—")
    eq_sectors = by_sector[~by_sector["Sector"].isin(["Fixed Income", "Cash", "Commodity", "Other", "Crypto",
                                                       "Mixed / unclassified"])]
    m[2].metric(f"Top equity sector: {eq_sectors.iloc[0]['Sector']}" if len(eq_sectors) else "Top equity sector",
                pct(eq_sectors.iloc[0]["Weight"]) if len(eq_sectors) else "—")
    under, visible = exposure.underlying_stocks(holdings, sec)
    m[3].metric("Visible at stock level", pct(visible, dec=0),
                help="Share of invested money we can see stock-by-stock: direct stocks + funds' top-10 holdings.")
    for t, v in stock_w.items():
        if total and v / total > SINGLE_STOCK_LIMIT:
            st.warning(f"⚠️ Concentration: {t} is {pct(v / total)} of the account (limit {pct(SINGLE_STOCK_LIMIT, dec=0)}).")
    for _, row in eq_sectors.iterrows():
        if row["Weight"] > SECTOR_LIMIT:
            st.warning(f"⚠️ Concentration: {row['Sector']} is {pct(row['Weight'])} of the account "
                       f"(limit {pct(SECTOR_LIMIT, dec=0)}).")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"**By holding**, vs the plan in effect ({d:%b %d, %Y})")
        bh["Target"] = bh["Ticker"].map(lambda t: targets.get(t, 0.0))
        bh["Drift"] = bh["Weight"] - bh["Target"]
        for t, w in targets.items():
            if t not in set(bh["Ticker"]) and w > 0:
                bh.loc[len(bh)] = [t, t, "", 0.0, 0.0, w, -w]
        st.dataframe(styled(bh, ["Value"], ["Weight", "Target"], ["Drift"]), hide_index=True,
                     column_config={"Name": st.column_config.TextColumn(width="medium")})
    with c2:
        st.plotly_chart(hbar(bh.rename(columns={"Ticker": "Holding"}), "Holding", targets=targets),
                        key=f"hold_bar_{client['id']}")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**By asset class** (funds looked through)")
        st.plotly_chart(hbar(exposure.by_asset_class(holdings, cash, sec), "Asset class"),
                        key=f"ac_bar_{client['id']}")
    with c4:
        st.markdown("**By sector** (funds looked through)")
        st.plotly_chart(hbar(by_sector, "Sector"), key=f"sec_bar_{client['id']}")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**By industry** (single stocks; funds grouped by category)")
        st.plotly_chart(hbar(exposure.by_industry(holdings, cash, sec), "Industry"), key=f"ind_bar_{client['id']}")
    with c6:
        st.markdown("**Biggest underlying stocks** (direct + inside funds' top 10)")
        if len(under):
            st.dataframe(styled(under, ["Value"]).format({"Weight (of portfolio)": lambda v: pct(v, dec=2)}),
                         hide_index=True)
        else:
            st.caption("No stock-level data (no single stocks, and no fund top-holdings available).")
    st.caption("Fund look-through uses Yahoo's current fund data (sector weights, asset mix, top 10), even for "
               "past dates. Fix wrong classifications on the Securities & data page.")


def plan_tab(out: dict, client: dict, versions: list[dict]):
    res = out.get("result")
    applied_dates = {}
    if res is not None:
        for when, p in res.applied_plans:
            applied_dates[str(pd.Timestamp(p.effective_date).date())] = when
    st.markdown("**Plan history**")
    for i, v in enumerate(versions):
        eff = pd.Timestamp(v["effective_date"])
        with st.container(border=True):
            a, b, c = st.columns([1.2, 4, 0.8], vertical_alignment="center")
            status = "Initial plan" if i == 0 else ("Scheduled" if eff > pd.Timestamp.today() else "Applied")
            when = applied_dates.get(str(eff.date()))
            a.markdown(f"**{eff:%b %d, %Y}**  \n<span style='opacity:.65'>{status}"
                       + (f", traded {when:%b %d}" if when is not None and when.date() != eff.date() else "")
                       + "</span>", unsafe_allow_html=True)
            b.markdown(weights_text(v["weights"]) + (f"  \n_{v['note']}_" if v.get("note") else ""))
            if i > 0:
                c.button("Delete", key=f"del_pv_{v['id']}", on_click=db.delete_plan_version, args=(v["id"],))

    st.markdown("**Change the plan**")
    st.caption("The account is rebalanced to the new weights at the close on the effective date (or the next "
               "trading day). Pick a past date to practice a review at that point in history. An effective "
               "date on or before the start date replaces the initial plan.")
    with st.form(f"plan_form_{client['id']}"):
        a, b = st.columns([1, 1])
        with a:
            eff = st.date_input("Effective date", value=date.today(), min_value=date(1990, 1, 1),
                                max_value=date.today() + timedelta(days=366), key=f"pf_eff_{client['id']}")
            edited = weights_editor(f"pf_w_{client['id']}", versions[-1]["weights"] if versions else None)
        with b:
            note = st.text_area("Rationale (what changed, and why?)", key=f"pf_note_{client['id']}", height=200)
        submitted = st.form_submit_button("Save plan change", type="primary")
    if submitted:
        weights, errs = parse_weights(edited)
        if errs:
            for e in errs:
                st.error(e)
            return
        start = pd.Timestamp(client["start_date"])
        replace_initial = pd.Timestamp(eff) <= start
        new_v = {"effective_date": (start if replace_initial else pd.Timestamp(eff)).date().isoformat(),
                 "weights": weights, "note": note}
        candidate = ([new_v] + versions[1:]) if replace_initial else versions + [new_v]
        candidate = sorted(candidate, key=lambda v: v["effective_date"])
        check = try_run(client, candidate)
        if check["status"] == "error":
            st.error(check["message"])
            return
        if replace_initial:
            db.delete_plan_version(versions[0]["id"])
        db.add_plan_version(client["id"], new_v["effective_date"], weights, note)
        st.rerun()


def activity_tab(out: dict, client: dict):
    led = out["result"].ledger.copy()
    types = sorted(led["type"].unique())
    default = [t for t in types if t not in ("BUY", "SELL", "DIVIDEND", "INTEREST")] or types
    pick = st.multiselect("Show", types, default=default, key=f"led_types_{client['id']}")
    view = led[led["type"].isin(pick)].iloc[::-1]
    st.dataframe(view, hide_index=True, column_config={
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "type": "Type", "ticker": "Ticker",
        "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
        "price": st.column_config.NumberColumn("Price", format="dollar"),
        "amount": st.column_config.NumberColumn("Amount", format="dollar"),
        "note": "Note",
    })
    st.download_button("Download full activity (CSV)", led.to_csv(index=False).encode(),
                       file_name=f"{client['name']}_activity.csv".replace(" ", "_"), mime="text/csv")
    st.caption("Prices are split-adjusted to today's share units, so share counts and prices before a split "
               "won't match old statements. Dollar amounts are unaffected.")


def profile_tab(client: dict, versions: list[dict]):
    with st.form(f"profile_{client['id']}"):
        c = client_fields(f"pf{client['id']}_", client)
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        errs = check_client(c)
        if errs:
            for e in errs:
                st.error(e)
        else:
            check = try_run({**client, **c}, versions)
            if check["status"] == "error":
                st.error(check["message"])
            else:
                db.update_client(client["id"], c)
                st.rerun()
    with st.expander("Delete this client"):
        sure = st.checkbox("Yes, delete this client and its plan history", key=f"sure_{client['id']}")
        if st.button("Delete client", disabled=not sure, key=f"del_{client['id']}"):
            db.delete_client(client["id"])
            go_to("All clients")
            st.rerun()


# ----------------------------------------------------------------------------- page: securities & data

def page_data():
    st.title("Securities & data")
    st.caption(f"Source: {data.get_source().name}. Raw data is stored exactly as received; "
               "analytics/cleaning.py turns it into what the engine uses.")
    sec = db.get_securities()
    log = db.get_fetch_log()
    if not sec:
        st.info("No securities yet. They're added automatically when you create clients.")
        return
    rows = []
    for t, m in sec.items():
        lg = log.get(t, {})
        look = ", ".join(n for n, k in [("sectors", "lookthrough_sectors"), ("asset mix", "lookthrough_assets"),
                                         ("top 10", "lookthrough_top")] if m.get(k))
        rows.append({"Ticker": t, "Name": m.get("name"), "Type": m.get("quote_type"),
                     "Asset class": m.get("asset_class"), "Sector": m.get("sector"), "Industry": m.get("industry"),
                     "Category": m.get("category"), "Fund look-through": look or "—",
                     "Data from": lg.get("first_date"), "Data through": lg.get("last_date"),
                     "Fetched": lg.get("fetched_at"), "Edited": bool(m.get("user_edited")),
                     "Message": lg.get("message") or ""})
    df = pd.DataFrame(rows)
    st.markdown("**Edit classifications** (edits stick even when data is refreshed)")
    edited = st.data_editor(
        df, hide_index=True, key="sec_editor",
        disabled=[c for c in df.columns if c not in ("Name", "Asset class", "Sector", "Industry")],
        column_config={"Asset class": st.column_config.SelectboxColumn(options=ASSET_CLASSES)},
    )
    if st.button("Save classification edits", type="primary"):
        n = 0
        for (_, old), (_, new) in zip(df.iterrows(), edited.iterrows()):
            if any(_norm(old[c]) != _norm(new[c]) for c in ("Name", "Asset class", "Sector", "Industry")):
                db.edit_security(new["Ticker"], _norm(new["Name"]) or new["Ticker"], _norm(new["Asset class"]) or "Other",
                                 _norm(new["Sector"]) or None, _norm(new["Industry"]) or None)
                n += 1
        if n:
            st.success(f"Saved {n} change(s).")
        else:
            st.info("No changes.")

    st.markdown("**Data checks** (from analytics/cleaning.py)")
    raw = db.read_raw_prices(sorted(sec))
    any_issue = False
    for t in sorted(sec):
        issues = cleaning.validate_prices(raw[raw["ticker"] == t])
        if issues:
            any_issue = True
            with st.expander(f"{t}: {len(issues)} issue(s)"):
                for i in issues:
                    st.write("• " + i)
    if not any_issue:
        st.caption("No issues found.")


# ----------------------------------------------------------------------------- router

page = st.session_state["nav"]
if page == "All clients":
    page_clients()
elif page == "New client":
    page_new_client()
elif page == "Client":
    page_client()
else:
    page_data()
