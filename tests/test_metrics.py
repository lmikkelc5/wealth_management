import numpy as np
import pandas as pd
import pytest

from analytics import exposure, metrics
from core.engine import PlanVersion, SimConfig, simulate


def test_xirr_simple():
    r = metrics.xirr(["2020-01-01", "2020-12-31"], [-100, 110])
    assert r == pytest.approx(0.10, abs=0.002)


def test_twr_ignores_flows():
    """With 100% in one asset, TWR must equal the asset's price return no matter the flows."""
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2019-01-02", "2021-06-30")
    px = pd.DataFrame({"A": 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(d))))}, index=d)
    dv = pd.DataFrame(0.0, index=d, columns=["A"])
    c = SimConfig(start_date=d[0], initial_capital=50_000, plans=[PlanVersion(d[0], {"A": 1.0})],
                  monthly_contribution=2_000, monthly_withdrawal=700, withdrawal_rule="pro_rata")
    res = simulate(c, px, dv)
    r = metrics.daily_returns(res.daily)
    assert metrics.total_return(r) == pytest.approx(px["A"].iloc[-1] / px["A"].iloc[0] - 1, rel=1e-9)


def test_yearly_table_reconciles():
    rng = np.random.default_rng(2)
    d = pd.bdate_range("2019-03-01", "2021-12-31")
    px = pd.DataFrame({"A": 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(d))))}, index=d)
    dv = pd.DataFrame(0.0, index=d, columns=["A"])
    dv.loc[d[d.is_month_end], "A"] = 0.05
    c = SimConfig(start_date=d[0], initial_capital=100_000, plans=[PlanVersion(d[0], {"A": 0.9})],
                  monthly_withdrawal=500, annual_fee=0.01, dividend_policy="cash")
    res = simulate(c, px, dv, pd.Series(0.03, index=d))
    r = metrics.daily_returns(res.daily)
    yt = metrics.yearly_table(res.daily, r)
    # Each year's end value = next year's start value; components add up.
    assert yt["End value"].iloc[:-1].tolist() == pytest.approx(yt["Start value"].iloc[1:].tolist())
    recon = (yt["Start value"] + yt["Contributions"] - yt["Withdrawals"] - yt["Fees"]
             + yt["Dividends + interest"] + yt["Price gain/loss"])
    assert recon.tolist() == pytest.approx(yt["End value"].tolist())
    # Yearly returns chain to the total.
    assert np.prod(1 + yt["Return"]) - 1 == pytest.approx(metrics.total_return(r))


def test_max_drawdown():
    idx = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3], index=pd.bdate_range("2020-01-01", periods=5))
    md = metrics.max_drawdown(idx)
    assert md["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1)
    assert md["recovered"] == idx.index[4]


def test_exposure_lookthrough():
    sec = {
        "SPY": {"quote_type": "ETF", "asset_class": "Equity", "user_edited": 0,
                "lookthrough_assets": {"stockPosition": 1.0},
                "lookthrough_sectors": {"technology": 0.5, "healthcare": 0.5},
                "lookthrough_top": [["AAPL", "Apple", 0.1]]},
        "AGG": {"quote_type": "ETF", "asset_class": "Fixed Income", "user_edited": 0,
                "lookthrough_assets": {"bondPosition": 0.9, "cashPosition": 0.1}},
        "AAPL": {"quote_type": "EQUITY", "asset_class": "Equity", "sector": "Technology",
                 "industry": "Consumer Electronics", "name": "Apple"},
    }
    h = {"SPY": 60_000, "AGG": 30_000, "AAPL": 5_000}
    sec_tbl = exposure.by_sector(h, 5_000, sec).set_index("Sector")["Value"]
    assert sec_tbl["Technology"] == pytest.approx(35_000)
    assert sec_tbl["Healthcare"] == pytest.approx(30_000)
    assert sec_tbl["Fixed Income"] == pytest.approx(27_000)
    assert sec_tbl["Cash"] == pytest.approx(8_000)
    under, visible = exposure.underlying_stocks(h, sec)
    assert under.set_index("Symbol").loc["AAPL", "Value"] == pytest.approx(11_000)
