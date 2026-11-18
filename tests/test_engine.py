"""Engine tests on tiny, hand-checkable scenarios.

Add your own Excel-checked cases here: build prices/dividends by hand, compute the expected
ending values in Excel, and assert the engine matches.
"""
import numpy as np
import pandas as pd
import pytest

from core.engine import PlanVersion, SimConfig, SimError, NotStartedYet, simulate


def frame(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def zeros_like(prices):
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


def cfg(start, weights, **kw):
    return SimConfig(start_date=pd.Timestamp(start), initial_capital=kw.pop("capital", 100_000.0),
                     plans=[PlanVersion(pd.Timestamp(start), weights)], **kw)


def test_buy_and_hold_tracks_price():
    d = pd.bdate_range("2020-01-02", periods=5)
    px = frame(d, A=[10, 11, 12, 11, 12.0])
    res = simulate(cfg(d[0], {"A": 1.0}, rebalance_rule="none"), px, zeros_like(px))
    assert res.shares["A"] == pytest.approx(10_000)
    assert res.daily["value"].iloc[-1] == pytest.approx(120_000)
    assert res.daily["cash"].iloc[-1] == pytest.approx(0)


def test_dividend_reinvested_buys_shares():
    d = pd.bdate_range("2020-01-02", periods=5)
    px = frame(d, A=[10.0] * 5)
    dv = zeros_like(px)
    dv.iloc[2, 0] = 0.5                       # $0.50/share on day 3
    res = simulate(cfg(d[0], {"A": 1.0}, rebalance_rule="none"), px, dv)
    # 10,000 shares * 0.50 = $5,000 -> 500 more shares at $10
    assert res.shares["A"] == pytest.approx(10_500)
    assert res.daily["value"].iloc[-1] == pytest.approx(105_000)
    assert res.daily["dividends"].sum() == pytest.approx(5_000)


def test_dividend_to_cash():
    d = pd.bdate_range("2020-01-02", periods=5)
    px = frame(d, A=[10.0] * 5)
    dv = zeros_like(px)
    dv.iloc[2, 0] = 0.5
    res = simulate(cfg(d[0], {"A": 1.0}, rebalance_rule="none", dividend_policy="cash"), px, dv)
    assert res.shares["A"] == pytest.approx(10_000)
    assert res.daily["cash"].iloc[-1] == pytest.approx(5_000)


def test_annual_rebalance_restores_weights():
    d = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    px = frame(d, A=[10, 20, 20, 20.0], B=[10, 10, 10, 10.0])
    res = simulate(cfg(d[0], {"A": 0.6, "B": 0.4}, rebalance_rule="annual"), px, zeros_like(px))
    # Day 2: A doubled -> A=120k, B=40k, total 160k. Jan 4: rebalance to 96k / 64k.
    hv = res.holdings_value
    assert hv.loc["2020-12-31", "A"] == pytest.approx(120_000)
    assert hv.loc["2021-01-04", "A"] == pytest.approx(96_000)
    assert hv.loc["2021-01-04", "B"] == pytest.approx(64_000)


def test_no_rebalance_lets_weights_drift():
    d = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04"])
    px = frame(d, A=[10, 20, 20.0], B=[10, 10, 10.0])
    res = simulate(cfg(d[0], {"A": 0.6, "B": 0.4}, rebalance_rule="none"), px, zeros_like(px))
    assert res.holdings_value.loc["2021-01-04", "A"] == pytest.approx(120_000)


def test_withdrawal_from_cash_first():
    d = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    px = frame(d, A=[10.0] * 4)
    res = simulate(cfg(d[0], {"A": 0.9}, monthly_withdrawal=1_000, rebalance_rule="none"),
                   px, zeros_like(px))
    # 90k in A, 10k cash. Feb 3: $1,000 comes out of cash; no shares sold.
    assert res.daily.loc["2020-02-03", "withdrawal"] == pytest.approx(1_000)
    assert res.daily.loc["2020-02-03", "cash"] == pytest.approx(9_000)
    assert res.shares["A"] == pytest.approx(9_000)


def test_withdrawal_pro_rata_sells():
    d = pd.DatetimeIndex(["2020-01-31", "2020-02-03"])
    px = frame(d, A=[10.0, 10.0], B=[20.0, 20.0])
    res = simulate(cfg(d[0], {"A": 0.5, "B": 0.5}, monthly_withdrawal=10_000, withdrawal_rule="pro_rata",
                       rebalance_rule="none"), px, zeros_like(px))
    # No cash: sell $5k of each.
    assert res.holdings_value.iloc[-1]["A"] == pytest.approx(45_000)
    assert res.holdings_value.iloc[-1]["B"] == pytest.approx(45_000)
    assert res.daily["value"].iloc[-1] == pytest.approx(90_000)


def test_withdrawal_overweight_first():
    d = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03"])
    px = frame(d, A=[10.0, 15.0, 15.0], B=[10.0, 10.0, 10.0])
    res = simulate(cfg(d[0], {"A": 0.5, "B": 0.5}, monthly_withdrawal=10_000,
                       withdrawal_rule="overweight_first", rebalance_rule="none"), px, zeros_like(px))
    # Before: A 75k, B 50k (125k). After withdrawal 115k -> targets 57.5k each.
    # A is 17.5k over target, enough to cover 10k, so only A is sold.
    hv = res.holdings_value.iloc[-1]
    assert hv["A"] == pytest.approx(65_000)
    assert hv["B"] == pytest.approx(50_000)


def test_quarterly_fee():
    d = pd.DatetimeIndex(["2020-03-30", "2020-03-31", "2020-04-01"])
    px = frame(d, A=[10.0] * 3)
    res = simulate(cfg(d[0], {"A": 1.0}, annual_fee=0.01, rebalance_rule="none"), px, zeros_like(px))
    # Apr 1: 1%/4 of 100k = $250, raised by selling A.
    assert res.daily.loc["2020-04-01", "fee"] == pytest.approx(250)
    assert res.daily["value"].iloc[-1] == pytest.approx(99_750)


def test_monthly_contribution_invested_at_targets():
    d = pd.DatetimeIndex(["2020-01-31", "2020-02-03"])
    px = frame(d, A=[10.0, 10.0], B=[20.0, 20.0])
    res = simulate(cfg(d[0], {"A": 0.5, "B": 0.4}, monthly_contribution=1_000, rebalance_rule="none"),
                   px, zeros_like(px))
    hv = res.holdings_value.iloc[-1]
    assert hv["A"] == pytest.approx(50_500)
    assert hv["B"] == pytest.approx(40_400)
    assert res.daily["cash"].iloc[-1] == pytest.approx(10_100)
    assert res.daily["flow"].iloc[-1] == pytest.approx(1_000)


def test_contribution_end_date():
    d = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-03-02"])
    px = frame(d, A=[10.0] * 3)
    res = simulate(cfg(d[0], {"A": 1.0}, monthly_contribution=1_000,
                       contribution_end=pd.Timestamp("2020-02-15"), rebalance_rule="none"), px, zeros_like(px))
    assert res.daily["deposit"].iloc[1:].sum() == pytest.approx(1_000)


def test_plan_change_rebalances_on_effective_date():
    d = pd.bdate_range("2020-01-02", periods=6)
    px = frame(d, A=[10.0] * 6, B=[10.0] * 6)
    c = cfg(d[0], {"A": 1.0}, rebalance_rule="none")
    c.plans.append(PlanVersion(pd.Timestamp("2020-01-07"), {"B": 1.0}, "switch"))
    res = simulate(c, px, zeros_like(px))
    hv = res.holdings_value
    assert hv.loc["2020-01-06", "A"] == pytest.approx(100_000)
    assert hv.loc["2020-01-07", "A"] == pytest.approx(0)
    assert hv.loc["2020-01-07", "B"] == pytest.approx(100_000)


def test_future_plan_is_pending():
    d = pd.bdate_range("2020-01-02", periods=3)
    px = frame(d, A=[10.0] * 3)
    c = cfg(d[0], {"A": 1.0})
    c.plans.append(PlanVersion(pd.Timestamp("2030-01-01"), {"A": 0.5}))
    res = simulate(c, px, zeros_like(px))
    assert len(res.pending_plans) == 1


def test_missing_price_on_start_raises():
    d = pd.bdate_range("2020-01-02", periods=3)
    px = frame(d, A=[10.0] * 3, B=[np.nan, 10.0, 10.0])
    with pytest.raises(SimError, match="B has no price"):
        simulate(cfg(d[0], {"A": 0.5, "B": 0.5}), px, zeros_like(px))


def test_start_after_last_data_is_not_started():
    d = pd.bdate_range("2020-01-02", periods=3)
    px = frame(d, A=[10.0] * 3)
    with pytest.raises(NotStartedYet):
        simulate(cfg("2021-01-01", {"A": 1.0}), px, zeros_like(px))


def test_depletion():
    d = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-03-02", "2020-04-01"])
    px = frame(d, A=[10.0] * 4)
    res = simulate(cfg(d[0], {"A": 1.0}, capital=15_000, monthly_withdrawal=10_000), px, zeros_like(px))
    assert res.daily["withdrawal"].sum() == pytest.approx(15_000)
    assert res.depleted_on == pd.Timestamp("2020-03-02")
    assert res.daily["value"].iloc[-1] == pytest.approx(0)


def test_interest_on_cash_compounds_daily():
    d = pd.bdate_range("2020-01-02", periods=11)
    px = frame(d, A=[10.0] * 11)
    rate = pd.Series(0.0504, index=d)        # 0.0504 / 252 = 0.0002 per day
    res = simulate(cfg(d[0], {}, rebalance_rule="none"), px, zeros_like(px), rate)
    assert res.daily["value"].iloc[-1] == pytest.approx(100_000 * 1.0002 ** 10)


def test_withdrawal_steps_up_each_year():
    d = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2021-02-01"])
    px = frame(d, A=[10.0] * 3)
    res = simulate(cfg(d[0], {"A": 1.0}, monthly_withdrawal=1_000, withdrawal_growth=0.03), px, zeros_like(px))
    assert res.daily.loc["2020-02-03", "withdrawal"] == pytest.approx(1_000)
    assert res.daily.loc["2021-02-01", "withdrawal"] == pytest.approx(1_030)


def test_threshold_rebalance():
    d = pd.bdate_range("2020-01-02", periods=3)
    px = frame(d, A=[10.0, 10.5, 15.0], B=[10.0, 10.0, 10.0])
    res = simulate(cfg(d[0], {"A": 0.5, "B": 0.5}, rebalance_rule="threshold", rebalance_band=0.05),
                   px, zeros_like(px))
    led = res.ledger
    # Day 2: A at 51.2% (no trade). Day 3: A at 60% -> rebalance.
    reb = led[led["note"].str.contains("band", na=False)]
    assert set(reb["date"]) == {d[2]}
    assert res.holdings_value.iloc[-1]["A"] == pytest.approx(res.holdings_value.iloc[-1]["B"])


def test_value_identity_holds():
    """End value = start + deposits - withdrawals - fees + income + price change, every day."""
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2018-01-02", "2021-12-31")
    px = frame(d, A=50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(d)))),
               B=30 * np.exp(np.cumsum(rng.normal(0, 0.005, len(d)))))
    dv = zeros_like(px)
    dv.loc[d[d.is_quarter_end], "A"] = 0.3
    rate = pd.Series(0.02, index=d)
    c = cfg(d[0], {"A": 0.55, "B": 0.4}, monthly_contribution=500, monthly_withdrawal=800,
            annual_fee=0.01, rebalance_rule="quarterly", dividend_policy="cash")
    res = simulate(c, px, dv, rate)
    daily, hv = res.daily, res.holdings_value
    price_gain = (hv.shift(1).fillna(0) / px.shift(1) * (px - px.shift(1))).fillna(0)
    # Recompute day-over-day: holdings before today's trades earn today's price change.
    total = (daily["value"].iloc[0] + daily["deposit"].iloc[1:].sum() - daily["withdrawal"].sum()
             - daily["fee"].sum() + daily["dividends"].sum() + daily["interest"].sum() + price_gain.sum().sum())
    assert daily["value"].iloc[-1] == pytest.approx(total, rel=1e-9)
