"""Smoke tests: drive the Streamlit app headlessly on synthetic data and check every page renders."""
import datetime
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture()
def at(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVISOR_DATA_SOURCE", "synthetic")
    monkeypatch.setenv("ADVISOR_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("ADVISOR_SYNTHETIC_END", "2026-11-13")
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception
    return app


def ok(app):
    assert not app.exception, app.exception
    errors = [e.value for e in app.error]
    assert not errors, errors


def click(app, label, index=0):
    [b for b in app.button if b.label == label][index].click().run()
    ok(app)


def test_examples_and_every_page(at):
    click(at, "Load example clients")
    assert len([b for b in at.button if b.label == "Open"]) == 2
    for i in range(2):
        at.sidebar.radio[0].set_value("All clients").run()
        click(at, "Open", i)
        assert at.sidebar.radio[0].value == "Client"
        assert len(at.tabs) == 6
    at.sidebar.radio[0].set_value("Securities & data").run()
    ok(at)
    at.sidebar.radio[0].set_value("New client").run()
    ok(at)


def test_create_client_via_form(at):
    at.sidebar.radio[0].set_value("New client").run()
    at.text_input(key="nc_name").input("Test Client")
    at.date_input(key="nc_start").set_value(datetime.date(2010, 6, 1))
    at.number_input(key="nc_wd").set_value(2000.0)
    click(at, "Create client and invest")
    assert at.sidebar.radio[0].value == "Client"
    assert at.title[0].value == "Test Client"


def test_plan_change_is_saved_and_applied(at):
    from core import db
    click(at, "Load example clients")
    click(at, "Open", 0)      # retiree (sorted first by name)
    cid = at.session_state["client_pick"]
    note_key = f"pf_note_{cid}"
    at.date_input(key=f"pf_eff_{cid}").set_value(datetime.date(2009, 3, 9))
    at.text_area(key=note_key).input("Rebalance into equities after the crash")
    click(at, "Save plan change")
    versions = db.list_plan_versions(cid)
    assert [v["effective_date"] for v in versions] == ["2005-01-03", "2009-03-09"]
    assert any("Plan change" in str(getattr(m, "value", "")) or "Rebalance into equities" in str(getattr(m, "value", ""))
               for m in at.markdown)


def test_ticker_before_inception_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("ADVISOR_DATA_SOURCE", "synthetic")
    monkeypatch.setenv("ADVISOR_DB", str(tmp_path / "t.db"))
    from core import data, db, service
    db.init_db()
    client = {"start_date": "2005-01-03", "initial_capital": 100000, "monthly_contribution": 0,
              "monthly_withdrawal": 0, "withdrawal_growth_pct": 0, "advisory_fee_pct": 1, "dividend_policy": "reinvest",
              "rebalance_rule": "annual", "rebalance_band_pct": 5, "withdrawal_rule": "pro_rata",
              "benchmark": "SPY:60, AGG:40"}
    versions = [{"effective_date": "2005-01-03", "weights": {"SPY": 60, "BND": 40}, "note": ""}]
    data.refresh(service.client_tickers(client, versions))
    out = service.run_client(client, versions)
    assert out["status"] == "error"
    assert "BND has no price on 2005-01-03" in out["message"]
