"""The held book: bought once, left alone, sold on the session it said it would.

The day-trade ledger next door cannot be run at this account size -- 238 names
on 500 pays 478 of commission on day one and there is no day two. What the
search committed instead is one position rebalanced every sixty sessions, so
this record has to do something the other one never did: hold, and know when it
intends to stop.

What is pinned here is the seam that makes it a forward record rather than a
story: the plan is written before the price it will be filled at exists, the
fill happens at the next open, and the sale happens on the session the plan
named -- not on the day the number looks good.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import holding                                    # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(holding, "STORE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def panel():
    return synthetic_panel()


class FakeRun:
    run_id = "run-1"

    def __init__(self, as_of, picks, trusted=False):
        self.trust = {"trusted": trusted, "reason": "test"}
        self.signals = [
            {"symbol": symbol, "as_of": as_of, "probability_up": probability,
             "confidence": abs(probability - 0.5) * 2}
            for symbol, probability in picks]


def sessions_of(panel, symbol=None):
    symbol = symbol or sorted(panel)[0]
    return list(panel[symbol].index)


# --- the seam -----------------------------------------------------------------

def test_the_plan_is_written_before_the_price_it_will_pay(store, panel):
    symbols = sorted(panel)
    as_of = sessions_of(panel)[-40]

    plan = holding.plan_next(FakeRun(as_of.date().isoformat(),
                                     [(symbols[0], 0.8), (symbols[1], 0.3)]))

    assert plan["buy"][0]["symbol"] == symbols[0]
    assert plan["as_of"] == as_of.date().isoformat()
    # Nothing about a price yet: the session it will buy in has not opened.
    assert "price" not in plan["buy"][0]


def test_it_fills_at_the_first_open_after_the_plan(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    as_of = dates[-40]

    holding.plan_next(FakeRun(as_of.date().isoformat(), [(symbols[0], 0.9)]))
    out = holding.advance(panel)

    bought = out["filled"]["bought"][0]
    assert pd.Timestamp(bought["session"]) == dates[-39]
    assert bought["price"] == pytest.approx(float(panel[symbols[0]].loc[dates[-39]]["open"]))


def test_a_plan_whose_session_has_not_happened_waits(store, panel):
    symbols = sorted(panel)
    future = (sessions_of(panel)[-1] + pd.Timedelta(days=10)).date().isoformat()

    holding.plan_next(FakeRun(future, [(symbols[0], 0.9)]))
    out = holding.advance(panel)

    assert out["filled"] is None
    assert "has not closed yet" in out["note"]


# --- holding, which is most of what it does -----------------------------------

def test_it_holds_and_says_when_it_will_decide_again(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]),
                      book={"rebalance_every": 20})
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-39])

    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-35])

    assert out["exited"] is None
    assert out["holding"]["sessions_left"] == 16
    assert out["holding"]["review_on"] == dates[-19].date().isoformat()
    assert "until it decides again" in out["note"]


def test_holding_costs_nothing_while_it_holds(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    holding.advance(panel, today=dates[-39])

    before = len(holding._read())
    for day in range(5):
        holding.advance(panel, today=dates[-38 + day])
    assert len(holding._read()) == before, "holding wrote to the ledger"


def test_it_sells_on_the_session_the_plan_named(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]),
                      book={"rebalance_every": 20})
    filled = holding.advance(panel, book={"rebalance_every": 20},
                             today=dates[-39])["filled"]

    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19])

    sold = out["exited"]["sold"][0]
    assert sold["session"] == dates[-19].date().isoformat()
    assert sold["price"] == pytest.approx(float(panel[symbols[0]].loc[dates[-19]]["open"]))
    # Twenty sessions after it bought, which is what the plan said.
    assert sold["bought_at"] == filled["bought"][0]["price"]
    assert out["exited"]["cash_after"] > 0


def test_one_position_at_a_time(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    run = FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)])

    assert holding.plan_next(run) is not None
    assert holding.plan_next(run) is None, "planned twice"
    holding.advance(panel)
    assert holding.plan_next(FakeRun(dates[-30].date().isoformat(),
                                     [(symbols[1], 0.95)])) is None


def test_nothing_it_calls_up_means_it_stays_in_cash(store, panel):
    symbols = sorted(panel)
    as_of = sessions_of(panel)[-40].date().isoformat()

    assert holding.plan_next(FakeRun(as_of, [(symbols[0], 0.3),
                                             (symbols[1], 0.45)])) is None
    assert holding.state()["plan"] is None


# --- the money ----------------------------------------------------------------

def test_the_round_trip_is_charged_at_both_ends(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]),
                      book={"rebalance_every": 20})
    filled = holding.advance(panel, book={"rebalance_every": 20},
                             today=dates[-39])["filled"]
    exited = holding.advance(panel, book={"rebalance_every": 20},
                             today=dates[-19])["exited"]

    from trader import broker
    expected = broker.per_side(500.0, holding.SCHEDULE) + 500.0 * holding.SPREAD
    assert filled["fees"] == pytest.approx(expected, rel=1e-6)
    assert exited["fees"] > 0
    assert filled["cash_after"] == pytest.approx(500.0 - filled["fees"])


def test_the_account_shows_the_pick_and_the_sell_date(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]),
                      book={"rebalance_every": 20})
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-39])

    state = holding.account(panel, today=dates[-35])

    assert state["position"]["marks"][0]["symbol"] == symbols[0]
    assert state["position"]["review_on"] == dates[-19].date().isoformat()
    assert state["position"]["sessions_left"] > 0
    assert state["equity"] == pytest.approx(
        state["cash"] + state["position"]["worth"], abs=0.01)


def test_an_empty_account_has_the_whole_shape(store):
    state = holding.account({})
    for key in ("starting_cash", "cash", "equity", "profit", "return", "realised",
                "fees", "trades", "book", "waiting_to_buy", "position", "closed"):
        assert key in state
    assert state["equity"] == holding.STARTING_CASH


def test_the_record_is_append_only(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    before = open(holding._path(), encoding="utf-8").read()

    holding.advance(panel)
    after = open(holding._path(), encoding="utf-8").read()
    assert after.startswith(before)


def test_nothing_that_builds_features_can_reach_this_record():
    """Checked on the imports: evaluate.py discusses holding things in prose,
    which a grep for the word cannot tell apart from importing this."""
    import ast
    import inspect

    from trader import cross, dataset, evaluate, features, labels, macro

    for module in (dataset, features, labels, macro, cross, evaluate):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        assert not {name for name in imported if "holding" in name}, module.__name__


def test_a_name_is_billed_where_it_is_listed(store, panel):
    """The first position this book ever planned was a Stockholm share, and the
    US price list would have charged twice as much plus a currency markup."""
    from trader import broker

    assert holding.schedule_for("INVE-B.ST") == "avanza_mini_se"
    assert holding.schedule_for("AAPL") == "avanza_mini_us"
    assert holding.schedule_for("SAP.DE") == "avanza_mini_de"

    swedish = broker.per_side(500.0, holding.schedule_for("INVE-B.ST"))
    american = broker.per_side(500.0, holding.schedule_for("AAPL"))
    assert swedish < american
    # No currency to cross for a Swedish share held in a Swedish account.
    assert broker.resolve(holding.schedule_for("INVE-B.ST")).fx_per_side == 0.0
