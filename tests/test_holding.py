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

import json
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

    # The whole allocation leaves the account: the fee is paid out of it and
    # the rest buys shares. This line used to read `500 - fees`, which left the
    # money that bought the shares sitting in cash as well -- a $500 book with
    # one position marked itself at $991 the next session.
    bought = filled["bought"][0]
    assert filled["cash_after"] == pytest.approx(500.0 - bought["value"])
    assert bought["shares"] * bought["price"] == pytest.approx(
        bought["value"] - bought["fee"], rel=1e-6)


def test_a_filled_book_is_worth_what_it_paid_before_the_price_moves(store, panel):
    """Cash plus shares at the entry price is the account, not twice it.

    The arithmetic that caught this: 1.21 shares of a 412.00 name is 498.50,
    and the book had 498.50 recorded as cash as well. Every mark after a fill
    was therefore about double, which on the one record here that cannot be
    re-run is the worst place for a number to be wrong.
    """
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    filled = holding.advance(panel, today=dates[-39])["filled"]

    at_cost = sum(b["shares"] * b["price"] for b in filled["bought"])
    assert filled["cash_after"] + at_cost + filled["fees"] == pytest.approx(
        holding.STARTING_CASH, abs=0.01)

    marked = holding.mark(panel, today=dates[-39])
    assert marked["equity"] <= holding.STARTING_CASH * 1.05


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


# --- what happens at a review --------------------------------------------------
#
# A review is not automatically a sale. Selling in order to buy the same stock
# back pays a round trip to stand still, and the strategy that was tested never
# did it -- the simulation only ever charged for a change.

def held_for(panel, symbols, every=20):
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]),
                      book={"rebalance_every": every})
    holding.advance(panel, book={"rebalance_every": every}, today=dates[-39])
    return dates


def test_a_review_that_still_likes_the_name_trades_nothing(store, panel):
    symbols = sorted(panel)
    dates = held_for(panel, symbols)
    lines = len(holding._read())

    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                          run=FakeRun(dates[-19].date().isoformat(),
                                      [(symbols[0], 0.88), (symbols[1], 0.4)]))

    assert out["exited"] is None
    assert out["renewed"]["symbol"] == symbols[0]
    assert "nothing was traded" in out["note"]
    # Lines may be appended -- a review is an event, and so is a change in the
    # watch -- but none of them may be a trade.
    trades = [e for e in holding._read() if "fill" in e or "exit" in e]
    assert len(trades) == 1, "it traded at a review that changed nothing"
    assert holding.account(panel, today=dates[-19])["fees"] == pytest.approx(
        holding.state()["holding"]["fees"])
    assert len(holding._read()) > lines


def test_a_review_that_prefers_another_name_sells(store, panel):
    symbols = sorted(panel)
    dates = held_for(panel, symbols)

    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                          run=FakeRun(dates[-19].date().isoformat(),
                                      [(symbols[0], 0.51), (symbols[1], 0.93)]))

    assert out["renewed"] is None
    assert out["exited"] is not None
    assert symbols[1] in out["note"]
    assert holding.state()["holding"] is None


def test_a_review_with_nothing_worth_buying_goes_to_cash(store, panel):
    symbols = sorted(panel)
    dates = held_for(panel, symbols)

    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                          run=FakeRun(dates[-19].date().isoformat(),
                                      [(s, 0.4) for s in symbols]))

    assert out["exited"] is not None
    assert "nothing is called up" in out["note"]


def test_a_renewed_position_waits_another_full_interval(store, panel):
    symbols = sorted(panel)
    dates = held_for(panel, symbols)
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                    run=FakeRun(dates[-19].date().isoformat(), [(symbols[0], 0.9)]))

    # The clock restarts from the review session, so it is not due again.
    out = holding.advance(panel, book={"rebalance_every": 20}, today=dates[-10],
                          run=FakeRun(dates[-10].date().isoformat(),
                                      [(symbols[1], 0.99)]))
    assert out["exited"] is None
    assert out["holding"]["sessions_left"] > 0


def test_it_ignores_the_model_entirely_between_reviews(store, panel):
    """The deafness is the strategy: a better-looking name tomorrow is not a
    reason to pay a round trip today."""
    symbols = sorted(panel)
    dates = held_for(panel, symbols, every=60)
    lines = len(holding._read())

    for day in range(1, 10):
        out = holding.advance(panel, book={"rebalance_every": 60},
                              today=dates[-39 + day],
                              run=FakeRun(dates[-39 + day].date().isoformat(),
                                          [(symbols[1], 0.99)]))
        assert out["exited"] is None and out["renewed"] is None

    assert len(holding._read()) == lines


# --- watching the case it bought on --------------------------------------------
#
# A warning is free and a change of mind is a round trip, so the two are kept
# apart: this re-reads today's reasons for the name already held and says how
# much of the original argument survives. It never trades on what it finds.

def reasoned(symbol, probability, drivers):
    """A signal carrying per-feature contributions, the way a run records them."""
    return {"symbol": symbol, "as_of": "2026-09-25",
            "probability_up": probability,
            "confidence": abs(probability - 0.5) * 2,
            "all_contributions": [{"feature": name, "effect": effect, "z": -1.0}
                                  for name, effect in drivers.items()]}


class ReasonedRun:
    run_id = "run-2"
    trust = {"trusted": False, "reason": "test"}

    def __init__(self, signals):
        self.signals = signals


def bought_on(store, panel, symbol, drivers, probability=0.9, every=60):
    dates = sessions_of(panel)
    run = ReasonedRun([reasoned(symbol, probability, drivers)])
    run.signals[0]["as_of"] = dates[-40].date().isoformat()
    holding.plan_next(run, book={"rebalance_every": every})
    holding.advance(panel, book={"rebalance_every": every}, today=dates[-39])
    return dates


def test_it_remembers_why_it_bought(store, panel):
    symbols = sorted(panel)
    bought_on(store, panel, symbols[0], {"rsi_14": 0.03, "volatility_20d": 0.02,
                                         "gold_return_20d": -0.05})

    pattern = holding.state()["holding"]["bought"][0]["pattern"]
    # Only the reasons that argued for the buy.
    assert [d["feature"] for d in pattern] == ["rsi_14", "volatility_20d"]


def test_an_intact_pattern_says_so_without_writing_every_day(store, panel):
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03, "volatility_20d": 0.02})
    same = ReasonedRun([reasoned(symbols[0], 0.9, {"rsi_14": 0.03,
                                                   "volatility_20d": 0.02})])

    first = holding.advance(panel, book={"rebalance_every": 60},
                            today=dates[-38], run=same)
    lines = len(holding._read())
    second = holding.advance(panel, book={"rebalance_every": 60},
                             today=dates[-37], run=same)

    assert first["watch"]["status"] == "intact"
    assert second["watch"]["status"] == "intact"
    assert len(holding._read()) == lines, "it wrote a diary entry for no news"


def test_a_broken_pattern_warns_and_does_not_sell(store, panel):
    """The point of the whole thing: notice, say so, hold anyway."""
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03, "volatility_20d": 0.02})

    gone = ReasonedRun([reasoned(symbols[0], 0.62, {"rsi_14": -0.01,
                                                    "volatility_20d": -0.02})])
    out = holding.advance(panel, book={"rebalance_every": 60}, today=dates[-38],
                          run=gone)

    assert out["watch"]["status"] == "broken"
    assert "have gone" in out["watch"]["note"]
    assert out["exited"] is None, "a warning is not a reason to pay a round trip"
    assert holding.state()["holding"] is not None


def test_the_model_turning_against_it_is_a_break(store, panel):
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03})

    turned = ReasonedRun([reasoned(symbols[0], 0.41, {"rsi_14": 0.03})])
    out = holding.advance(panel, book={"rebalance_every": 60}, today=dates[-38],
                          run=turned)

    assert out["watch"]["status"] == "broken"
    assert "no longer calls it up" in out["watch"]["note"]
    assert out["exited"] is None


def test_a_weakening_case_reads_as_drifting(store, panel):
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03, "volatility_20d": 0.02})

    weaker = ReasonedRun([reasoned(symbols[0], 0.7, {"rsi_14": 0.03,
                                                     "volatility_20d": -0.01})])
    out = holding.advance(panel, book={"rebalance_every": 60}, today=dates[-38],
                          run=weaker)

    assert out["watch"]["status"] == "drifting"
    assert out["watch"]["agreement"] == pytest.approx(0.5)


def test_the_warning_is_forgotten_when_the_position_is(store, panel):
    """A warning belongs to a position. Selling ends both."""
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03}, every=20)

    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-38],
                    run=ReasonedRun([reasoned(symbols[0], 0.41, {"rsi_14": 0.03})]))
    assert holding.state()["watch"]["status"] == "broken"

    # The review arrives and the model prefers something else, so it sells.
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                    run=ReasonedRun([reasoned(symbols[1], 0.95, {"rsi_14": 0.04})]))
    assert holding.state()["holding"] is None
    assert holding.state()["watch"] is None


# --- the record, read aloud ----------------------------------------------------

def test_the_log_says_what_it_did_in_order(store, panel):
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03}, every=20)
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-38],
                    run=ReasonedRun([reasoned(symbols[0], 0.41, {"rsi_14": 0.03})]))
    holding.advance(panel, book={"rebalance_every": 20}, today=dates[-19],
                    run=ReasonedRun([reasoned(symbols[1], 0.95, {"rsi_14": 0.04})]))

    entries = holding.log()
    kinds = [e["kind"] for e in entries]
    assert kinds == ["sold", "broken", "bought", "decided"], kinds

    newest = entries[0]
    assert newest["text"].startswith(f"Sold {symbols[0]} at")
    assert "after costs" in newest["text"]
    assert all(e["when"] for e in entries)


def test_the_log_explains_a_warning_without_claiming_it_acted(store, panel):
    symbols = sorted(panel)
    dates = bought_on(store, panel, symbols[0], {"rsi_14": 0.03}, every=60)
    holding.advance(panel, book={"rebalance_every": 60}, today=dates[-38],
                    run=ReasonedRun([reasoned(symbols[0], 0.41, {"rsi_14": 0.03})]))

    warning = [e for e in holding.log() if e["kind"] == "broken"][0]
    assert "has broken" in warning["text"]
    assert "Holding anyway" in warning["text"]


def test_an_empty_record_logs_nothing(store):
    assert holding.log() == []


# --- many books, one of them funded --------------------------------------------
#
# A challenger has to earn a forward record the same way the champion does:
# same machinery, same costs, same forward-only discipline. The only difference
# is that nobody would have traded it. So every line carries the name of the
# book it belongs to, and the books keep separate accounts.

def test_the_record_written_before_books_had_names_belongs_to_the_champion(
        store, panel):
    """The one thing here that cannot be recreated is the record already built."""
    symbols = sorted(panel)
    dates = sessions_of(panel)
    # A line in the old shape: no book_id at all.
    path = holding._path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"plan": {
            "at": "2026-09-25T00:00:00+00:00", "as_of": dates[-40].date().isoformat(),
            "run_id": "old", "book": holding.BOOK, "trusted": False,
            "buy": [{"symbol": symbols[0], "weight": 1.0,
                     "probability_up": 0.9, "pattern": []}]}}) + "\n")

    assert holding.state()["plan"]["run_id"] == "old"
    assert holding.state(holding.CHAMPION)["plan"]["run_id"] == "old"


def test_a_challenger_keeps_its_own_account(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.register("wide-5", rule={"top_n": 2, "rebalance_every": 20},
                     why="does spreading it thinner survive the fees?")

    run = FakeRun(dates[-40].date().isoformat(),
                  [(symbols[0], 0.9), (symbols[1], 0.85)] + [(s, 0.3) for s in symbols[2:]])
    holding.plan_all(run)
    holding.advance_all(panel, today=dates[-39])

    champion = holding.state(holding.CHAMPION)
    challenger = holding.state("wide-5")

    assert len(champion["holding"]["bought"]) == 1
    assert len(challenger["holding"]["bought"]) == 2

    # Separate accounts: each spent its own $500 on its own names. This used to
    # be checked by comparing leftover cash, which only differed because the
    # fill was failing to debit what it spent -- with that fixed both books are
    # fully invested, so the thing to check is that they hold different things
    # and are marked apart.
    assert champion["holding"]["bought"][0]["symbol"] != tuple(
        b["symbol"] for b in challenger["holding"]["bought"])
    assert {b["symbol"] for b in challenger["holding"]["bought"]} != {
        b["symbol"] for b in champion["holding"]["bought"]}
    for book in (champion, challenger):
        spent = sum(b["value"] for b in book["holding"]["bought"])
        assert book["cash"] == pytest.approx(holding.STARTING_CASH - spent, abs=0.01)
    assert holding.account(panel, book_id="wide-5")["equity"] > 0


def test_only_one_book_is_funded(store):
    holding.register("challenger", rule={"top_n": 3}, why="testing")
    assert holding.funded() == holding.CHAMPION
    assert holding.books()["challenger"]["funded"] is False

    holding.fund("challenger", why="it won the contest")
    assert holding.funded() == "challenger"
    assert holding.books()[holding.CHAMPION]["funded"] is False

    with pytest.raises(ValueError, match="not in the contest"):
        holding.fund("nobody", why="typo")


def test_a_book_cannot_join_twice(store):
    holding.register("twice", rule={"top_n": 2}, why="first")
    with pytest.raises(ValueError, match="already in the contest"):
        holding.register("twice", rule={"top_n": 3}, why="second")


def test_prices_are_fetched_for_every_book(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.register("other", rule={"top_n": 1}, why="testing")

    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[3], 0.95)]),
                      book_id="other")

    assert holding.symbols_to_price() == {symbols[0], symbols[3]}


def test_each_book_follows_its_own_rule(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.register("quick", rule={"top_n": 1, "rebalance_every": 20},
                     why="does reviewing sooner help?")

    run = FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)])
    holding.plan_all(run)
    holding.advance_all(panel, today=dates[-39])

    # The champion reviews every 60 sessions, the challenger every 20, so only
    # one of them is due at the same moment.
    stepped = holding.advance_all(panel, today=dates[-19],
                                  run=FakeRun(dates[-19].date().isoformat(),
                                              [(symbols[1], 0.99)]))
    assert stepped[holding.CHAMPION]["exited"] is None
    assert stepped["quick"]["exited"] is not None


def test_the_contest_puts_the_funded_book_first(store, panel):
    holding.register("shadow", rule={"top_n": 2}, why="testing")
    listed = holding.accounts(panel)

    assert [book["book_id"] for book in listed][0] == holding.CHAMPION
    assert listed[0]["funded"] is True
    assert {book["book_id"] for book in listed} == {holding.CHAMPION, "shadow"}
    assert listed[1]["why"] == "testing"


def test_a_log_belongs_to_its_own_book(store, panel):
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.register("other", rule={"top_n": 1}, why="testing")
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[3], 0.95)]),
                      book_id="other")

    assert symbols[0] in holding.log()[0]["text"]
    assert symbols[3] in holding.log(book_id="other")[0]["text"]
    assert len(holding.log()) == 1


# --- correcting a record that must not be edited ------------------------------

def test_a_void_strikes_a_line_and_everything_after_it(store, panel):
    """The file is append-only, so a correction is a line in it.

    This exists because a real fill was written with the wrong cash, and the
    mark that followed it inherited the error. Editing the file would have made
    the record unfalsifiable; striking forward from the bad line leaves both
    the mistake and the correction readable.
    """
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    filled = holding.advance(panel, today=dates[-39])["filled"]
    holding.mark(panel, today=dates[-39])

    assert holding.state()["holding"] is not None
    assert holding.marks()

    holding.void(filled["at"], kind="fill",
                 reason="filled against the wrong cash")

    # The book is back to the plan it had before the fill, and can act again.
    after = holding.state()
    assert after["holding"] is None
    assert after["plan"] is not None
    assert holding.marks() == []


def test_a_void_needs_a_reason(store):
    with pytest.raises(ValueError, match="reason"):
        holding.void("2026-01-01T00:00:00+00:00", kind="fill", reason="")


def test_the_struck_lines_are_still_in_the_file(store, panel):
    """Struck, not deleted -- otherwise the correction is just an edit."""
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    filled = holding.advance(panel, today=dates[-39])["filled"]
    holding.void(filled["at"], kind="fill", reason="testing")

    raw = open(holding._path(), encoding="utf-8").read()
    assert "fill" in raw and "void" in raw and "testing" in raw


def test_a_void_does_not_strike_what_happens_next(store, panel):
    """A void is a line, not a mode. What the book does afterwards stands.

    The first implementation filtered the void lines out and then truncated,
    which loses where the void sat in the file -- so the corrected fill written
    straight after it was dropped too, and the book looked like it had silently
    refused to act.
    """
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    bad = holding.advance(panel, today=dates[-39])["filled"]
    holding.void(bad["at"], kind="fill", reason="wrong cash")

    again = holding.advance(panel, today=dates[-39])["filled"]
    assert again is not None
    assert holding.state()["holding"] is not None
    assert holding.state()["holding"]["at"] == again["at"]


def test_a_position_with_no_recorded_case_gets_no_warning(store, panel):
    """"0 of 0 reasons still hold" is a verdict computed from nothing.

    A plan written before the pattern was carried leaves the position with an
    empty case, and the comparison then reports a break against no evidence.
    The warning is only worth anything because it is read off the record, so
    with no record it says nothing.
    """
    symbols = sorted(panel)
    dates = sessions_of(panel)
    holding.plan_next(FakeRun(dates[-40].date().isoformat(), [(symbols[0], 0.9)]))
    holding.advance(panel, today=dates[-39])

    held = holding.state()["holding"]
    held["bought"][0]["pattern"] = []

    run = FakeRun(dates[-38].date().isoformat(), [(symbols[0], 0.40)])
    assert holding.watch(held, run, None) is None
    assert not [e for e in holding._lines() if "watch" in e]
