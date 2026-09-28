"""The forward paper ledger, and the one rule that gives it any value.

A backtest can be mined. This record cannot, because when each line is written
the outcome has not happened yet -- and that property survives exactly as long
as nobody lets the model learn from it. Most of what is checked here is that
seam: the ledger is append-only, the model never reads it, and the fill happens
at a price that existed after the prediction rather than before.

The rest is arithmetic that would be easy to get quietly wrong: weights that do
not sum to one, a missing price counted as a flat position, a second entry for
a day that already has one.
"""

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import paper                                      # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "LEDGER_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def panel():
    return synthetic_panel()


class FakeRun:
    run_id = "test-run"

    def __init__(self, as_of, probabilities, symbols, trusted=False):
        self.trust = {"trusted": trusted, "reason": "test"}
        self.signals = [
            {"symbol": s, "as_of": as_of, "probability_up": p,
             "confidence": abs(p - 0.5) * 2}
            for s, p in zip(symbols, probabilities)]


# --- the seam ---------------------------------------------------------------

def test_nothing_that_builds_features_can_reach_the_ledger():
    """The rule the whole page is about, enforced rather than intended.

    A model that trains on its own paper results turns the one un-mineable
    measurement in the project into another training set, and it would take
    months of calendar time to discover.
    """
    import inspect

    from trader import cross, dataset, evaluate, features, labels, macro

    for module in (dataset, features, labels, macro, cross, evaluate):
        source = inspect.getsource(module)
        assert "paper" not in source, (
            f"{module.__name__} references the paper ledger; the feature side "
            f"must not be able to read its own results")


def test_the_ledger_is_append_only(ledger, panel):
    symbols = sorted(panel)
    dates = list(panel[symbols[0]].index)

    paper.record_intent(FakeRun(dates[-10].date().isoformat(),
                                [0.6, 0.4, 0.55, 0.45, 0.52, 0.48], symbols))
    path, _ = paper._paths()
    before = open(path, encoding="utf-8").read()

    paper.record_intent(FakeRun(dates[-9].date().isoformat(),
                                [0.4, 0.6, 0.45, 0.55, 0.48, 0.52], symbols))
    after = open(path, encoding="utf-8").read()

    assert after.startswith(before), "an earlier entry was rewritten"
    assert len(after.splitlines()) == 2


def test_settling_does_not_edit_the_intent(ledger, panel):
    symbols = sorted(panel)
    dates = list(panel[symbols[0]].index)
    paper.record_intent(FakeRun(dates[-10].date().isoformat(),
                                [0.6, 0.4, 0.55, 0.45, 0.52, 0.48], symbols))

    path, _ = paper._paths()
    intent = open(path, encoding="utf-8").read().splitlines()[0]

    paper.settle(panel)
    assert open(path, encoding="utf-8").read().splitlines()[0] == intent


def test_a_second_entry_for_the_same_day_is_refused(ledger, panel):
    symbols = sorted(panel)
    as_of = list(panel[symbols[0]].index)[-10].date().isoformat()
    run = FakeRun(as_of, [0.6, 0.4, 0.55, 0.45, 0.52, 0.48], symbols)

    assert paper.record_intent(run) is not None
    assert paper.record_intent(run) is None, "the book was doubled"


# --- the fill ---------------------------------------------------------------

def test_the_fill_uses_a_price_from_after_the_signal(ledger, panel):
    """Entry at the next open, not at the close the signal was computed from.

    This is the discipline the whole project needed and could not enforce in a
    backtest: here it is structural, because the price simply does not exist
    yet when the line is written.
    """
    symbols = sorted(panel)
    frame = panel[symbols[0]]
    as_of = frame.index[-10]

    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.9, 0.1, 0.5, 0.5, 0.5, 0.5], symbols))
    paper.settle(panel)

    settlements = [e["settlement"] for e in paper._read_ledger()
                   if "settlement" in e]
    assert len(settlements) == 1
    mark = next(m for m in settlements[0]["marks"] if m["symbol"] == symbols[0])

    session = pd.Timestamp(mark["session"])
    assert session > as_of, "filled at or before the signal's own session"

    row = frame.loc[session]
    # The ledger rounds to six decimals so the file stays readable; compare at
    # the precision it actually stores.
    assert mark["open"] == pytest.approx(float(row["open"]), abs=1e-6)
    assert mark["close"] == pytest.approx(float(row["close"]), abs=1e-6)
    # And the return is open-to-close, not close-to-close.
    assert mark["gross_return"] == pytest.approx(
        float(row["close"] / row["open"] - 1.0), abs=1e-6)


def test_both_sides_of_the_round_trip_are_charged(ledger, panel):
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10]
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.9, 0.1, 0.5, 0.5, 0.5, 0.5], symbols))
    paper.settle(panel)

    mark = [e["settlement"] for e in paper._read_ledger()
            if "settlement" in e][0]["marks"][0]
    assert mark["net_return"] == pytest.approx(
        mark["gross_return"] - 2 * paper.COST_PER_SIDE, abs=1e-12)


def test_a_session_that_has_not_happened_is_left_pending(ledger, panel):
    symbols = sorted(panel)
    future = (panel[symbols[0]].index[-1] + pd.Timedelta(days=30))
    paper.record_intent(FakeRun(future.date().isoformat(),
                                [0.6] * 6, symbols))

    out = paper.settle(panel)
    assert out["settled"] == 0
    assert out["pending"] == 1


def test_a_missing_price_is_dropped_rather_than_counted_flat(ledger, panel):
    """Counting an unavailable name as a zero return would quietly dilute
    every other position toward nothing."""
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10]
    # The one that goes missing sits at a coin flip, so it carries almost none
    # of the book and the rest is still nearly all of it. When most of the book
    # is the part that is missing, the day is left pending instead -- no
    # position can exceed a quarter of the account, so three of six names can
    # never be 80% of it. See MIN_SETTLED_WEIGHT and the test below.
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.97, 0.03, 0.95, 0.05, 0.9, 0.52], symbols))

    partial = {s: panel[s] for s in symbols[:5]}
    paper.settle(partial)

    settlement = [e["settlement"] for e in paper._read_ledger()
                  if "settlement" in e][0]
    assert settlement["positions"] == 5
    assert set(settlement["missing"]) == set(symbols[5:])


# --- sizing -----------------------------------------------------------------

def test_weights_sum_to_one(panel):
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": p, "confidence": abs(p - 0.5) * 2}
               for s, p in zip(symbols, [0.7, 0.3, 0.55, 0.45, 0.52, 0.48])]

    positions = paper.size_positions(signals)
    assert sum(p.weight for p in positions) == pytest.approx(1.0)


def test_conviction_gets_more_money(panel):
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": p, "confidence": abs(p - 0.5) * 2}
               for s, p in zip(symbols, [0.70, 0.52, 0.51, 0.49, 0.48, 0.30])]

    by_symbol = {p.symbol: p for p in paper.size_positions(signals)}
    assert by_symbol[symbols[0]].weight > by_symbol[symbols[1]].weight


def test_no_single_name_can_take_the_account(panel):
    """A model briefly certain about one symbol must not be able to bet it all."""
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": p, "confidence": abs(p - 0.5) * 2}
               for s, p in zip(symbols, [0.999, 0.5001, 0.5, 0.5, 0.5, 0.4999])]

    positions = paper.size_positions(signals, max_weight=0.25)
    # The cap binds, and renormalising afterwards must not undo it by much.
    assert max(p.weight for p in positions) < 0.6
    assert sum(p.weight for p in positions) == pytest.approx(1.0)


def test_top_n_takes_both_ends(panel):
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": p, "confidence": abs(p - 0.5) * 2}
               for s, p in zip(symbols, [0.70, 0.65, 0.55, 0.45, 0.35, 0.30])]

    positions = paper.size_positions(signals, top_n=2)
    assert len(positions) == 4
    assert {p.direction for p in positions} == {1, -1}


def test_every_probability_at_a_coin_flip_does_not_divide_by_zero(panel):
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": 0.5, "confidence": 0.0}
               for s in symbols]

    positions = paper.size_positions(signals)
    assert sum(p.weight for p in positions) == pytest.approx(1.0)


# --- reading it back --------------------------------------------------------

def test_an_empty_account_has_the_full_shape(ledger):
    """Day one is the normal state, and every reader would otherwise need a
    special case -- which is how a page raises KeyError the first time."""
    state = paper.account()
    for key in ("starting_cash", "cash", "profit", "return", "days", "sharpe",
                "tstat", "max_drawdown", "equity", "pending", "verdict"):
        assert key in state


def test_a_short_record_refuses_to_conclude_anything(ledger, panel):
    symbols = sorted(panel)
    rng = np.random.default_rng(0)
    for date in panel[symbols[0]].index[-12:-2]:
        paper.record_intent(FakeRun(date.date().isoformat(),
                                    0.5 + rng.normal(0, 0.05, 6), symbols))
    paper.settle(panel)

    state = paper.account()
    assert state["days"] == 10
    assert "too few" in state["verdict"].lower()


def test_shadow_days_are_counted_and_named(ledger, panel):
    symbols = sorted(panel)
    date = panel[symbols[0]].index[-10]
    paper.record_intent(FakeRun(date.date().isoformat(), [0.6] * 6, symbols,
                                trusted=False))
    paper.settle(panel)

    assert paper.account()["shadow_days"] == 1


def test_divergence_needs_a_record_before_it_says_anything(ledger):
    out = paper.divergence({"days": 5, "sharpe": 3.0},
                           {"executable_sharpe": -2.0})
    assert not out["comparable"]
    assert "too few" in out["note"]


def test_divergence_flags_a_real_gap():
    out = paper.divergence({"days": 400, "sharpe": 2.0},
                           {"executable_sharpe": -2.5})
    assert out["comparable"]
    assert out["diverged"]
    assert "come apart" in out["note"]


def test_divergence_is_quiet_when_they_agree():
    out = paper.divergence({"days": 400, "sharpe": -2.3},
                           {"executable_sharpe": -2.5})
    assert not out["diverged"]
    assert "within what a record this short can tell apart" in out["note"]


# --- the ledger keeps two kinds of line, and both have to be read as such ------
#
# `settle` writes outcomes as new lines beside the intents rather than editing
# history, which is right -- but it left every reader walking a list of two
# different shapes. Found by the scheduler in auto.py on its first real cycle:
# it settled a position, which appended the first settlement line, and every
# write to the ledger from that moment on raised KeyError('as_of').

def test_an_intent_can_still_be_recorded_once_a_settlement_exists(ledger, panel):
    """Otherwise the forward record stops growing the day it first pays out,
    and the only sign is a warning in a log nobody is reading."""
    symbols = sorted(panel)
    dates = list(panel[symbols[0]].index)

    paper.record_intent(FakeRun(dates[-10].date().isoformat(),
                                [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))
    assert paper.settle(panel)["settled"] == 1

    written = paper.record_intent(FakeRun(dates[-5].date().isoformat(),
                                          [0.4, 0.6, 0.45, 0.55, 0.48, 0.52],
                                          symbols))
    assert written is not None
    assert written["as_of"] == dates[-5].date().isoformat()


def test_settling_again_does_not_settle_the_same_day_twice(ledger, panel):
    """A scheduler settles every half hour. Re-marking a day that already has
    an outcome would append a second settlement and count the same return
    twice in the P&L."""
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10]
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))

    first = paper.settle(panel)
    assert first["settled"] == 1

    again = paper.settle(panel)
    assert again["settled"] == 0
    assert again["pending"] == 0

    settlements = [e for e in paper._read_ledger() if "settlement" in e]
    assert len(settlements) == 1
    assert paper.account()["days"] == 1


# --- settling the right book, with the right prices ---------------------------
#
# Found by reading the real ledger: an intent over 238 names was settled against
# ten of them, because `settle_paper` loaded prices for the watchlist it was
# handed rather than for the symbols in the ledger. The ten carried 3.2% of the
# book, were renormalised to 100%, and became the P&L page's only day.

def test_a_book_that_is_mostly_unpriced_is_left_pending_not_rescaled(ledger, panel):
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10]
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.95, 0.05, 0.9, 0.1, 0.85, 0.15], symbols))

    out = paper.settle({symbols[0]: panel[symbols[0]]})

    assert out["settled"] == 0
    assert out["pending"] == 1
    assert not [e for e in paper._read_ledger() if "settlement" in e]
    assert paper.account()["days"] == 0


# --- correcting the record without editing it ---------------------------------

def test_a_voided_intent_is_ignored_by_everything(ledger, panel):
    """A test fixture once wrote six synthetic symbols into the real ledger.
    They can never price, so they sat in `pending` for ever."""
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10].date().isoformat()
    paper.record_intent(FakeRun(as_of, [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))

    paper.void(as_of, reason="written by a test", kind="intent")

    assert paper.settle(panel)["pending"] == 0
    assert paper.account()["pending"] == 0
    assert paper.account()["days"] == 0


def test_a_voided_settlement_is_dropped_and_can_be_settled_again(ledger, panel):
    """The correction path for the real one: a day settled against the wrong
    prices is voided, and the next pass fills it properly."""
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10].date().isoformat()
    paper.record_intent(FakeRun(as_of, [0.97, 0.03, 0.95, 0.52, 0.48, 0.53],
                                symbols))
    paper.settle({s: panel[s] for s in symbols[:3]})
    wrong = paper.account()["return"]

    paper.void(as_of, reason="settled against three of six names",
               kind="settlement")
    assert paper.account()["days"] == 0

    assert paper.settle(panel)["settled"] == 1
    state = paper.account()
    assert state["days"] == 1
    assert state["return"] != wrong

    settlement = [e["settlement"] for e in paper._read_ledger()
                  if "settlement" in e][-1]
    assert settlement["positions"] == 6


def test_the_void_is_appended_like_everything_else(ledger, panel):
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10].date().isoformat()
    paper.record_intent(FakeRun(as_of, [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))

    path, _ = paper._paths()
    before = open(path, encoding="utf-8").read()
    paper.void(as_of, reason="testing", kind="intent")
    after = open(path, encoding="utf-8").read()

    assert after.startswith(before)
    assert json.loads(after.splitlines()[-1])["void"]["reason"] == "testing"


def test_the_symbols_to_price_come_from_the_ledger(ledger, panel):
    """What `settle_paper` needs to ask the price feed for is not a watchlist:
    it is whatever the unsettled entries actually hold."""
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10].date().isoformat()
    paper.record_intent(FakeRun(as_of, [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))

    assert paper.pending_symbols() == set(symbols)

    paper.settle(panel)
    assert paper.pending_symbols() == set()


def test_a_name_at_an_exact_coin_flip_is_not_in_the_book(panel):
    """Its weight rounds to zero, so it is not held -- but it was still being
    written as a position and counted in the hit rate."""
    symbols = sorted(panel)
    signals = [{"symbol": s, "probability_up": p, "confidence": abs(p - 0.5) * 2}
               for s, p in zip(symbols, [0.7, 0.3, 0.6, 0.4, 0.55, 0.5])]

    positions = paper.size_positions(signals)

    assert symbols[5] not in {p.symbol for p in positions}
    assert len(positions) == 5
    assert sum(p.weight for p in positions) == pytest.approx(1.0)


# --- what a real broker takes --------------------------------------------------
#
# The ledger charges five basis points a side, which is spread and slippage and
# not commission. Avanza and Nordnet charge a percentage or a minimum, whichever
# is larger, and on a book of 238 names in a 500 account -- a position of about
# two -- the minimum is the entire cost.

def test_the_minimum_decides_the_bill_on_a_small_position():
    from trader import broker

    tiny = broker.describe(2.10, "avanza_mini_us")
    assert tiny["minimum_binds"]
    # A dollar each way plus currency, against a position of 2.10.
    assert tiny["round_trip_fraction"] > 0.9

    large = broker.describe(5000.0, "avanza_mini_us")
    assert not large["minimum_binds"]
    assert large["round_trip_fraction"] == pytest.approx(2 * (0.0025 + 0.0025),
                                                         rel=1e-6)


def test_the_break_even_is_where_the_percentage_overtakes_the_minimum():
    from trader import broker

    where = broker.break_even_value("avanza_mini_us")
    assert where == pytest.approx(1.0 / 0.0025)
    assert broker.per_side(where, "avanza_mini_us") == pytest.approx(
        where * 0.0025 + where * 0.0025)


def test_a_free_schedule_still_charges_its_currency_markup():
    from trader import broker

    assert broker.per_side(1000.0, "avanza_start_se") == 0.0
    assert broker.per_side(1000.0, "nordnet_mini_nordic") == pytest.approx(2.5)


def test_an_unknown_schedule_is_refused_by_name():
    from trader import broker

    with pytest.raises(ValueError, match="unknown schedule"):
        broker.resolve("my_mates_broker")


def test_the_account_prices_itself_against_real_schedules(ledger, panel):
    symbols = sorted(panel)
    as_of = panel[symbols[0]].index[-10]
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))
    paper.settle(panel)

    state = paper.account()
    named = {b["schedule"]: b for b in state["brokers"]}
    assert len(state["brokers"]) == len(paper.DEFAULT_BROKERS)

    spread = [b for b in state["brokers"] if "spread" in b["schedule"]][0]
    avanza = [b for b in state["brokers"] if "Avanza" in b["schedule"]][0]

    assert avanza["trades"] == 6
    assert avanza["commission_paid"] > spread["commission_paid"]
    # Six positions of about 83 each, a dollar minimum a side: the minimum
    # binds and the bill is far more than the spread-only figure.
    assert avanza["profit"] < spread["profit"]
    assert avanza["source"].startswith("avanza.se")
    assert named[avanza["schedule"]]["break_even_value"] == pytest.approx(400.0)
