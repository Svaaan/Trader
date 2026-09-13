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
    paper.record_intent(FakeRun(as_of.date().isoformat(),
                                [0.9, 0.1, 0.6, 0.4, 0.55, 0.45], symbols))

    partial = {s: panel[s] for s in symbols[:3]}
    paper.settle(partial)

    settlement = [e["settlement"] for e in paper._read_ledger()
                  if "settlement" in e][0]
    assert settlement["positions"] == 3
    assert set(settlement["missing"]) == set(symbols[3:])


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
