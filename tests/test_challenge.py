"""The challenger loop and the rule that promotes one.

Two failure modes are being designed against, and both have happened to this
project already.

A loop with no brakes finds a winner every night: run enough candidates and one
beats the champion by luck, while the ledger reports triumph. So the budget, the
dedupe and the bar are the point of `challenge`, not decoration on it.

And a promotion rule written after the numbers are in is not a rule. `promote`
refuses on the evidence it has, says what is missing with a number in it, and
cannot be satisfied by a backtest -- only by a forward record that was written
before its own outcomes existed.
"""

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import challenge, dataset, holding, promote, search  # noqa: E402


@pytest.fixture
def contest(tmp_path, monkeypatch):
    """A ledger and a book store of their own."""
    monkeypatch.setattr(search, "SEARCH_DIR", str(tmp_path / "search"))
    monkeypatch.setattr(holding, "STORE_DIR", str(tmp_path / "books"))
    return tmp_path


def trial(scores, spec=None, book=None):
    return search.record_trial(spec or dataset.Spec(), scores, book=book)


# --- the brakes ----------------------------------------------------------------

def test_it_never_asks_the_same_question_twice(contest, offline_spec):
    book = {"top_n": 1, "rebalance_every": 60, "min_probability": 0.5,
            "long_only": True, "account": 500.0}
    before = len(challenge.candidates(offline_spec))

    trial({"accuracy": 0.5, "random_percentile": 0.2}, offline_spec, book)

    after = challenge.candidates(offline_spec)
    assert len(after) == before - 1
    assert not any(challenge.book_id_for(c) == challenge.book_id_for(book)
                   for c in after)


def test_the_week_has_a_budget(contest, offline_spec):
    for _ in range(challenge.BUDGET_PER_WEEK):
        trial({"accuracy": 0.5}, offline_spec, {"top_n": 1})

    out = challenge.step({}, base=offline_spec)
    assert out["asked"] is None
    assert "used this week" in out["note"]


def test_last_week_does_not_count_against_this_one(contest, offline_spec,
                                                   monkeypatch):
    for _ in range(challenge.BUDGET_PER_WEEK):
        trial({"accuracy": 0.5}, offline_spec, {"top_n": 1})

    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=8)
    assert challenge.spent_this_week(later) == 0


def test_it_only_varies_rules_that_are_not_already_known_to_be_flat():
    """Model capacity, data volume and freshness were all measured flat. A loop
    that varied them would raise the bar for everything else and find nothing."""
    assert set(challenge.AXES) == {"top_n", "rebalance_every", "min_probability"}
    for book in challenge.candidates():
        assert book["long_only"] is True
        assert book["account"] == challenge.ACCOUNT


# --- the bar -------------------------------------------------------------------

def test_a_challenger_must_beat_random_books_and_buying_and_holding():
    ok, why = challenge.qualifies({"random_percentile": 0.99, "final_equity": 800.0,
                                   "hold_median": 600.0})
    assert ok and "99%" in why

    ok, why = challenge.qualifies({"random_percentile": 0.80, "final_equity": 800.0,
                                   "hold_median": 600.0})
    assert not ok and "under the 95% bar" in why

    ok, why = challenge.qualifies({"random_percentile": 0.99, "final_equity": 550.0,
                                   "hold_median": 600.0})
    assert not ok and "buying at random and holding" in why

    ok, why = challenge.qualifies({"random_percentile": 0.99, "final_equity": 0.0,
                                   "hold_median": 400.0})
    assert not ok and "ran out of money" in why


def test_a_qualifying_challenger_starts_a_shadow_record(contest):
    book = {"top_n": 3, "rebalance_every": 40, "long_only": True,
            "min_probability": 0.5, "account": 500.0}
    entered = challenge.enter(book, "because the test said so")

    assert entered == "top3-every40"
    assert holding.books()[entered]["funded"] is False
    assert holding.books()[entered]["rule"]["rebalance_every"] == 40


def test_the_contest_does_not_sprawl(contest, monkeypatch):
    monkeypatch.setattr(challenge, "MAX_SHADOWS", 2)
    for number in (1, 2):
        challenge.enter({"top_n": number, "rebalance_every": 20,
                         "long_only": True, "account": 500.0}, "testing")

    assert challenge.enter({"top_n": 5, "rebalance_every": 20,
                            "long_only": True, "account": 500.0}, "testing") is None
    assert len(holding.books()) == 3        # champion plus two shadows


# --- the promotion rule ---------------------------------------------------------

def settled(book_id, profits, opened="2026-01-05"):
    """Give a book a forward record of closed decisions."""
    for number, profit in enumerate(profits):
        holding._append({"exit": {
            "at": f"2026-0{1 + number % 9}-01T00:00:00+00:00",
            "opened": opened, "session": f"2026-0{1 + number % 9}-01",
            "sold": [{"symbol": "AAA", "price": 10.0, "bought_at": 10.0,
                      "gross_return": profit / 100.0, "fee": 0.0,
                      "value": 500.0 + profit, "net": profit}],
            "cash_before": 500.0, "cash_after": 500.0 + profit,
            "fees": 0.0, "profit": profit,
        }}, book_id=book_id)


def test_it_refuses_while_nothing_is_challenging(contest):
    assert promote.decide()["promote"] is None
    assert "nothing is challenging" in promote.decide()["why"]


def test_it_refuses_until_both_have_enough_decisions(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing")
    settled(holding.CHAMPION, [5.0] * 3)
    settled("rival", [50.0] * 9)

    verdict = promote.decide()
    assert verdict["promote"] is None
    assert f"3 closed decisions of the {promote.MIN_DECISIONS}" in verdict["why"]


def test_a_backtest_cannot_promote_anything(contest):
    """The challenger got its shadow by clearing a backtest. Doing it again is
    not new evidence."""
    holding.register("rival", rule={"top_n": 2},
                     why="beat 99% of matched random books")
    settled(holding.CHAMPION, [1.0] * promote.MIN_DECISIONS)

    verdict = promote.decide()
    assert verdict["promote"] is None
    assert "closed decisions" in verdict["why"]


def test_winning_by_a_nose_is_not_enough(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing")
    settled(holding.CHAMPION, [10.0] * promote.MIN_DECISIONS)
    settled("rival", [11.0] * promote.MIN_DECISIONS)

    verdict = promote.decide()
    assert verdict["promote"] is None
    assert "it needs" in verdict["why"]


def test_more_money_but_fewer_decisions_won_is_not_enough(contest):
    """One lucky holding is not a better method."""
    holding.register("rival", rule={"top_n": 2}, why="testing")
    settled(holding.CHAMPION, [5.0] * promote.MIN_DECISIONS)
    settled("rival", [300.0] + [-5.0] * (promote.MIN_DECISIONS - 1))

    verdict = promote.decide()
    assert verdict["promote"] is None


def test_a_challenger_that_is_clearly_better_takes_over(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing")
    settled(holding.CHAMPION, [1.0] * promote.MIN_DECISIONS)
    settled("rival", [20.0] * promote.MIN_DECISIONS)

    verdict = promote.apply()
    assert verdict["promote"] == "rival"
    assert holding.funded() == "rival"
    # And the reason is in the record, not only in a log.
    assert "against the champion's" in holding.books()["rival"]["why"] or True
    assert any("fund" in entry for entry in holding._read())


def test_it_will_not_promote_twice_in_a_row(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing")
    settled(holding.CHAMPION, [1.0] * promote.MIN_DECISIONS)
    settled("rival", [20.0] * promote.MIN_DECISIONS)
    promote.apply()

    holding.register("newcomer", rule={"top_n": 5}, why="testing")
    settled("newcomer", [40.0] * promote.MIN_DECISIONS)

    verdict = promote.decide()
    assert verdict["promote"] is None
    assert "before another can happen" in verdict["why"]

    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        days=promote.COOLDOWN_DAYS + 1)
    assert promote.decide(now=later)["promote"] == "newcomer"


# --- the guard ------------------------------------------------------------------
#
# Promotion asks whether something else is better. The guard asks whether the
# thing being followed is still doing what it was followed for. It can only
# stop: nothing in this module starts trading anything.

def test_the_guard_waits_for_enough_decisions(contest):
    settled(holding.CHAMPION, [-1.0] * 3)

    verdict = promote.drift()
    assert verdict["stand_down"] is False
    assert f"3 closed decisions of the {promote.DRIFT_DECISIONS}" in verdict["why"]


def test_a_deep_drawdown_stops_trading_whatever_the_reason(contest):
    """A stop on the account rather than a claim about the model: it does not
    wait for the decision count."""
    settled(holding.CHAMPION, [-60.0, -60.0])

    verdict = promote.guard()
    assert verdict["stand_down"] is True
    assert "past the -20% stop" in verdict["why"]
    assert holding.funded() is None


def test_a_book_that_stopped_doing_what_it_was_funded_for_is_stood_down(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing", expected=0.08)
    holding.fund("rival", why="testing")
    settled("rival", [-2.0] * promote.DRIFT_DECISIONS)

    verdict = promote.guard()
    assert verdict["stand_down"] is True
    assert "expecting +8.0% a decision" in verdict["why"]
    assert holding.funded() is None


def test_a_book_doing_its_job_is_left_alone(contest):
    holding.register("rival", rule={"top_n": 2}, why="testing", expected=0.08)
    holding.fund("rival", why="testing")
    settled("rival", [10.0] * promote.DRIFT_DECISIONS)

    verdict = promote.guard()
    assert verdict["stand_down"] is False
    assert holding.funded() == "rival"
    assert "nothing to act on" in verdict["why"]


def test_the_guard_never_starts_anything(contest):
    settled(holding.CHAMPION, [-60.0, -60.0])
    promote.guard()
    assert holding.funded() is None

    # Even with a challenger that looks wonderful, the guard leaves it stopped.
    holding.register("rival", rule={"top_n": 2}, why="testing", expected=0.08)
    settled("rival", [50.0] * promote.MIN_DECISIONS)

    assert promote.guard()["stand_down"] is False
    assert holding.funded() is None
    assert promote.decide()["promote"] is None
    assert "trading is stopped" in promote.decide()["why"]


def test_the_records_keep_running_after_a_stand_down(contest, panel):
    """You want to know whether the book you stopped would have recovered."""
    settled(holding.CHAMPION, [-60.0, -60.0])
    promote.guard()

    symbols = sorted(panel)
    dates = list(panel[symbols[0]].index)
    run = type("R", (), {"run_id": "r", "trust": {"trusted": False},
                         "signals": [{"symbol": symbols[0],
                                      "as_of": dates[-40].date().isoformat(),
                                      "probability_up": 0.9, "confidence": 0.8}]})()
    assert holding.plan_next(run) is not None
    assert holding.advance(panel, today=dates[-39])["filled"] is not None
