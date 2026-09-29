"""The challenger loop: try one new rule at a time, and enter the good ones.

A champion that is never challenged never improves, and a challenger loop with
no brakes is worse than none at all -- run enough of them and one beats the
champion by luck every single night, while the ledger reports triumph. So this
is deliberately slow and deliberately narrow.

**Narrow, because the model is not where the gains are.** Measured on the
validation period: quadrupling the data moves accuracy 0.2 points, a logistic
regression matches the network, an eighteen-month-old model matches a refitted
one, and specialists lose in six markets of seven. Every axis here is therefore
a *decision rule* -- what is held, how many, for how long. That is where every
improvement of the past year actually came from.

**Slow, because looking is what costs.** Each candidate scored is another
question asked of the same rows, and the bar rises with the count (see
search.corrected_threshold). So the loop spends a small, fixed budget per week,
never re-tries a rule already in the ledger, and enters at most a few shadows
into the contest at once.

A challenger that clears the bar does not take over. It is registered as a
shadow book in holding.py and starts earning a forward record of its own; only
that record can promote it, and only through promote.py.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging

from . import book as book_mod
from . import dataset as dataset_mod
from . import holding as holding_mod
from . import search as search_mod

logger = logging.getLogger(__name__)

# The rules worth varying, and nothing else. Model capacity, training length and
# data volume are all measured flat -- a loop that varied them would burn the
# bar for nothing. Short books are excluded because every long/short variant
# tested has come back negative.
AXES = {
    "top_n": [1, 2, 3, 5],
    "rebalance_every": [20, 40, 60, 120],
    "min_probability": [0.5, 0.55],
}

# How many new questions a week. Five is not a performance limit -- a candidate
# takes under a minute -- it is a statistical one: at 32 trials the bar is
# already 1.61x what it was on the first look, and every trial raises it for
# everything that follows.
BUDGET_PER_WEEK = 5

# What a challenger must clear before it is allowed to start a forward record:
# beat this share of matched random books with the same trade pattern, end
# above what buying at random and holding would have, and not go broke.
MIN_PERCENTILE = 0.95

# How many shadows may be running at once. Each is a real forward record that
# somebody has to read; a dozen of them is a leaderboard nobody trusts.
MAX_SHADOWS = 3

ACCOUNT = book_mod.DEFAULT_ACCOUNT


def _key(spec: dict, book: dict) -> tuple:
    """What makes two trials the same question."""
    return (
        spec.get("target"), spec.get("horizon"), spec.get("use_macro"),
        spec.get("use_cross"), spec.get("use_events"), spec.get("use_news"),
        spec.get("neutral_band"),
        book.get("top_n"), book.get("rebalance_every"),
        book.get("min_probability"), bool(book.get("long_only")),
        bool(book.get("account")),
    )


def asked_already() -> set:
    """Every question the ledger has already paid for."""
    return {_key(trial.get("spec") or {}, trial.get("book") or {})
            for trial in search_mod.read_trials()}


def spent_this_week(now: dt.datetime | None = None) -> int:
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(days=7)
    spent = 0
    for trial in search_mod.read_trials():
        try:
            at = dt.datetime.fromisoformat(trial["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if at >= since:
            spent += 1
    return spent


def candidates(base: dataset_mod.Spec | None = None) -> list:
    """Every rule in the grid that has not been asked about yet."""
    base = base or dataset_mod.Spec()
    spec = base.to_dict()
    asked = asked_already()

    out = []
    for values in itertools.product(*AXES.values()):
        book = dict(zip(AXES, values))
        book.update({"long_only": True, "account": ACCOUNT})
        if _key(spec, book) not in asked:
            out.append(book)
    return out


def describe(book: dict) -> str:
    return (f"top {book['top_n']}, every {book['rebalance_every']}"
            + (f", p>={book['min_probability']:g}"
               if (book.get("min_probability") or 0.5) > 0.5 else ""))


def book_id_for(book: dict) -> str:
    name = f"top{book['top_n']}-every{book['rebalance_every']}"
    if (book.get("min_probability") or 0.5) > 0.5:
        name += f"-p{int(float(book['min_probability']) * 100)}"
    return name


def qualifies(scores: dict) -> tuple:
    """Did this challenger earn the right to start a forward record?"""
    percentile = scores.get("random_percentile")
    if percentile is None:
        return False, "it was not scored in money, so there is no percentile"
    if scores.get("final_equity", 0) <= 0:
        return False, "it ran out of money"
    if percentile < MIN_PERCENTILE:
        return False, (f"it beat {percentile:.0%} of matched random books, "
                       f"under the {MIN_PERCENTILE:.0%} bar")
    if scores.get("final_equity", 0) <= (scores.get("hold_median") or 0):
        return False, (f"it ended at {scores['final_equity']:.2f}, under the "
                       f"{scores['hold_median']:.2f} that buying at random and "
                       f"holding would have made")
    return True, (f"beat {percentile:.0%} of matched random books and ended at "
                  f"{scores['final_equity']:.2f} against a "
                  f"{scores['hold_median']:.2f} buy-and-hold median")


def step(frames: dict, *, base: dataset_mod.Spec | None = None,
         budget: int | None = None, now: dt.datetime | None = None) -> dict:
    """Ask one new question, if the week has room for it.

    One per call on purpose: a scheduler cycle should stay short, and a loop
    that spends its whole budget in one evening is a loop that has stopped
    thinking about what it is asking.
    """
    budget = BUDGET_PER_WEEK if budget is None else budget
    spent = spent_this_week(now)
    if spent >= budget:
        return {"asked": None, "note": f"{spent} of {budget} trials used this week"}

    waiting = candidates(base)
    if not waiting:
        return {"asked": None, "note": "every rule in the grid has been tried"}

    book = waiting[0]
    out = search_mod.run_search(
        frames, [{}], base=base or dataset_mod.Spec(), books=[book],
        note=f"challenger: {describe(book)}, entered by the loop")

    best = (out.get("best") or [{}])[0]
    scores = best.get("validation") or {}
    ok, why = qualifies(scores)

    result = {"asked": describe(book), "book": book, "trial": best.get("trial"),
              "qualified": ok, "why": why,
              "percentile": scores.get("random_percentile"),
              "final_equity": scores.get("final_equity"),
              "entered": None}

    if ok:
        rebalances = max(int(scores.get("rebalances") or 1), 1)
        expected = ((scores["final_equity"] / ACCOUNT) - 1.0) / rebalances
        result["expected"] = round(expected, 6)
        result["entered"] = enter(book, why, expected=expected)
    logger.info("Challenger %s: %s", describe(book), why)
    return result


def enter(book: dict, why: str, *, expected: float | None = None) -> str | None:
    """Register a qualifying challenger as a shadow, if there is room."""
    contest = holding_mod.books()
    shadows = [b for b in contest.values() if not b["funded"]]
    if len(shadows) >= MAX_SHADOWS:
        logger.info("Not entering: %d shadows already running", len(shadows))
        return None

    book_id = book_id_for(book)
    if book_id in contest:
        return None

    holding_mod.register(book_id, rule={
        "top_n": book["top_n"], "rebalance_every": book["rebalance_every"],
        "long_only": True}, why=why, expected=expected)
    return book_id
