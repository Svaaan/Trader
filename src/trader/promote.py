"""When a challenger takes over, and the rule that decides it.

Written before any contest has run, which is the only time such a rule can be
written honestly. Once there are numbers on the table, every threshold becomes
an argument about a particular challenger.

The rule refuses far more often than it passes, and the reasons matter:

**Backtests cannot promote.** A challenger has already cleared a backtest to be
here -- that is what got it a shadow book. Promoting on the same evidence would
be promoting on the thing that has fooled this project three times. Only the
forward record, written before its own outcomes existed, can move the money.

**Few decisions, so no t-test theatre.** A book reviewed every sixty sessions
makes four decisions a year. Eight of them is two years, and eight paired
outcomes will not support a significance test worth the name. So the rule asks
for agreement rather than significance: more decisions won, *and* a margin on
the money, *and* no promotion in the recent past.

**The champion keeps the tie.** Incumbency is worth something here: switching
costs a round trip, and a challenger that merely matches has shown nothing.
"""

from __future__ import annotations

import datetime as dt
import logging

from . import holding as holding_mod

logger = logging.getLogger(__name__)

# How many closed positions each book needs before the comparison means
# anything. At a sixty-session review that is about two years -- which is the
# honest price of evidence that cannot be mined, not a target to be shortened.
MIN_DECISIONS = 8

# How far ahead the challenger has to be, in total return over its own record.
# Below this the difference is one lucky holding.
MARGIN = 0.05

# And how long after a promotion before another can happen. A book that has
# just taken over has no record of its own to judge.
COOLDOWN_DAYS = 180

# --- the guard ---------------------------------------------------------------
#
# Promotion asks whether something else is better. The guard asks a different
# question: is the thing being followed still doing what it was followed for?
# It can only ever stop -- nothing in this module starts trading anything.

# How many closed decisions before the guard is allowed an opinion. Fewer than
# this and it would be reacting to one bad quarter.
DRIFT_DECISIONS = 6

# A stop on the account rather than a claim about the model: at this much down
# the question is no longer whether the method is sound.
DRAWDOWN_LIMIT = -0.20


def _closed(book_id: str) -> list:
    return holding_mod.state(book_id)["closed"]


def _record(book_id: str) -> dict:
    """What one book has actually done, from its own closed positions."""
    closed = _closed(book_id)
    profit = sum(entry.get("profit", 0.0) for entry in closed)
    wins = sum(1 for entry in closed if entry.get("profit", 0.0) > 0)
    start = holding_mod.STARTING_CASH
    return {
        "book_id": book_id,
        "decisions": len(closed),
        "profit": round(profit, 2),
        "return": round(profit / start, 6) if start else 0.0,
        "wins": wins,
        "since": closed[0]["opened"] if closed else None,
    }


def _last_promotion() -> dt.datetime | None:
    latest = None
    for entry in holding_mod._read():
        if "fund" in entry:
            try:
                at = dt.datetime.fromisoformat(entry["fund"]["at"])
            except (KeyError, TypeError, ValueError):
                continue
            latest = at if latest is None or at > latest else latest
    return latest


def drift(*, now: dt.datetime | None = None) -> dict:
    """Is the funded book still doing what it was funded for?

    Two conditions, both deliberately crude, because four decisions a year will
    not support anything subtle. The first is a drawdown stop. The second is
    the honest version of "it stopped working": the book was funded on a
    backtest that expected to make money, and over enough real decisions it has
    lost money instead. Neither is a significance test and neither pretends to
    be one.
    """
    funded_id = holding_mod.funded()
    if funded_id is None:
        return {"stand_down": False, "funded": None,
                "why": "nothing is funded; trading is already stopped"}

    record = _record(funded_id)
    expected = (holding_mod.books().get(funded_id) or {}).get("expected")
    out = {"stand_down": False, "funded": funded_id, "record": record,
           "expected": expected, "why": ""}

    if record["return"] <= DRAWDOWN_LIMIT:
        out.update(stand_down=True, why=(
            f"{funded_id} is {record['return']:+.1%} over {record['decisions']} "
            f"decisions, past the {DRAWDOWN_LIMIT:+.0%} stop"))
        return out

    if record["decisions"] < DRIFT_DECISIONS:
        out["why"] = (f"{record['decisions']} closed decisions of the "
                      f"{DRIFT_DECISIONS} the guard needs")
        return out

    if expected is not None and expected > 0 and record["return"] < 0:
        out.update(stand_down=True, why=(
            f"{funded_id} was funded on a backtest expecting "
            f"{expected:+.1%} a decision and has returned "
            f"{record['return']:+.1%} over {record['decisions']} real ones"))
        return out

    out["why"] = (f"{funded_id} is {record['return']:+.1%} over "
                  f"{record['decisions']} decisions; nothing to act on")
    return out


def guard(*, now: dt.datetime | None = None) -> dict:
    """Act on the drift rule. Stops trading; never starts it."""
    verdict = drift(now=now)
    if verdict["stand_down"]:
        holding_mod.stand_down(why=verdict["why"])
    return verdict


def decide(*, now: dt.datetime | None = None) -> dict:
    """Should anything take over? Almost always no, and it says why.

    Returns the verdict, the two records it compared, and what is still
    missing -- so that "not yet" is a sentence with a number in it rather than
    a shrug.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    champion_id = holding_mod.funded()
    if champion_id is None:
        return {"champion": None, "challengers": [], "promote": None,
                "why": "trading is stopped; a challenger cannot take over "
                       "something nobody is following"}
    champion = _record(champion_id)

    challengers = [_record(book_id) for book_id in holding_mod.books()
                   if book_id != champion_id]
    verdict = {"champion": champion, "challengers": challengers,
               "promote": None, "why": ""}

    if not challengers:
        verdict["why"] = "nothing is challenging it"
        return verdict

    since = _last_promotion()
    if since is not None and (now - since).days < COOLDOWN_DAYS:
        waiting = COOLDOWN_DAYS - (now - since).days
        verdict["why"] = (f"the last promotion was {(now - since).days} days ago; "
                          f"{waiting} more before another can happen")
        return verdict

    if champion["decisions"] < MIN_DECISIONS:
        verdict["why"] = (f"the champion has {champion['decisions']} closed "
                          f"decisions of the {MIN_DECISIONS} needed to compare")
        return verdict

    ready = [c for c in challengers if c["decisions"] >= MIN_DECISIONS]
    if not ready:
        best = max(challengers, key=lambda c: c["decisions"])
        verdict["why"] = (f"the best-placed challenger has {best['decisions']} "
                          f"closed decisions of the {MIN_DECISIONS} needed")
        return verdict

    # Ahead on the money by a margin, and ahead on the count of decisions won.
    # Either alone is one lucky holding or a long run of small wins that lost
    # money overall.
    # Not *worse* on the share of decisions won, rather than strictly better:
    # a champion that has won all of them cannot be beaten on that count, and
    # refusing on it would make the money margin unreachable. What this still
    # excludes is the case it was written for -- one enormous holding carrying
    # a book that lost most of its decisions.
    beaten = [c for c in ready
              if c["return"] >= champion["return"] + MARGIN
              and c["wins"] / max(c["decisions"], 1)
              >= champion["wins"] / max(champion["decisions"], 1)]

    if not beaten:
        closest = max(ready, key=lambda c: c["return"])
        verdict["why"] = (
            f"{closest['book_id']} is the closest at {closest['return']:+.1%} "
            f"against the champion's {champion['return']:+.1%}; it needs "
            f"{champion['return'] + MARGIN:+.1%} and a better share of decisions won")
        return verdict

    winner = max(beaten, key=lambda c: c["return"])
    verdict["promote"] = winner["book_id"]
    verdict["why"] = (
        f"{winner['book_id']} returned {winner['return']:+.1%} over "
        f"{winner['decisions']} closed decisions against the champion's "
        f"{champion['return']:+.1%} over {champion['decisions']}, winning "
        f"{winner['wins']}/{winner['decisions']} against "
        f"{champion['wins']}/{champion['decisions']}")
    return verdict


def apply(*, now: dt.datetime | None = None) -> dict:
    """Act on the rule. Funds the winner when there is one, and says so loudly.

    Nothing here is reversible by deleting a file: the promotion is a line in
    the same append-only record as everything else, with the numbers that
    justified it.
    """
    verdict = decide(now=now)
    if verdict["promote"]:
        holding_mod.fund(verdict["promote"], why=verdict["why"])
        logger.warning("Promoted %s: %s", verdict["promote"], verdict["why"])
    return verdict
