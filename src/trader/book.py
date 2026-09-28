"""A book simulated in money, because a minimum fee is not a percentage.

Everything else in this project scores a portfolio in fractions: a position is a
weight, a cost is basis points, and the account size never appears. That works
until the broker charges a minimum per trade, at which point the same strategy
is free at one account size and fatal at another. Measured on the real panel: a
500 account holding the model's top five, rebalanced every session, pays 504 of
commission over the test period and **ends at zero**. The same five names
rebalanced every sixty sessions pay 92 and end at 792.

So a book here is simulated forward in money: positions are dollars, trades are
charged what the broker charges, and an account that runs out of money stops.

The second thing this module exists for is the comparison. A book of five names
rebalanced eight times is fourteen decisions, and any single run of it is
unreadable on its own -- concentrated books have enormous spread whatever picks
them. `percentile` runs the identical trade pattern with names drawn at random,
hundreds of times, and reports where the real book lands. Fifty per cent is no
skill. That is a permutation test, and it is the only honest way to read a
number produced by this few decisions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import broker as broker_mod
from . import evaluate as evaluate_mod

logger = logging.getLogger(__name__)

# What the page and the search assume unless told otherwise. Small on purpose:
# it is the size at which the minimum fee decides everything, which is the fact
# worth designing around rather than discovering later.
DEFAULT_ACCOUNT = 500.0

# Charged on the value traded, each way, on top of commission. Spread and
# slippage are real whatever the broker charges, and at Avanza Start -- where
# commission is nothing -- this is the entire cost.
DEFAULT_SPREAD = evaluate_mod.DEFAULT_COST

# How many random books to compare against. Enough that a 95th percentile means
# something and the whole search still finishes in a couple of minutes.
DRAWS = 200


def targets(signal: pd.DataFrame, sessions, *, top: int, long_only: bool = True,
            every: int = 1, lag: int = 1) -> list:
    """The intended book for each session, from a signal `lag` sessions old.

    Between rebalances the book is simply held: the point of a longer interval
    is that nothing is traded, so nothing is charged. The lag is what keeps it
    executable -- the signal comes from a close that had already happened.
    """
    by_date = {date: group for date, group in signal.groupby("date")}
    held: dict = {}
    out = []

    for index, _ in enumerate(sessions):
        if index % max(every, 1) == 0:
            held = {}
            if index - lag >= 0:
                day = by_date.get(sessions[index - lag])
                if day is not None and len(day):
                    if long_only:
                        best = day.nlargest(top, "p")
                        if float(best["p"].max()) > 0.5:
                            held = {row.symbol: 1.0 / top
                                    for row in best.itertuples()}
                    else:
                        ranked = day.assign(
                            edge=(day["p"] - 0.5).abs()).nlargest(top, "edge")
                        held = {row.symbol: (1.0 if row.p > 0.5 else -1.0) / top
                                for row in ranked.itertuples()}
        out.append(dict(held))
    return out


def simulate(books: list, moves: pd.DataFrame, *, account: float = DEFAULT_ACCOUNT,
             schedule=None, spread: float = DEFAULT_SPREAD) -> dict:
    """Walk the account forward one session at a time.

    `books[i]` is what should be held on session i; the difference from what is
    already held is what gets traded, so carrying a position costs nothing.
    Returns the daily net return series as well as the money, because the rest
    of the project scores series rather than balances.
    """
    columns = {name: index for index, name in enumerate(moves.columns)}
    rows = moves.to_numpy(dtype=np.float64)

    equity, holding, trades, fees = float(account), {}, 0, 0.0
    returns, curve = [], []

    for index in range(len(books)):
        opening = equity
        if index and holding:
            equity *= 1.0 + sum(
                weight * rows[index][columns[name]]
                for name, weight in holding.items() if name in columns)

        target = books[index]
        charge = 0.0
        for name in set(target) | set(holding):
            moved = abs(target.get(name, 0.0) - holding.get(name, 0.0))
            if moved < 1e-9:
                continue
            value = moved * equity
            charge += broker_mod.per_side(value, schedule) + value * spread
            trades += 1
        equity -= charge
        fees += charge
        holding = {n: w for n, w in target.items() if abs(w) > 1e-9}

        returns.append(equity / opening - 1.0 if opening > 0 else 0.0)
        curve.append(equity)

        if equity <= 0:
            logger.info("The account ran out of money on session %d", index)
            return {"final": 0.0, "trades": trades, "fees": fees,
                    "returns": pd.Series(returns), "equity": curve,
                    "broke": True, "account": float(account)}

    if holding:                                  # close out, so it ends in cash
        value = sum(abs(w) for w in holding.values()) * equity
        charge = broker_mod.per_side(value, schedule) + value * spread
        equity -= charge
        fees += charge
        trades += len(holding)

    return {"final": equity, "trades": trades, "fees": fees,
            "returns": pd.Series(returns), "equity": curve, "broke": False,
            "account": float(account)}


def percentile(final: float, moves: pd.DataFrame, *, top: int, every: int,
               account: float = DEFAULT_ACCOUNT, schedule=None,
               spread: float = DEFAULT_SPREAD, draws: int = DRAWS,
               seed: int = 0) -> dict:
    """Where this book lands against the same trade pattern, picked at random.

    The comparison holds everything constant except the choice of names: the
    same number of positions, the same rebalance dates, the same commission and
    spread on the same account. What is left is whether the picking was worth
    anything.
    """
    names = list(moves.columns)
    rng = np.random.default_rng(seed)
    finals = []

    for _ in range(draws):
        books, held = [], {}
        for index in range(len(moves)):
            if index % max(every, 1) == 0:
                picked = rng.choice(names, size=min(top, len(names)),
                                    replace=False)
                held = {name: 1.0 / top for name in picked}
            books.append(dict(held))
        finals.append(simulate(books, moves, account=account,
                               schedule=schedule, spread=spread)["final"])

    finals = np.array(finals, dtype=float)
    return {
        "beaten": float((finals < final).mean()),
        "median": round(float(np.median(finals)), 2),
        "worst_tenth": round(float(np.percentile(finals, 10)), 2),
        "best_tenth": round(float(np.percentile(finals, 90)), 2),
        "draws": int(draws),
    }


def buy_and_hold(moves: pd.DataFrame, *, count: int = 5,
                 account: float = DEFAULT_ACCOUNT, schedule=None,
                 spread: float = DEFAULT_SPREAD, draws: int = DRAWS,
                 seed: int = 0) -> dict:
    """The bar: buy `count` names at random and do nothing until the end.

    Two trades and no decisions. Every active strategy has to beat this after
    its own turnover, and on this panel most of them do not come close.
    """
    return percentile(float("inf"), moves, top=count, every=len(moves) + 10,
                      account=account, schedule=schedule, spread=spread,
                      draws=draws, seed=seed)
