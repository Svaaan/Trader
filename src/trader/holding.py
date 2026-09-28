"""The committed book, held forward: what it bought, and when it will sell.

The paper ledger next door records a fresh book every session and closes it at
that session's close. That shape is dead at this account size -- 238 names on
500 pays 478 of commission on day one and there is no day two -- and the search
concluded with a different one: hold a very small book and leave it alone.
Trial 28, long only, top one, rebalanced every sixty sessions: twelve trades
over a year and a half, 51 of fees, ahead of 96.2% of the same trade pattern
picked at random.

So this is a different kind of record. It has one position at a time, it knows
when it intends to sell, and every step is a line appended to a file:

    plan   what it wants to buy, written before the price exists
    fill   what it actually paid, at the first open after the plan
    exit   what it sold for, on the session the plan said to review

The plan is written from a close that has already happened and filled at the
next open, which is the same discipline the rest of the project grades itself
on. Nothing here is ever rewritten, and the model never reads any of it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os

import pandas as pd

from . import broker as broker_mod

logger = logging.getLogger(__name__)

STORE_DIR = os.environ.get(
    "TRADER_HOLDING",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "paper"),
)
STORE_FILE = "holdings.jsonl"

STARTING_CASH = 500.0

# What the search committed. Kept here as the default rather than read from the
# search ledger, because a forward record should say what it was following even
# if somebody commits something else tomorrow.
BOOK = {"top_n": 1, "rebalance_every": 60, "long_only": True}

# Which price list a name is billed on. The first position this book ever
# planned was INVE-B.ST, and billing a Stockholm share at the US schedule would
# have charged 2.50 a side plus currency instead of 1.25 and none -- the fee is
# a property of the market, not of the strategy.
SCHEDULES = {"ST": "avanza_mini_se", "CO": "avanza_mini_se",
             "OL": "avanza_mini_se", "HE": "avanza_mini_de",
             "DE": "avanza_mini_de", "PA": "avanza_mini_de",
             "AS": "avanza_mini_de", "MI": "avanza_mini_de",
             "MC": "avanza_mini_de", "BR": "avanza_mini_de"}
SCHEDULE = "avanza_mini_us"
SPREAD = 0.0005


def schedule_for(symbol: str) -> str:
    """The price list this symbol is billed on, from where it is listed."""
    suffix = symbol.rsplit(".", 1)[-1] if "." in symbol else ""
    return SCHEDULES.get(suffix, SCHEDULE)


def _path() -> str:
    return os.path.join(os.path.abspath(STORE_DIR), STORE_FILE)


def _read() -> list:
    try:
        with open(_path(), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return []


def _append(entry: dict) -> dict:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def state() -> dict:
    """Where the account stands, read from the lines in order.

    Returns the plan waiting to be filled (if any), the position being held (if
    any), everything already closed, and the cash.
    """
    cash, plan, holding, closed = STARTING_CASH, None, None, []

    for entry in _read():
        if "plan" in entry:
            plan = entry["plan"]
        elif "fill" in entry:
            holding = entry["fill"]
            cash = entry["fill"]["cash_after"]
            plan = None
        elif "exit" in entry:
            closed.append(entry["exit"])
            cash = entry["exit"]["cash_after"]
            holding = None

    return {"cash": cash, "plan": plan, "holding": holding, "closed": closed}


def _sessions_between(frames: dict, symbol: str, start, end) -> int:
    frame = frames.get(symbol)
    if frame is None:
        return 0
    index = frame.index
    return int(((index > pd.Timestamp(start)) & (index <= pd.Timestamp(end))).sum())


def _next_session(frames: dict, symbol: str, after):
    frame = frames.get(symbol)
    if frame is None:
        return None
    later = frame.index[frame.index > pd.Timestamp(after)]
    return later[0] if len(later) else None


def symbols_to_price() -> set:
    """Whatever the record needs prices for: the plan waiting, or the holding.

    The same rule the other ledger learned the hard way -- what to fetch is
    named by the record, not by whichever universe the caller happens to have.
    """
    current = state()
    names = set()
    if current["plan"]:
        names.update(p["symbol"] for p in current["plan"]["buy"])
    if current["holding"]:
        names.update(p["symbol"] for p in current["holding"]["bought"])
    return names


def plan_next(run, *, book: dict | None = None) -> dict | None:
    """Write down what to buy, before the price it will be bought at exists.

    Refuses if something is already held or already planned: this book holds one
    thing at a time, and the point of the interval is that it is left alone.
    """
    book = {**BOOK, **(book or {})}
    current = state()
    if current["holding"] or current["plan"]:
        return None

    signals = [s for s in (run.signals or []) if s.get("probability_up") is not None]
    if not signals:
        logger.info("No signals to plan from")
        return None

    if book.get("long_only", True):
        ranked = sorted(signals, key=lambda s: -float(s["probability_up"]))
        chosen = [s for s in ranked[:int(book["top_n"])]
                  if float(s["probability_up"]) > 0.5]
    else:
        ranked = sorted(signals, key=lambda s: -abs(float(s["probability_up"]) - 0.5))
        chosen = ranked[:int(book["top_n"])]

    if not chosen:
        logger.info("Nothing the model calls up; staying in cash")
        return None

    as_of = max(s["as_of"] for s in chosen)
    entry = {"plan": {
        "at": _now(),
        "as_of": as_of,
        "run_id": getattr(run, "run_id", None),
        "book": book,
        "trusted": bool((getattr(run, "trust", None) or {}).get("trusted")),
        "why": (getattr(run, "trust", None) or {}).get("reason", ""),
        "buy": [{"symbol": s["symbol"],
                 "weight": round(1.0 / len(chosen), 6),
                 "probability_up": s["probability_up"],
                 "confidence": s.get("confidence")} for s in chosen],
    }}
    logger.info("Plan: buy %s at the next open (from %s)",
                ", ".join(s["symbol"] for s in chosen), as_of)
    return _append(entry)["plan"]


def advance(frames: dict, *, book: dict | None = None, today=None) -> dict:
    """Fill a waiting plan, or sell a position whose review date has arrived.

    Called on a schedule. Does nothing most days, which is the entire point of
    the strategy it is recording.
    """
    book = {**BOOK, **(book or {})}
    current = state()
    out = {"filled": None, "exited": None, "holding": None, "note": None}

    if current["plan"] and not current["holding"]:
        filled = _fill(current["plan"], current["cash"], frames)
        if filled:
            out["filled"] = filled
            current = state()
        else:
            out["note"] = "the session after the plan has not closed yet"

    holding = current["holding"]
    if holding:
        due = _sessions_left(holding, frames, today=today)
        out["holding"] = {**holding, "sessions_left": due["left"],
                          "review_on": due["review_on"],
                          "held_sessions": due["held"]}
        if due["left"] <= 0:
            exited = _exit(holding, current["cash"], frames)
            if exited:
                out["exited"] = exited
                out["holding"] = None
                out["note"] = "sold: the review session arrived"
        else:
            out["note"] = (f"holding {', '.join(p['symbol'] for p in holding['bought'])}"
                           f", {due['left']} session(s) until it decides again")

    return out


def _fill(plan: dict, cash: float, frames: dict) -> dict | None:
    """Buy at the first open after the plan was written."""
    bought, spent = [], 0.0
    for position in plan["buy"]:
        symbol = position["symbol"]
        session = _next_session(frames, symbol, plan["as_of"])
        if session is None:
            return None
        row = frames[symbol].loc[session]
        value = cash * position["weight"]
        fee = broker_mod.per_side(value, schedule_for(symbol)) + value * SPREAD
        bought.append({
            "symbol": symbol, "session": session.date().isoformat(),
            "price": round(float(row["open"]), 6),
            "value": round(value, 4), "fee": round(fee, 4),
            "shares": round((value - fee) / float(row["open"]), 8),
        })
        spent += fee

    entry = {"fill": {
        "at": _now(), "as_of": plan["as_of"], "run_id": plan.get("run_id"),
        "book": plan["book"], "trusted": plan.get("trusted"),
        "bought": bought,
        "session": bought[0]["session"],
        "cash_before": round(cash, 4),
        "cash_after": round(cash - spent, 4),
        "fees": round(spent, 4),
    }}
    logger.info("Filled: %s at %s", ", ".join(
        f"{b['symbol']} {b['price']}" for b in bought), bought[0]["session"])
    return _append(entry)["fill"]


def _sessions_left(holding: dict, frames: dict, today=None) -> dict:
    """How many sessions until this book is reviewed, and on what date."""
    book = {**BOOK, **(holding.get("book") or {})}
    every = int(book.get("rebalance_every") or 1)
    symbol = holding["bought"][0]["symbol"]
    entered = holding["session"]

    frame = frames.get(symbol)
    if frame is None:
        return {"left": every, "review_on": None, "held": 0}

    today = pd.Timestamp(today) if today is not None else frame.index[-1]
    held = _sessions_between(frames, symbol, entered, today)
    later = frame.index[frame.index > pd.Timestamp(entered)]
    review = later[every - 1] if len(later) >= every else None

    return {"left": max(every - held, 0),
            "review_on": review.date().isoformat() if review is not None else None,
            "held": held}


def _exit(holding: dict, cash: float, frames: dict) -> dict | None:
    """Sell at the open of the session the plan named."""
    sold, proceeds = [], 0.0
    for position in holding["bought"]:
        symbol = position["symbol"]
        due = _sessions_left(holding, frames)
        if due["review_on"] is None:
            return None
        session = pd.Timestamp(due["review_on"])
        frame = frames.get(symbol)
        if frame is None or session not in frame.index:
            return None

        price = float(frame.loc[session]["open"])
        value = position["shares"] * price
        fee = broker_mod.per_side(value, schedule_for(symbol)) + value * SPREAD
        sold.append({
            "symbol": symbol, "session": session.date().isoformat(),
            "price": round(price, 6), "value": round(value, 4),
            "fee": round(fee, 4),
            "bought_at": position["price"],
            "gross_return": round(price / position["price"] - 1.0, 6),
            "net": round(value - fee - position["value"], 4),
        })
        proceeds += value - fee

    entry = {"exit": {
        "at": _now(), "opened": holding["session"],
        "session": sold[0]["session"], "sold": sold,
        "cash_before": round(cash, 4),
        "cash_after": round(cash + proceeds, 4),
        "fees": round(sum(s["fee"] for s in sold), 4),
        "profit": round(sum(s["net"] for s in sold), 4),
    }}
    logger.info("Sold: %s at %s, profit %.2f", ", ".join(
        s["symbol"] for s in sold), sold[0]["session"], entry["exit"]["profit"])
    return _append(entry)["exit"]


def account(frames: dict | None = None, *, today=None) -> dict:
    """What the page shows: the pick, the plan to sell, and what it has done."""
    current = state()
    holding, plan, closed = current["holding"], current["plan"], current["closed"]

    position = None
    if holding and frames:
        due = _sessions_left(holding, frames, today=today)
        marks = []
        for bought in holding["bought"]:
            frame = frames.get(bought["symbol"])
            last = float(frame["close"].iloc[-1]) if frame is not None and len(frame) else None
            marks.append({
                **bought,
                "last": round(last, 6) if last is not None else None,
                "move": (round(last / bought["price"] - 1.0, 6)
                         if last is not None else None),
                "worth": round(bought["shares"] * last, 4) if last is not None else None,
            })
        position = {
            "since": holding["session"], "book": holding["book"],
            "trusted": holding.get("trusted"), "marks": marks,
            "held_sessions": due["held"], "sessions_left": due["left"],
            "review_on": due["review_on"],
            "worth": round(sum(m["worth"] or 0.0 for m in marks), 2),
        }

    equity = current["cash"] + (position["worth"] if position else 0.0)
    realised = sum(e["profit"] for e in closed)

    return {
        "starting_cash": STARTING_CASH,
        "cash": round(current["cash"], 2),
        "equity": round(equity, 2),
        "profit": round(equity - STARTING_CASH, 2),
        "return": round(equity / STARTING_CASH - 1.0, 6),
        "realised": round(realised, 2),
        "fees": round(sum(e.get("fees", 0.0) for e in closed)
                      + (holding.get("fees", 0.0) if holding else 0.0), 2),
        "trades": 2 * len(closed) + (len(holding["bought"]) if holding else 0),
        "book": BOOK,
        "waiting_to_buy": plan,
        "position": position,
        "closed": closed[-10:],
        "schedule": ", ".join(sorted({
            broker_mod.resolve(schedule_for(p["symbol"])).name
            for p in ((holding or {}).get("bought")
                      or (plan or {}).get("buy") or [])
        })) or broker_mod.resolve(SCHEDULE).name,
    }
