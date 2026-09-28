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

# How many of the reasons behind a buy are remembered as "the pattern". The
# model gives a contribution per feature; these are the ones that actually
# pushed the answer up, biggest first.
PATTERN_SIZE = 5

# When the pattern is called broken. Both are definitions rather than tuned
# numbers: below half, the model no longer says up at all, and below half the
# drivers agreeing, most of what it bought on has gone. Neither has been tested
# as a reason to trade -- see `watch`.
CALLS_IT_UP = 0.5
AGREEMENT_BROKEN = 0.5


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
    cash, plan, holding, closed, watched = STARTING_CASH, None, None, [], None

    for entry in _read():
        if "plan" in entry:
            plan = entry["plan"]
        elif "fill" in entry:
            holding = entry["fill"]
            cash = entry["fill"]["cash_after"]
            plan = None
        elif "watch" in entry:
            watched = entry["watch"]
        elif "renew" in entry:
            # A review that confirmed the same name. Nothing traded, nothing
            # paid; the only thing that moves is when it next decides.
            if holding:
                holding = {**holding, "session": entry["renew"]["from_session"],
                           "renewed": entry["renew"]["at"]}
        elif "exit" in entry:
            closed.append(entry["exit"])
            cash = entry["exit"]["cash_after"]
            holding = None
            watched = None

    return {"cash": cash, "plan": plan, "holding": holding, "closed": closed,
            "watch": watched}


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


def _pattern_of(signal: dict) -> list:
    """The reasons that pushed this name up, biggest first.

    Only the ones arguing for the position. A feature that pushed *against* it
    and lost is not part of the case for buying, so it is not part of the case
    breaking either.
    """
    contributions = signal.get("all_contributions") or signal.get("reasons") or []
    pushing = [c for c in contributions if float(c.get("effect") or 0.0) > 0]
    pushing.sort(key=lambda c: -float(c["effect"]))
    return [{"feature": c["feature"],
             "effect": round(float(c["effect"]), 6),
             "z": round(float(c.get("z") or 0.0), 3)}
            for c in pushing[:PATTERN_SIZE]]


def _compare(pattern: list, signal: dict, bought_at: float) -> dict:
    """Is the case it bought on still the case?

    Not a trading rule and not a prediction: it re-reads today's reasons for the
    name already held and says how much of the original argument survives. A
    warning costs nothing, and at this account size a change of mind costs a
    round trip -- so the two are kept apart on purpose.
    """
    now = float(signal.get("probability_up") or 0.0)
    today = {c["feature"]: float(c.get("effect") or 0.0)
             for c in (signal.get("all_contributions") or [])}

    kept, lost = [], []
    for driver in pattern or []:
        still = today.get(driver["feature"], 0.0)
        (kept if still > 0 else lost).append({
            "feature": driver["feature"],
            "was": driver["effect"], "now": round(still, 6)})

    counted = len(kept) + len(lost)
    agreement = (len(kept) / counted) if counted else 1.0

    if now < CALLS_IT_UP:
        status = "broken"
        note = (f"the model no longer calls it up ({now:.3f} against "
                f"{bought_at:.3f} when it bought)")
    elif agreement < AGREEMENT_BROKEN:
        status = "broken"
        note = (f"{len(lost)} of {counted} reasons it bought on have gone "
                f"({', '.join(l['feature'] for l in lost[:3])})")
    elif agreement < 1.0 or now < bought_at:
        status = "drifting"
        note = (f"{len(kept)} of {counted} reasons still hold, probability "
                f"{now:.3f} against {bought_at:.3f}")
    else:
        status = "intact"
        note = f"every reason it bought on still holds ({now:.3f})"

    return {"status": status, "note": note, "probability_now": round(now, 4),
            "probability_at_entry": round(bought_at, 4),
            "agreement": round(agreement, 3),
            "kept": kept, "lost": lost}


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
                 "confidence": s.get("confidence"),
                 # Why it wants this one, kept so that later it can be asked
                 # whether the reason is still there.
                 "pattern": _pattern_of(s)} for s in chosen],
    }}
    logger.info("Plan: buy %s at the next open (from %s)",
                ", ".join(s["symbol"] for s in chosen), as_of)
    return _append(entry)["plan"]


def _candidate(run, book: dict) -> dict | None:
    """What the model would buy today, if it were asked. Only used at a review."""
    signals = [s for s in (getattr(run, "signals", None) or [])
               if s.get("probability_up") is not None]
    if not signals:
        return None
    best = max(signals, key=lambda s: float(s["probability_up"]))
    if book.get("long_only", True) and float(best["probability_up"]) <= 0.5:
        return None
    return best


def advance(frames: dict, *, book: dict | None = None, today=None,
            run=None) -> dict:
    """Fill a waiting plan, or review a position whose date has arrived.

    Called on a schedule. Does nothing most days, which is the entire point of
    the strategy it is recording.

    A review is not automatically a sale. Pass the latest run and it will ask
    what the model would buy now: the same name means nothing is traded and the
    next review moves out, a different name means one round trip, and nothing
    worth buying means cash. Selling in order to buy the same stock back would
    pay a round trip to stand still -- and the strategy that was tested never
    did that, because it only ever charged for a change.
    """
    book = {**BOOK, **(book or {})}
    current = state()
    out = {"filled": None, "exited": None, "holding": None, "note": None,
           "renewed": None, "watch": None}

    if current["plan"] and not current["holding"]:
        filled = _fill(current["plan"], current["cash"], frames)
        if filled:
            out["filled"] = filled
            current = state()
        else:
            out["note"] = "the session after the plan has not closed yet"

    holding = current["holding"]
    if holding:
        if run is not None:
            out["watch"] = watch(holding, run,
                                 (current.get("watch") or {}).get("status"))
        due = _sessions_left(holding, frames, today=today)
        out["holding"] = {**holding, "sessions_left": due["left"],
                          "review_on": due["review_on"],
                          "held_sessions": due["held"]}
        if due["left"] <= 0:
            held_names = {p["symbol"] for p in holding["bought"]}
            candidate = _candidate(run, book) if run is not None else None

            if candidate and candidate["symbol"] in held_names and len(held_names) == 1:
                renewed = _renew(holding, candidate, due)
                out["renewed"] = renewed
                out["holding"] = {**holding, "session": renewed["from_session"],
                                  "sessions_left": int(book["rebalance_every"]),
                                  "review_on": None}
                out["note"] = (
                    f"reviewed and kept {candidate['symbol']}: the model still "
                    f"calls it up ({float(candidate['probability_up']):.3f}), so "
                    f"nothing was traded and nothing was paid")
            else:
                exited = _exit(holding, current["cash"], frames)
                if exited:
                    out["exited"] = exited
                    out["holding"] = None
                    out["note"] = (
                        f"sold at the review: the model now prefers "
                        f"{candidate['symbol']}" if candidate
                        else "sold at the review: nothing is called up, so cash")
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
            # Carried from the plan: what it thought, and why. Without these
            # the position cannot be asked later whether its case still holds.
            "probability_up": position.get("probability_up"),
            "pattern": position.get("pattern") or [],
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


def watch(holding: dict, run, last_status: str | None) -> dict | None:
    """Re-read the case for what is held, and say so when it changes.

    Deliberately writes nothing unless the verdict changes: a record of events
    is useful, a daily diary of "still fine" is not. It never trades, and that
    is now measured rather than assumed.

    Tested on the validation window, paired -- identical picks, identical
    entries, only the exit differing. The case broke in seven of seven
    episodes, and **holding on from the break returned +8.65% on average**
    (median +3.96%, t +3.31), better in every single one. The money agrees:
    holding to review ended at 1,027 on 14 trades and 56 of fees, while selling
    on a break and switching ended at 495 on 122 trades and 326 of fees -- two
    thirds of the account spent on acting on the warning.

    So a break is worth knowing and is not worth trading. If anybody wires this
    to an exit later, it has to beat that pairing first.
    """
    bought = (holding.get("bought") or [{}])[0]
    signals = {s["symbol"]: s for s in (getattr(run, "signals", None) or [])}
    signal = signals.get(bought.get("symbol"))
    if signal is None:
        return None

    verdict = _compare(bought.get("pattern") or [], signal,
                       float(bought.get("probability_up") or 0.0))
    if verdict["status"] == last_status:
        return verdict

    _append({"watch": {
        "at": _now(), "symbol": bought.get("symbol"),
        "session": signal.get("as_of"), **verdict,
    }})
    logger.info("Watching %s: %s -- %s", bought.get("symbol"),
                verdict["status"], verdict["note"])
    return verdict


def _renew(holding: dict, candidate: dict, due: dict) -> dict:
    """Record a review that confirmed the position. No trade, no fee."""
    entry = {"renew": {
        "at": _now(),
        "symbol": candidate["symbol"],
        "probability_up": candidate["probability_up"],
        "held_since": holding["session"],
        # The clock restarts from the session the review fell on.
        "from_session": due["review_on"] or holding["session"],
    }}
    logger.info("Reviewed and kept %s (%.3f): nothing traded",
                candidate["symbol"], float(candidate["probability_up"]))
    return _append(entry)["renew"]


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


def log(limit: int = 40) -> list:
    """The record as sentences, newest first.

    Every line in the ledger is already an event with a date on it, so the log
    is not a separate thing to maintain -- it is the same record, read aloud.
    A page that says "bought Investor B on the 28th because the model called it
    up, sells on the 19th of December" is worth more than four panels of
    statistics, and it cannot drift from the truth because it is the truth.
    """
    lines = []
    for entry in _read():
        if "plan" in entry:
            plan = entry["plan"]
            names = ", ".join(
                f"{b['symbol']} ({float(b['probability_up']):.1%} up)"
                for b in plan["buy"])
            lines.append({
                "when": plan["as_of"], "kind": "decided",
                "text": f"Decided to buy {names}, from the close of "
                        f"{plan['as_of']}. It fills at the next open.",
            })
        elif "fill" in entry:
            fill = entry["fill"]
            names = ", ".join(f"{b['symbol']} at {b['price']:g}"
                              for b in fill["bought"])
            lines.append({
                "when": fill["session"], "kind": "bought",
                "text": f"Bought {names}. {fill['fees']:.2f} in costs, "
                        f"{fill['cash_after']:.2f} left in cash.",
            })
        elif "watch" in entry:
            watch = entry["watch"]
            reading = {"intact": "still holds", "drifting": "is weakening",
                       "broken": "has broken"}.get(watch["status"], watch["status"])
            lines.append({
                "when": watch.get("session") or watch["at"][:10],
                "kind": watch["status"],
                "text": f"The case for {watch['symbol']} {reading}: "
                        f"{watch['note']}."
                        + (" Holding anyway -- a warning is free and a trade is not."
                           if watch["status"] == "broken" else ""),
            })
        elif "renew" in entry:
            renew = entry["renew"]
            lines.append({
                "when": renew["from_session"], "kind": "kept",
                "text": f"Reviewed {renew['symbol']} and kept it -- still called "
                        f"up at {float(renew['probability_up']):.1%}. Nothing "
                        f"traded, nothing paid.",
            })
        elif "exit" in entry:
            closed = entry["exit"]
            names = ", ".join(
                f"{s['symbol']} at {s['price']:g} ({s['gross_return']:+.1%})"
                for s in closed["sold"])
            lines.append({
                "when": closed["session"], "kind": "sold",
                "text": f"Sold {names}. {closed['profit']:+.2f} after costs, "
                        f"{closed['cash_after']:.2f} in cash.",
            })

    return list(reversed(lines))[:limit]


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
            "watch": current.get("watch"),
            "bought_because": (holding["bought"][0].get("pattern") or []),
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
        "log": log(),
        "schedule": ", ".join(sorted({
            broker_mod.resolve(schedule_for(p["symbol"])).name
            for p in ((holding or {}).get("bought")
                      or (plan or {}).get("buy") or [])
        })) or broker_mod.resolve(SCHEDULE).name,
    }
