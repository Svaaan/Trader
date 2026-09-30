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

# The book that is actually followed. Every line written before books had names
# belongs to it, so the record it has already built survives being joined.
CHAMPION = "champion"

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


def _lines(book_id: str = CHAMPION) -> list:
    """The lines belonging to one book, in order.

    A line written before any of this existed carries no name and belongs to
    the champion -- the record it has built is the one thing here that cannot
    be recreated, so it is joined rather than restarted.
    """
    return _apply_voids([entry for entry in _read()
                         if entry.get("book_id", CHAMPION) == book_id])


def void(anchor: str, *, kind: str, reason: str,
         book_id: str = CHAMPION) -> dict:
    """Strike a line and everything after it, by appending rather than editing.

    `kind` and `anchor` name the entry to strike from -- its key ("fill",
    "exit", "plan") and its `at` stamp. Both are needed because stamps are
    written to the second, and a plan filled in the same second as it was
    written would otherwise be struck along with the fill.

    Everything the book wrote from there on is dropped when the record is read
    back, so it returns to the state it was in beforehand and can act again.

    The reason is required and kept. A record that can be silently corrected is
    not a forward record, so the correction is a line in it -- and a reader can
    see both that something was wrong and what was said about it.
    """
    if not reason:
        raise ValueError("a void needs a reason; that is the point of it")
    entry = {"void": {"from": anchor, "kind": kind, "reason": reason,
                      "at": _now()}}
    logger.warning("Voided (%s) from the %s at %s: %s",
                   book_id, kind, anchor, reason)
    return _append(entry, book_id=book_id)["void"]


def _apply_voids(entries: list) -> list:
    """Drop what the void lines strike, in file order.

    A mark carries no stamp of its own, so "everything after" is positional:
    the anchor names one entry and the strike runs from it to the end of what
    had been written when the void was appended.
    """
    if not any("void" in entry for entry in entries):
        return entries

    # Walked in file order, because a void strikes what had already been
    # written when it was appended and nothing else. Filtering the voids out
    # first and truncating afterwards loses that boundary, and silently drops
    # whatever the book does next -- which is the whole point of voiding.
    kept: list = []
    for entry in entries:
        if "void" not in entry:
            kept.append(entry)
            continue
        mark = entry["void"]
        kind, anchor = mark.get("kind"), mark.get("from")
        cut = next((i for i, seen in enumerate(kept)
                    if isinstance(seen.get(kind), dict)
                    and seen[kind].get("at") == anchor), None)
        if cut is not None:
            kept = kept[:cut]
    return kept


def books() -> dict:
    """Every book in the contest: its rule, whether it is funded, and why.

    Registration is a line like any other, because a challenger joining is an
    event and the order of events is the whole value of this file.
    """
    known = {CHAMPION: {"id": CHAMPION, "rule": dict(BOOK), "funded": True,
                        "why": "the book the search committed", "since": None}}
    for entry in _read():
        if "register" in entry:
            item = entry["register"]
            known[item["id"]] = {
                "id": item["id"], "rule": {**BOOK, **(item.get("rule") or {})},
                "funded": bool(item.get("funded")), "why": item.get("why", ""),
                "since": item.get("at"), "expected": item.get("expected"),
            }
        elif "fund" in entry:
            # A fund line naming nothing stands everything down.
            for book in known.values():
                book["funded"] = book["id"] == entry["fund"].get("id")
    return known


def register(book_id: str, *, rule: dict, why: str, funded: bool = False,
             expected: float | None = None) -> dict:
    """Enter a book into the contest. Shadows are recorded, never funded.

    A challenger earns its record the same way the champion does -- same
    machinery, same costs, same forward-only discipline -- and the only
    difference is that nobody would have traded it.
    """
    if book_id in books() and book_id != CHAMPION:
        raise ValueError(f"{book_id} is already in the contest")

    entry = {"register": {"id": book_id, "rule": {**BOOK, **(rule or {})},
                          "why": why, "funded": bool(funded), "at": _now(),
                          # What its backtest expected per decision. The guard
                          # compares the forward record against this rather
                          # than against a number chosen after the fact.
                          "expected": expected}}
    logger.info("Registered %s: %s", book_id, why)
    return _append(entry, book_id=book_id)["register"]


def fund(book_id: str, *, why: str) -> dict:
    """Make this the book that would actually be traded. Exactly one at a time."""
    if book_id not in books():
        raise ValueError(f"{book_id} is not in the contest")

    entry = {"fund": {"id": book_id, "why": why, "at": _now()}}
    logger.warning("Funded book is now %s: %s", book_id, why)
    return _append(entry, book_id=book_id)["fund"]


def funded() -> str | None:
    """The book that would actually be traded, or None if trading has stopped.

    None is a real answer, not a missing one: `stand_down` records that nothing
    is being followed, and every book keeps its record either way -- you want to
    know whether the one you stopped would have recovered.
    """
    for book_id, book in books().items():
        if book["funded"]:
            return book_id
    return None


def stand_down(*, why: str) -> dict:
    """Stop funding anything. The records continue; the money does not.

    Only ever called by the guard in promote.py, and it only ever stops --
    nothing here can start trading something on its own.
    """
    entry = {"fund": {"id": None, "why": why, "at": _now()}}
    logger.warning("Standing down: %s", why)
    return _append(entry, book_id=CHAMPION)["fund"]


def rule_for(book_id: str = CHAMPION) -> dict:
    return books().get(book_id, {}).get("rule", dict(BOOK))


def _path() -> str:
    return os.path.join(os.path.abspath(STORE_DIR), STORE_FILE)


_cache: dict = {}


def _read() -> list:
    """The whole record, parsed once per version of the file.

    Stepping a contest of N books read the file N times per cycle, and every
    state() inside that read it again. The cache is keyed on what the file
    actually is -- its size and modification time -- so an append by any
    process invalidates it, and nothing here can serve a stale record.
    """
    path = _path()
    try:
        stat = os.stat(path)
    except OSError:
        return []

    signature = (path, stat.st_mtime_ns, stat.st_size)
    if _cache.get("signature") == signature:
        return _cache["lines"]

    try:
        with open(path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return []

    _cache["signature"], _cache["lines"] = signature, lines
    return lines


def _append(entry: dict, *, book_id: str = CHAMPION) -> dict:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({**entry, "book_id": book_id},
                                ensure_ascii=False) + "\n")
    return entry


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def state(book_id: str = CHAMPION) -> dict:
    """Where one book stands, read from its own lines in order.

    Returns the plan waiting to be filled (if any), the position being held (if
    any), everything already closed, and the cash. Each book keeps its own
    account, so their records are comparable without unpicking a shared one.
    """
    cash, plan, holding, closed, watched = STARTING_CASH, None, None, [], None
    marked = None

    for entry in _lines(book_id):
        if "plan" in entry:
            plan = entry["plan"]
        elif "fill" in entry:
            holding = entry["fill"]
            cash = entry["fill"]["cash_after"]
            plan = None
        elif "mark" in entry:
            marked = entry["mark"]["session"]
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
            "watch": watched, "marked": marked}


def mark(frames: dict, *, book_id: str = CHAMPION, today=None) -> dict | None:
    """Write down what this book is worth today. Once per session, per book.

    A book reviewed every sixty sessions closes four decisions a year, so a
    contest judged on closed decisions needs two years to say anything. But
    every session is already out-of-sample for every book -- the mark is
    written before the next day exists, which is the same property that makes
    the rest of this record un-mineable. Marking daily turns four observations
    a year into two hundred and fifty, and lets the difference between two
    books be tested while anybody still cares.
    """
    current = state(book_id)
    holding = current["holding"]

    session = None
    worth = current["cash"]
    if holding:
        for bought in holding["bought"]:
            frame = frames.get(bought["symbol"])
            if frame is None or not len(frame):
                return None
            index = frame.index
            if today is not None:
                index = index[index <= pd.Timestamp(today)]
                if not len(index):
                    return None
            session = index[-1] if session is None else min(session, index[-1])
            worth += bought["shares"] * float(frame["close"].loc[index[-1]])
    else:
        # In cash, so the mark rides whatever calendar the others are on.
        for frame in frames.values():
            if frame is not None and len(frame):
                index = frame.index
                if today is not None:
                    index = index[index <= pd.Timestamp(today)]
                if len(index):
                    session = index[-1] if session is None else session
                    break

    if session is None:
        return None

    stamp = pd.Timestamp(session).date().isoformat()
    if current.get("marked") == stamp:
        return None

    entry = {"mark": {"session": stamp, "equity": round(worth, 4),
                      "holding": [b["symbol"] for b in (holding or {}).get("bought", [])]}}
    return _append(entry, book_id=book_id)["mark"]


def marks(book_id: str = CHAMPION) -> list:
    """The daily worth of one book, oldest first."""
    return [entry["mark"] for entry in _lines(book_id) if "mark" in entry]


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
    """Whatever the record needs prices for, across every book in the contest.

    The same rule the other ledger learned the hard way -- what to fetch is
    named by the record, not by whichever universe the caller happens to have.
    """
    names = set()
    for book_id in books():
        current = state(book_id)
        if current["plan"]:
            names.update(p["symbol"] for p in current["plan"]["buy"])
        if current["holding"]:
            names.update(p["symbol"] for p in current["holding"]["bought"])
    return names


def plan_next(run, *, book: dict | None = None,
              book_id: str = CHAMPION) -> dict | None:
    """Write down what to buy, before the price it will be bought at exists.

    Refuses if something is already held or already planned: a book holds one
    thing at a time, and the point of the interval is that it is left alone.
    """
    book = {**rule_for(book_id), **(book or {})}
    current = state(book_id)
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
    logger.info("Plan (%s): buy %s at the next open (from %s)", book_id,
                ", ".join(s["symbol"] for s in chosen), as_of)
    return _append(entry, book_id=book_id)["plan"]


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
            run=None, book_id: str = CHAMPION) -> dict:
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
    book = {**rule_for(book_id), **(book or {})}
    current = state(book_id)
    out = {"book_id": book_id, "filled": None, "exited": None, "holding": None,
           "note": None, "renewed": None, "watch": None}

    if current["plan"] and not current["holding"]:
        filled = _fill(current["plan"], current["cash"], frames, book_id)
        if filled:
            out["filled"] = filled
            current = state(book_id)
        else:
            out["note"] = "the session after the plan has not closed yet"

    holding = current["holding"]
    if holding:
        if run is not None:
            out["watch"] = watch(holding, run,
                                 (current.get("watch") or {}).get("status"),
                                 book_id=book_id)
        due = _sessions_left(holding, frames, today=today)
        out["holding"] = {**holding, "sessions_left": due["left"],
                          "review_on": due["review_on"],
                          "held_sessions": due["held"]}
        if due["left"] <= 0:
            held_names = {p["symbol"] for p in holding["bought"]}
            candidate = _candidate(run, book) if run is not None else None

            if candidate and candidate["symbol"] in held_names and len(held_names) == 1:
                renewed = _renew(holding, candidate, due, book_id)
                out["renewed"] = renewed
                out["holding"] = {**holding, "session": renewed["from_session"],
                                  "sessions_left": int(book["rebalance_every"]),
                                  "review_on": None}
                out["note"] = (
                    f"reviewed and kept {candidate['symbol']}: the model still "
                    f"calls it up ({float(candidate['probability_up']):.3f}), so "
                    f"nothing was traded and nothing was paid")
            else:
                exited = _exit(holding, current["cash"], frames, book_id)
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


def advance_all(frames: dict, *, run=None, today=None) -> dict:
    """Step every book in the contest, each by its own rule.

    The champion and the challengers see exactly the same prices and the same
    model on the same day; what differs is only what each does about it.
    """
    stepped = {}
    for book_id in books():
        stepped[book_id] = advance(frames, today=today, run=run, book_id=book_id)
        mark(frames, book_id=book_id, today=today)
    return stepped


def plan_all(run, *, today=None) -> dict:
    """Let any book that is in cash choose, from the same signals."""
    return {book_id: plan_next(run, book_id=book_id) for book_id in books()}


def _fill(plan: dict, cash: float, frames: dict,
          book_id: str = CHAMPION) -> dict | None:
    """Buy at the first open after the plan was written.

    `value` is the whole allocation, and all of it leaves the account: the fee
    is paid out of it and the rest buys shares. Accumulating only the fee left
    the allocation sitting in cash while the shares it bought were also counted,
    so a $500 book that bought one position marked itself at $991 the next day.
    """
    bought, spent, charged = [], 0.0, 0.0
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
        spent += value
        charged += fee

    entry = {"fill": {
        "at": _now(), "as_of": plan["as_of"], "run_id": plan.get("run_id"),
        "book": plan["book"], "trusted": plan.get("trusted"),
        "bought": bought,
        "session": bought[0]["session"],
        "cash_before": round(cash, 4),
        "cash_after": round(cash - spent, 4),
        "fees": round(charged, 4),
    }}
    logger.info("Filled (%s): %s at %s", book_id, ", ".join(
        f"{b['symbol']} {b['price']}" for b in bought), bought[0]["session"])
    return _append(entry, book_id=book_id)["fill"]


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


def watch(holding: dict, run, last_status: str | None,
          *, book_id: str = CHAMPION) -> dict | None:
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

    # A position bought before the pattern was recorded has nothing to compare
    # today's reasons against, and `_compare` says so as "0 of 0 reasons still
    # hold" -- a verdict computed from nothing, phrased as a measurement. The
    # whole argument for this warning is that it is read off the record, so
    # when there is no record of the case, there is no warning.
    if not (bought.get("pattern") or []):
        return None

    verdict = _compare(bought["pattern"], signal,
                       float(bought.get("probability_up") or 0.0))
    if verdict["status"] == last_status:
        return verdict

    _append({"watch": {
        "at": _now(), "symbol": bought.get("symbol"),
        "session": signal.get("as_of"), **verdict,
    }}, book_id=book_id)
    logger.info("Watching %s: %s -- %s", bought.get("symbol"),
                verdict["status"], verdict["note"])
    return verdict


def _renew(holding: dict, candidate: dict, due: dict,
           book_id: str = CHAMPION) -> dict:
    """Record a review that confirmed the position. No trade, no fee."""
    entry = {"renew": {
        "at": _now(),
        "symbol": candidate["symbol"],
        "probability_up": candidate["probability_up"],
        "held_since": holding["session"],
        # The clock restarts from the session the review fell on.
        "from_session": due["review_on"] or holding["session"],
    }}
    logger.info("Reviewed and kept %s (%.3f) for %s: nothing traded",
                candidate["symbol"], float(candidate["probability_up"]), book_id)
    return _append(entry, book_id=book_id)["renew"]


def _exit(holding: dict, cash: float, frames: dict,
          book_id: str = CHAMPION) -> dict | None:
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
    logger.info("Sold (%s): %s at %s, profit %.2f", book_id, ", ".join(
        s["symbol"] for s in sold), sold[0]["session"], entry["exit"]["profit"])
    return _append(entry, book_id=book_id)["exit"]


def log(limit: int = 40, book_id: str = CHAMPION) -> list:
    """The record as sentences, newest first.

    Every line in the ledger is already an event with a date on it, so the log
    is not a separate thing to maintain -- it is the same record, read aloud.
    A page that says "bought Investor B on the 28th because the model called it
    up, sells on the 19th of December" is worth more than four panels of
    statistics, and it cannot drift from the truth because it is the truth.
    """
    lines = []
    for entry in _lines(book_id):
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


def account(frames: dict | None = None, *, today=None,
            book_id: str = CHAMPION) -> dict:
    """What the page shows: the pick, the plan to sell, and what it has done."""
    current = state(book_id)
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
        "book": rule_for(book_id),
        "waiting_to_buy": plan,
        "position": position,
        "closed": closed[-10:],
        "log": log(book_id=book_id),
        "book_id": book_id,
        "funded": books().get(book_id, {}).get("funded", book_id == CHAMPION),
        "why": books().get(book_id, {}).get("why", ""),
        "rule": rule_for(book_id),
        "schedule": ", ".join(sorted({
            broker_mod.resolve(schedule_for(p["symbol"])).name
            for p in ((holding or {}).get("bought")
                      or (plan or {}).get("buy") or [])
        })) or broker_mod.resolve(SCHEDULE).name,
    }


def accounts(frames: dict | None = None, *, today=None) -> list:
    """Every book side by side, the funded one first.

    This is the contest: the same machinery, the same costs and the same days,
    with nothing shared between them except the market.
    """
    out = [account(frames, today=today, book_id=book_id) for book_id in books()]
    return sorted(out, key=lambda book: (not book["funded"], book["book_id"]))
