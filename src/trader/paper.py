"""A forward paper-trading ledger: what the model would have done, and what it cost.

Every other measurement in this project is a backtest, and a backtest can be
mined. This one cannot, because when each entry is written the outcome has not
happened yet. That single property is worth more than any amount of careful
splitting, and it is also fragile in one specific way, so the rules here exist
to protect it.

**The ledger is append-only and the model never reads it.** It is a scorecard,
not an input. The moment a system tunes itself on its own paper results, those
results stop being a forward test and become another training set -- and you
will have spent months of calendar time producing a number with exactly the
problem the backtest had, except now you believe it more because it came from
"live" trading. `dataset.py` cannot import this module and nothing here returns
a feature.

What the system may legitimately do with it is *notice*: when the live record
diverges from what the backtest predicted, the model has stopped working, and
that is the one thing worth being told. See `divergence`.

**It fills at the next open, because that is when an order can go in.** The
signal is computed from a close. Nobody knows that close until the session has
ended, so nobody can be positioned at it. An entry written tonight is filled at
tomorrow's opening price and marked at tomorrow's close. This is not a
conservative choice -- it is the only honest one, and it means the ledger cannot
reproduce the close-to-close numbers that make a backtest look good. Measured on
this project's panel, that difference is the entire apparent edge.

**It respects the gate.** When the model has not earned an opinion the ledger
still records what it *would* have done, marked `shadow`, because a forward
record of a model with no edge is exactly how you learn that it still has none.
It is never presented as a recommendation.

**Concentration is a setting, and the default is not to.** The obvious design --
pick the five most confident names and back them heavily -- was measured on this
panel and is worse than trading everything, because the most confident calls are
the ones whose edge lives in the overnight gap:

    top N       close->close      open->close
        5      +26.0%  s 1.38    -7.3%  s -0.48
       10      +30.6%  s 1.94    -1.2%  s -0.05
      119      +17.0%  s 2.98    +5.1%  s  1.14

So `top_n=None` holds the whole book by default. Setting it concentrates, and
the P&L page will show what that did.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LEDGER_DIR = os.environ.get(
    "TRADER_PAPER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "paper"),
)

LEDGER_FILE = "ledger.jsonl"
ACCOUNT_FILE = "account.json"

# What the account starts with. Small on purpose: it is the size at which
# per-trade commission stops being a rounding error, which is a fact worth
# having in front of you rather than discovering later.
STARTING_CASH = 500.0

# Charged on the value traded, each way. The same figure evaluate.py uses.
COST_PER_SIDE = 0.0005


@dataclasses.dataclass
class Position:
    """One intended trade, before it has been filled or marked."""

    symbol: str
    direction: int              # +1 long, -1 short
    weight: float               # fraction of the book
    probability: float
    confidence: float


def _paths() -> tuple:
    root = os.path.abspath(LEDGER_DIR)
    return os.path.join(root, LEDGER_FILE), os.path.join(root, ACCOUNT_FILE)


def _read_ledger() -> list:
    path, _ = _paths()
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping an unparseable ledger line")
    return entries


def _append(entry: dict) -> None:
    path, _ = _paths()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Append, never rewrite. A ledger that can be edited afterwards is not a
    # forward record, it is a story about one.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- sizing -----------------------------------------------------------------

def size_positions(signals: list, *, top_n: int | None = None,
                   long_short: bool = True,
                   max_weight: float = 0.25) -> list:
    """Turn today's probabilities into a book, weighted by conviction.

    Weight is proportional to how far the model is from a coin flip, which is
    the fractional-Kelly shape without pretending the probabilities are
    calibrated well enough for real Kelly. They are not: on this panel they sit
    within about 0.02 of 0.5, and full Kelly on a barely-positive edge is how
    accounts end.

    `max_weight` caps any single name, because a model that is briefly very
    confident about one symbol should not be able to put the account in it.
    """
    usable = [s for s in signals
              if s.get("probability_up") is not None
              and np.isfinite(s.get("probability_up", np.nan))]
    if not usable:
        return []

    ranked = sorted(usable, key=lambda s: s["probability_up"])

    if top_n:
        if long_short:
            chosen = ranked[:top_n] + ranked[-top_n:]
        else:
            chosen = ranked[-top_n:]
    else:
        chosen = ranked

    positions = []
    for signal in chosen:
        probability = float(signal["probability_up"])
        direction = 1 if probability > 0.5 else -1
        if not long_short and direction < 0:
            continue
        # Distance from a coin flip, which is all the conviction there is.
        edge = abs(probability - 0.5)
        positions.append(Position(
            symbol=signal["symbol"], direction=direction, weight=edge,
            probability=round(probability, 6),
            confidence=round(float(signal.get("confidence") or edge * 2), 6)))

    total = sum(p.weight for p in positions)
    if total <= 0:
        # Every name is exactly at a coin flip. Equal weight rather than a
        # division by zero, and the P&L will say it earned nothing.
        for position in positions:
            position.weight = 1.0 / max(len(positions), 1)
        return positions

    weights = np.array([p.weight for p in positions], dtype=float)
    weights /= weights.sum()

    # Cap, then give the excess to the names that are not capped, then check
    # again -- because redistributing can push another name over.
    #
    # Capping once and renormalising afterwards does not cap anything: with one
    # name at 0.999 and the rest near zero, the cap trims it to 0.25 and the
    # renormalisation puts it straight back to 0.998. The account would have
    # gone into a single symbol the model happened to be briefly certain about,
    # which is the exact thing max_weight exists to prevent.
    if max_weight * len(weights) <= 1.0:
        # The cap cannot be satisfied and add to one; equal weight is the only
        # book that respects it.
        weights = np.full(len(weights), 1.0 / len(weights))
    else:
        for _ in range(len(weights)):
            over = weights > max_weight
            if not over.any():
                break
            excess = float((weights[over] - max_weight).sum())
            weights[over] = max_weight
            room = ~over
            if not room.any():
                break
            share = weights[room]
            weights[room] = (share + excess * share / share.sum()
                             if share.sum() > 0
                             else share + excess / room.sum())

    for position, weight in zip(positions, weights):
        position.weight = float(weight)

    return positions


# --- writing the forward record --------------------------------------------

def record_intent(run, *, top_n: int | None = None, long_short: bool = True,
                  now: dt.datetime | None = None) -> dict | None:
    """Write what the model intends to do, before the outcome exists.

    Called once per finished run. Refuses to write a second entry for the same
    signal date, so running it twice in an evening does not double the book.
    """
    signals = run.signals or []
    if not signals:
        logger.info("No signals to record")
        return None

    as_of = max(s["as_of"] for s in signals)
    if any(entry["as_of"] == as_of for entry in _read_ledger()):
        logger.info("Paper ledger already has an entry for %s", as_of)
        return None

    trusted = bool((run.trust or {}).get("trusted"))
    positions = size_positions(signals, top_n=top_n, long_short=long_short)
    if not positions:
        return None

    entry = {
        "as_of": as_of,
        "written_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(
            timespec="seconds"),
        "run_id": run.run_id,
        # A record made while the gate was shut is still worth having; it is
        # just not a recommendation, and it says so.
        "shadow": not trusted,
        "gate_reason": (run.trust or {}).get("reason", ""),
        "top_n": top_n,
        "long_short": long_short,
        "positions": [dataclasses.asdict(p) for p in positions],
        # Filled in later, by settle(), once the prices exist.
        "settled": False,
    }
    _append(entry)
    logger.info("Paper ledger: %d positions recorded for %s%s",
                len(positions), as_of, " (shadow)" if not trusted else "")
    return entry


def settle(frames: dict, *, now: dt.datetime | None = None) -> dict:
    """Fill and mark every entry whose prices have since arrived.

    Entry at the open after the signal date, exit at that session's close. Both
    sides pay `COST_PER_SIDE`. Anything whose session has not happened yet is
    left alone and picked up on a later pass.
    """
    entries = _read_ledger()
    unsettled = [e for e in entries if not e.get("settled")]
    if not unsettled:
        return {"settled": 0, "pending": 0}

    settled_count = 0
    results = []

    for entry in unsettled:
        as_of = pd.Timestamp(entry["as_of"])
        marks, missing = [], []

        for position in entry["positions"]:
            prices = frames.get(position["symbol"])
            if prices is None:
                missing.append(position["symbol"])
                continue

            later = prices.index[prices.index > as_of]
            if not len(later):
                missing.append(position["symbol"])
                continue

            session = later[0]
            row = prices.loc[session]
            gross = float(row["close"] / row["open"] - 1.0) * position["direction"]
            # Both sides of a same-day round trip.
            net = gross - 2 * COST_PER_SIDE

            marks.append({
                "symbol": position["symbol"],
                "session": session.date().isoformat(),
                "weight": position["weight"],
                "direction": position["direction"],
                "open": round(float(row["open"]), 6),
                "close": round(float(row["close"]), 6),
                "gross_return": round(gross, 6),
                "net_return": round(net, 6),
                "contribution": round(net * position["weight"], 8),
            })

        if not marks:
            continue

        # A position whose price never arrived is dropped and the rest
        # renormalised, rather than silently counted as flat.
        held = sum(m["weight"] for m in marks)
        scale = 1.0 / held if held > 0 else 0.0
        day_return = sum(m["contribution"] for m in marks) * scale

        results.append({
            "as_of": entry["as_of"],
            "session": marks[0]["session"],
            "shadow": entry.get("shadow", True),
            "positions": len(marks),
            "missing": missing,
            "return": round(day_return, 8),
            "hit_rate": round(
                float(np.mean([m["net_return"] > 0 for m in marks])), 4),
            "marks": marks,
        })
        settled_count += 1

    if results:
        # Settlement is written as new lines too -- the intent line stays
        # exactly as it was written, and the outcome sits beside it.
        for result in results:
            _append({"settlement": result,
                     "written_utc": (now or dt.datetime.now(dt.timezone.utc))
                     .isoformat(timespec="seconds")})

        # And the intent lines are marked settled in a rewritten index rather
        # than by editing history.
        _mark_settled({r["as_of"] for r in results})

    return {"settled": settled_count,
            "pending": len(unsettled) - settled_count}


def _mark_settled(dates: set) -> None:
    """Record which intents have outcomes, without touching the ledger.

    The ledger is append-only on purpose, so "which of these is done" lives in
    a separate index that can be rebuilt from the ledger at any time.
    """
    _, account_path = _paths()
    os.makedirs(os.path.dirname(account_path), exist_ok=True)

    state = {}
    if os.path.exists(account_path):
        try:
            with open(account_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception:                               # noqa: BLE001
            state = {}

    state["settled"] = sorted(set(state.get("settled", [])) | set(dates))
    with open(account_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _settled_dates() -> set:
    _, account_path = _paths()
    if not os.path.exists(account_path):
        return set()
    try:
        with open(account_path, encoding="utf-8") as handle:
            return set(json.load(handle).get("settled", []))
    except Exception:                                   # noqa: BLE001
        return set()


def _read_ledger_with_state() -> list:
    settled = _settled_dates()
    entries = []
    for entry in _read_ledger():
        if "settlement" in entry:
            entries.append(entry)
            continue
        entry = dict(entry)
        entry["settled"] = entry["as_of"] in settled
        entries.append(entry)
    return entries


# --- reading it back --------------------------------------------------------

def account(starting_cash: float = STARTING_CASH) -> dict:
    """The equity curve and everything a P&L page needs, with error bars.

    The statistics carry the same discipline as the rest of the project: a few
    weeks of paper trading cannot distinguish a strategy from a coin however
    good the total looks, and this says so rather than printing a percentage.
    """
    settlements = [e["settlement"] for e in _read_ledger()
                   if isinstance(e, dict) and "settlement" in e]
    settlements.sort(key=lambda s: s["session"])

    if not settlements:
        # The full shape, not a stub. An empty ledger is the normal state on
        # day one and every reader of this would otherwise need its own
        # special case -- which is how a page ends up raising KeyError the
        # first time somebody opens it.
        return {"starting_cash": starting_cash, "cash": starting_cash,
                "profit": 0.0, "return": 0.0, "days": 0, "wins": 0,
                "hit_rate": None, "sharpe": 0.0, "tstat": 0.0,
                "max_drawdown": 0.0, "shadow_days": 0, "equity": [],
                "pending": _pending_count(),
                "verdict": "Nothing has settled yet."}

    equity, cash = [], starting_cash
    for day in settlements:
        cash *= (1.0 + day["return"])
        equity.append({"session": day["session"],
                       "return": day["return"],
                       "equity": round(cash, 2),
                       "positions": day["positions"],
                       "shadow": day["shadow"],
                       "hit_rate": day["hit_rate"]})

    returns = np.array([d["return"] for d in settlements], dtype=float)
    days = len(returns)
    spread = float(returns.std())
    tstat = (float(returns.mean()) / (spread / math.sqrt(days))
             if spread and days > 1 else 0.0)
    sharpe = (float(returns.mean()) / spread * math.sqrt(252)
              if spread else 0.0)

    curve = np.array([d["equity"] for d in equity], dtype=float)
    peak = np.maximum.accumulate(curve)
    drawdown = float((curve / peak - 1.0).min()) if len(curve) else 0.0

    return {
        "starting_cash": starting_cash,
        "cash": round(cash, 2),
        "profit": round(cash - starting_cash, 2),
        "return": round(cash / starting_cash - 1.0, 6),
        "days": days,
        "wins": int((returns > 0).sum()),
        "hit_rate": round(float(np.mean([d["hit_rate"] for d in settlements])), 4),
        "sharpe": round(sharpe, 3),
        "tstat": round(tstat, 3),
        "max_drawdown": round(drawdown, 4),
        "shadow_days": int(sum(1 for d in settlements if d["shadow"])),
        "equity": equity,
        "pending": _pending_count(),
        "verdict": _verdict(days, tstat, cash - starting_cash),
    }


def _pending_count() -> int:
    settled = _settled_dates()
    return sum(1 for e in _read_ledger()
               if "settlement" not in e and e["as_of"] not in settled)


def _verdict(days: int, tstat: float, profit: float) -> str:
    """One sentence, and it refuses to be impressed by a small sample."""
    if days < 20:
        return (f"{days} settled day(s). Far too few to mean anything -- a "
                f"fortnight of coin flips produces results that look like this "
                f"in both directions.")
    if days < 120:
        return (f"{days} settled days, t = {tstat:.2f}. Still short: a few "
                f"months cannot separate a strategy from luck, whichever way "
                f"the total points.")
    if abs(tstat) < 2.0:
        return (f"{days} days and a t-statistic of {tstat:.2f}. Whatever the "
                f"running total says, this is not distinguishable from chance.")
    if profit > 0:
        return (f"{days} days, t = {tstat:.2f}, up {profit:.2f}. Worth taking "
                f"seriously -- and worth checking against the backtest it was "
                f"supposed to reproduce before anything else.")
    return (f"{days} days, t = {tstat:.2f}, down {abs(profit):.2f}. "
            f"Consistently losing, which is at least a clear answer.")


def divergence(account_state: dict, evaluation: dict) -> dict:
    """Whether live results look like the backtest said they would.

    This is the only legitimate feedback loop in the design. The model must not
    train on its own paper results -- that turns the one un-mineable record in
    the project into another training set. But *noticing* that live performance
    has come apart from what was predicted is exactly the thing worth being
    told, because it is how you learn the model has stopped working.
    """
    days = int(account_state.get("days") or 0)
    if days < 20 or not evaluation:
        return {"comparable": False,
                "note": f"{days} settled days is too few to compare against "
                        f"anything."}

    expected = float(evaluation.get("executable_sharpe") or 0.0)
    actual = float(account_state.get("sharpe") or 0.0)

    # The standard error of a Sharpe over `days` observations is roughly
    # sqrt(252/days), which is wide at these sample sizes and should be.
    error = math.sqrt(252.0 / days)
    gap = actual - expected

    return {
        "comparable": True,
        "expected_sharpe": round(expected, 3),
        "actual_sharpe": round(actual, 3),
        "gap": round(gap, 3),
        "standard_error": round(error, 3),
        "diverged": abs(gap) > 2 * error,
        "note": (
            f"Live Sharpe {actual:.2f} against a backtest that said "
            f"{expected:.2f}. The gap is {abs(gap) / error:.1f} standard errors "
            f"at this sample size"
            + (" -- they have come apart, and the backtest is the one to stop "
               "believing." if abs(gap) > 2 * error else
               ", which is within what a record this short can tell apart.")),
    }
