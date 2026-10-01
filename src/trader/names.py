"""What each ticker is called, so nothing downstream has to remember.

This exists because of one sentence a 7B wrote: "VOLV-B.ST, the Swedish truck
maker". It was asked not to describe companies and it did anyway, three
prompt revisions running, and the description happened to be right -- Volvo is
a Swedish truck maker -- which is precisely what makes it dangerous. A model
recalling a fact it was told not to recall is a model that will one day recall
a wrong one about a name nobody in the room recognises.

The fix is not a firmer instruction. It is to stop asking it to remember: if
"AB Volvo (publ)" travels beside VOLV-B.ST in the material, writing the name
is reading rather than recall, and a reader can check it. That turned out to
be the fix three times over on this page -- an ambiguous sentence about the
archive's age, a comparison the model was made to compute, and now this.

**Names are not features.** Nothing here reaches a label or a score. A company
name is not a thing anybody can trade on, and the block that could use one
refuses to exist for another year anyway.

**The store is the cache.** Names change rarely -- a rename is news in itself
-- so a symbol is asked about once and kept. A lookup that fails is recorded as
a failure with the time, so a delisted ticker is not re-asked on every cycle,
which is the same bargain `prices.py` strikes with its fetch log.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

STORE_DIR = os.environ.get(
    "TRADER_NAMES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "names"),
)
STORE_FILE = "names.json"

# A name that was found is kept indefinitely; one that was not is re-asked
# after this long. A ticker the provider does not know today may be one it
# knows next week, and may equally be a symbol that no longer exists.
RETRY_HOURS = 168

# How many unknown symbols one pass will ask about. The whole universe is a
# couple of hundred lookups at about a third of a second each, which is a
# minute of somebody's afternoon on the first run and nothing after that.
PER_PASS = 40

_lock = threading.Lock()


def _path() -> str:
    return os.path.join(os.path.abspath(STORE_DIR), STORE_FILE)


def _read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _write(store: dict) -> None:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def name_of(symbol: str) -> str | None:
    """What this ticker is called, if the store knows. Never a guess."""
    entry = _read().get(symbol)
    return entry.get("name") if isinstance(entry, dict) else None


def names_for(symbols) -> dict:
    """Every name the store holds for these symbols. Missing ones are absent."""
    store = _read()
    out = {}
    for symbol in symbols:
        entry = store.get(symbol)
        if isinstance(entry, dict) and entry.get("name"):
            out[symbol] = entry["name"]
    return out


def _stale(entry, now: dt.datetime) -> bool:
    """Whether a previous failure is old enough to be worth repeating."""
    if not isinstance(entry, dict):
        return True
    if entry.get("name"):
        return False
    try:
        asked = dt.datetime.fromisoformat(entry["asked"])
    except (KeyError, TypeError, ValueError):
        return True
    if asked.tzinfo is None:
        asked = asked.replace(tzinfo=dt.timezone.utc)
    return now - asked > dt.timedelta(hours=RETRY_HOURS)


def missing(symbols, *, now: dt.datetime | None = None) -> list:
    """Symbols the store cannot name and is willing to ask about again."""
    now = now or dt.datetime.now(dt.timezone.utc)
    store = _read()
    return [symbol for symbol in symbols if _stale(store.get(symbol), now)]


def _fetch(symbol: str) -> str | None:
    """The provider's long name for one ticker, or None."""
    import yfinance                                     # imported late: slow

    try:
        info = yfinance.Ticker(symbol).info or {}
    except Exception as exc:                            # noqa: BLE001
        logger.info("No name for %s: %s", symbol, exc)
        return None

    # longName over shortName: "Volvo, AB ser. B" is how a terminal abbreviates
    # a name, not how anybody says it.
    name = info.get("longName") or info.get("shortName")
    return str(name).strip() or None if name else None


def learn(symbols, *, limit: int = PER_PASS,
          now: dt.datetime | None = None) -> dict:
    """Ask about the symbols the store cannot name. Safe to run on a timer.

    Returns what it learned this pass. Bounded, because the first run on a
    fresh universe is a couple of hundred network calls and a scheduler cycle
    should not disappear into it.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    wanted = missing(symbols, now=now)[:limit]
    if not wanted:
        return {"asked": 0, "learned": 0, "named": len(names_for(symbols))}

    found = {}
    for symbol in wanted:
        name = _fetch(symbol)
        if name:
            found[symbol] = name

    with _lock:
        store = _read()
        for symbol in wanted:
            store[symbol] = {"name": found.get(symbol), "asked": _now()}
        _write(store)

    if found:
        logger.info("Learned %d name(s): %s", len(found),
                    ", ".join(sorted(found)[:5]))

    return {"asked": len(wanted), "learned": len(found),
            "named": len(names_for(symbols))}


def describe(symbol: str) -> str:
    """The ticker, with its name beside it when one is known.

    The shape everything downstream uses, so a name is written the same way
    wherever it appears and a reader learns to read past it.
    """
    name = name_of(symbol)
    return f"{symbol} ({name})" if name else symbol
