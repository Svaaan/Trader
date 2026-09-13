"""A point-in-time news store, which is the only kind worth training on.

The request that produced this file was "can we scrape economic news to add
context". The answer is yes, and the reason this module looks so unlike a
scraper is that the scraping is the easy half and the timestamps are the hard
one.

**Why a backfilled archive cannot be trusted.** Ask any provider today for news
about a symbol and you get what it has now, ranked by what it thinks matters
now. Three things have happened to that archive since the dates it covers:
stories that turned out to be important were promoted, stories that turned out
to be noise were dropped, and `published_at` is very often the moment the item
entered somebody's database rather than the moment it crossed the wire. A model
trained on that has been shown which stories mattered, which is precisely the
thing it was supposed to work out. None of it is visible in the output. The
accuracy simply comes out high.

The source available here returns ten recent items per symbol. That is not a
limitation to work around -- it is the shape of the honest problem. There is no
archive to backfill from, so the store is built forwards.

**So the store records when *we* saw a thing, not when somebody says it
happened.** `captured_utc` is written by this machine at the moment the item is
appended and cannot be revised by anyone. `published_utc` is kept beside it as
the provider's claim, and features never use it. An item counts towards a
session only if it was captured before that session -- the same one-day lag the
macro and cross-sectional blocks take, for the same reason and with the same
mixed-timezone panel in mind.

**Append-only.** Items are never rewritten and never deleted, because the value
of the file is entirely in the fact that its contents could not have been
adjusted after the fact. A store that gets tidied is an archive again.

**And so news features are inert until the store has history.** Turning them on
over an empty store would feed the model a column of zeros for every historical
row and a real number for today, which is not a feature, it is a date stamp.
`readiness` reports how much history exists; the pipeline refuses to use the
block until it clears MIN_HISTORY_DAYS, and says so rather than quietly
including it.

Start the collector now -- `python watch.py --news` appends on every pass -- and
the block becomes usable in a year. That is the real cost of doing this
honestly, and it is worth knowing before building rather than after.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORE_DIR = os.environ.get(
    "TRADER_NEWS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "news"),
)

NEWS_NAMES = [
    "news_count_1d",
    "news_count_5d",
    "news_burst",
]

# How much captured history the store needs before the block is worth using.
# A year covers a full cycle of reporting seasons and is roughly the point at
# which a count has a distribution rather than a value.
MIN_HISTORY_DAYS = 365

# Sessions of lag, matching macro.LAG_SESSIONS and cross.LAG_SESSIONS. An item
# captured during Tuesday is available to Wednesday's features.
LAG_SESSIONS = 1


def _store_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(os.path.abspath(STORE_DIR), f"{safe}.jsonl")


def _read(symbol: str) -> list:
    """Every item ever captured for this symbol, oldest first."""
    path = _store_path(symbol)
    if not os.path.exists(path):
        return []

    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # One malformed line should not cost the whole history.
                logger.warning("Skipping unparseable line in %s", path)
    return items


def _fetch(symbol: str) -> list:
    """Whatever the provider is showing right now, flattened."""
    import yfinance                                     # imported late: it is slow

    try:
        raw = yfinance.Ticker(symbol).news or []
    except Exception as exc:                            # noqa: BLE001
        logger.warning("No news for %s: %s", symbol, exc)
        return []

    out = []
    for item in raw:
        content = item.get("content") or item
        if not isinstance(content, dict):
            continue

        provider = content.get("provider") or {}
        url = content.get("canonicalUrl") or {}

        out.append({
            "id": str(item.get("id") or content.get("id") or "") or None,
            "title": content.get("title"),
            "summary": content.get("summary") or content.get("description"),
            # The provider's claim, kept for reading and never used as a feature.
            "published_utc": content.get("pubDate") or content.get("displayTime"),
            "provider": provider.get("displayName") if isinstance(provider, dict) else None,
            "url": url.get("url") if isinstance(url, dict) else None,
        })

    return [item for item in out if item.get("id") and item.get("title")]


def collect(symbols, *, now: dt.datetime | None = None) -> dict:
    """One pass: append anything not already stored. Safe to run on a timer.

    Returns how many new items each symbol gained. Idempotent by item id, so
    running it twice in a minute adds nothing the second time and running it
    never simply leaves a gap that is visible in `readiness`.
    """
    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")
    added = {}

    for symbol in symbols:
        known = {item.get("id") for item in _read(symbol)}
        fresh = [item for item in _fetch(symbol) if item["id"] not in known]

        if fresh:
            path = _store_path(symbol)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Append, never rewrite. The whole value of the file is that its
            # earlier lines could not have been adjusted afterwards.
            with open(path, "a", encoding="utf-8") as handle:
                for item in fresh:
                    handle.write(json.dumps({**item, "captured_utc": stamp},
                                            ensure_ascii=False) + "\n")

        added[symbol] = len(fresh)

    total = sum(added.values())
    if total:
        logger.info("News store gained %d items across %d symbols",
                    total, sum(1 for v in added.values() if v))
    return added


def readiness(symbols) -> dict:
    """How much captured history exists, and whether it is enough to use.

    The pipeline consults this rather than assuming. A block that is silently
    all zeros for nine years and real for the last week would be a date stamp
    wearing a feature's name.
    """
    earliest, items = None, 0
    per_symbol = {}

    for symbol in symbols:
        stored = _read(symbol)
        items += len(stored)
        stamps = [item.get("captured_utc") for item in stored
                  if item.get("captured_utc")]
        first = min(stamps) if stamps else None
        per_symbol[symbol] = {"items": len(stored), "since": first}
        if first and (earliest is None or first < earliest):
            earliest = first

    days = 0
    if earliest:
        try:
            start = dt.datetime.fromisoformat(earliest)
            if start.tzinfo is None:
                start = start.replace(tzinfo=dt.timezone.utc)
            days = (dt.datetime.now(dt.timezone.utc) - start).days
        except ValueError:
            days = 0

    return {
        "items": items,
        "since": earliest,
        "history_days": days,
        "needed_days": MIN_HISTORY_DAYS,
        "ready": days >= MIN_HISTORY_DAYS and items > 0,
        "per_symbol": per_symbol,
    }


def build(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """News features for one symbol, counted by capture time.

    Uses `captured_utc` only. An item the provider says was published last
    Tuesday but which this machine first saw today counts from today, because
    that is the only claim about it that cannot have been revised.
    """
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index, columns=NEWS_NAMES, dtype="float64")

    stored = _read(symbol)
    stamps = []
    for item in stored:
        raw = item.get("captured_utc")
        if not raw:
            continue
        try:
            moment = pd.Timestamp(raw)
        except (ValueError, TypeError):
            continue
        stamps.append(moment.tz_localize(None) if moment.tzinfo else moment)

    if not stamps:
        return out.fillna(0.0)

    # Items per calendar day, reindexed onto the sessions we care about and
    # lagged, so a session only ever sees what was captured before it.
    daily = pd.Series(1, index=pd.DatetimeIndex(stamps).normalize())
    daily = daily.groupby(level=0).sum().sort_index()
    aligned = daily.reindex(index.union(daily.index), fill_value=0).sort_index()
    aligned = aligned.shift(LAG_SESSIONS).fillna(0.0)

    count_1d = aligned.rolling(1).sum().reindex(index).fillna(0.0)
    count_5d = aligned.rolling(5).sum().reindex(index).fillna(0.0)
    average = aligned.rolling(60).mean().reindex(index)

    out["news_count_1d"] = np.log1p(count_1d)
    out["news_count_5d"] = np.log1p(count_5d)
    # Today against the usual amount of coverage: a burst is the signal, a
    # steady drip is just a company being large.
    out["news_burst"] = np.log1p(count_1d) - np.log1p(average.fillna(0.0))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def recent(symbol: str, limit: int = 8) -> list:
    """The newest stored items, for the reading panel rather than the model."""
    stored = _read(symbol)
    stored.sort(key=lambda item: item.get("captured_utc") or "", reverse=True)
    return stored[:limit]
