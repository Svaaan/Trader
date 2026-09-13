"""How close the next scheduled announcement is, and how recent the last one was.

Two integers per symbol per day, and they are the best value-for-effort in this
whole project. Post-earnings-announcement drift -- prices continuing to move in
the direction of a surprise for weeks after it lands -- is among the few equity
anomalies that has survived fifty years of people trying to arbitrage it away,
and it is visible at exactly the daily horizon this project predicts on. It
needs no scraping, no language model, and no vendor.

It is also point-in-time clean by construction, which almost nothing in the news
family is. A company announces the date it will report several weeks ahead, so
"there is an announcement in nine days" was knowable nine days out. Contrast a
sentiment score, where the honest version needs the wire timestamp, the revision
history, and an archive that was not re-ranked by what turned out to matter.

Two limits, both deliberate and both visible in the output rather than papered
over:

**The schedule is only known a few weeks ahead.** A gap of ninety days was not
knowable ninety days out; nobody had announced it yet. So the countdown is
clipped at `SCHEDULE_HORIZON` and everything beyond it reads the same -- "far",
which is genuinely all that was known.

**Coverage is uneven.** US names come back with a decade of quarterly dates.
Some European names come back with three. A symbol with too few is given neutral
values and *named in the coverage report*, so a panel that is quietly half
unlabelled shows up as a number rather than as a mystery in the results.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from concurrent import futures

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get(
    "TRADER_EVENT_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "events"),
)

EVENT_NAMES = [
    "days_to_earnings",
    "days_since_earnings",
    "in_earnings_drift",
]

# Beyond this many days, "when is the next one" had not been announced yet, so
# the model is told "far" rather than a number nobody had.
SCHEDULE_HORIZON = 30

# How long drift is worth marking after a report. The literature puts the effect
# at roughly a quarter; sixty sessions is most of it without running into the
# next announcement.
DRIFT_WINDOW = 60

# Below this many known dates a symbol is treated as having no coverage, rather
# than having its two or three dates stretched across ten years.
MIN_DATES = 8

# Shared with prices.py: these are waits on somebody else's server, and a few
# in flight is most of the win.
FETCH_WORKERS = int(os.environ.get("TRADER_FETCH_WORKERS", "6"))


def _cache_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(os.path.abspath(CACHE_DIR), f"{safe}.json")


def earnings_dates(symbol: str, *, refresh: bool = False) -> list[dt.date]:
    """Every reporting date this provider knows about, cached to disk.

    Past reporting dates are facts and do not get revised, so caching them is
    safe in the way caching a revised economic series is not.
    """
    path = _cache_path(symbol)

    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            return [dt.date.fromisoformat(d) for d in raw.get("dates", [])]
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Event cache for %s unreadable (%s); refetching",
                           symbol, exc)

    import yfinance                                     # imported late: it is slow

    dates: list[dt.date] = []
    try:
        frame = yfinance.Ticker(symbol).get_earnings_dates(limit=80)
        if frame is not None and len(frame):
            dates = sorted({ts.date() for ts in pd.DatetimeIndex(frame.index)})
    except Exception as exc:                            # noqa: BLE001
        logger.warning("No earnings calendar for %s: %s", symbol, exc)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"symbol": symbol,
                   "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "dates": [d.isoformat() for d in dates]}, handle, indent=2)

    return dates


def has_coverage(symbol: str, *, refresh: bool = False) -> bool:
    """Whether this symbol has enough reporting dates to be worth using."""
    return len(earnings_dates(symbol, refresh=refresh)) >= MIN_DATES


def build(symbol: str, index: pd.DatetimeIndex, *,
          refresh: bool = False) -> pd.DataFrame:
    """Event features for one symbol, on the dates it is asked for.

    A symbol with no usable calendar gets the neutral row -- maximally far from
    any announcement, not in a drift window -- which is the same thing the model
    sees for a name that genuinely has nothing scheduled. `has_coverage` is how
    a caller tells the two apart, and pipeline records it per run.
    """
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index, columns=EVENT_NAMES, dtype="float64")

    dates = earnings_dates(symbol, refresh=refresh)

    if len(dates) < MIN_DATES:
        out["days_to_earnings"] = 1.0            # scaled: 1.0 is the far end
        out["days_since_earnings"] = 1.0
        out["in_earnings_drift"] = 0.0
        return out

    stamps = np.array([np.datetime64(d) for d in dates], dtype="datetime64[D]")
    days = index.values.astype("datetime64[D]")

    # For each row, where the next announcement sits in the sorted list.
    after = np.searchsorted(stamps, days, side="left")
    before = after - 1

    to_next = np.full(len(days), SCHEDULE_HORIZON, dtype="float64")
    valid = after < len(stamps)
    to_next[valid] = (stamps[after[valid]] - days[valid]).astype("float64")

    since_last = np.full(len(days), DRIFT_WINDOW, dtype="float64")
    valid = before >= 0
    since_last[valid] = (days[valid] - stamps[before[valid]]).astype("float64")

    # Clipped, then scaled to [0, 1] so these sit in the same range as
    # everything else the model is given.
    to_next = np.clip(to_next, 0.0, SCHEDULE_HORIZON) / SCHEDULE_HORIZON
    drift = np.clip(since_last, 0.0, DRIFT_WINDOW)

    out["days_to_earnings"] = to_next
    out["days_since_earnings"] = drift / DRIFT_WINDOW
    # A flag as well as a distance: the drift effect is strongest immediately
    # after and decays, and a linear distance cannot express "just reported".
    out["in_earnings_drift"] = (drift < DRIFT_WINDOW).astype("float64")

    return out


def coverage(symbols, *, refresh: bool = False) -> dict:
    """Which symbols have a usable calendar and which do not.

    Recorded per run. A panel where half the names have no earnings data is
    still worth training on; a panel where that is true and nobody noticed is
    not, and the difference is only this dictionary.

    Fetched a few at a time on the first run, because a few hundred calendars
    one after another is the slowest thing this project does and it does it
    without saying anything. Cached symbols never reach the pool.
    """
    symbols = list(symbols)
    covered, missing = [], []

    def one(symbol: str):
        return symbol, has_coverage(symbol, refresh=refresh)

    with futures.ThreadPoolExecutor(max_workers=max(FETCH_WORKERS, 1)) as pool:
        pending = [pool.submit(one, symbol) for symbol in symbols]
        for done, future in enumerate(futures.as_completed(pending), start=1):
            try:
                symbol, ok = future.result()
                (covered if ok else missing).append(symbol)
            except Exception as exc:                    # noqa: BLE001
                logger.warning("Earnings coverage check failed: %s", exc)

            if len(symbols) > 20 and done % 25 == 0:
                logger.info("Earnings calendars: %d/%d", done, len(symbols))

    return {"covered": sorted(covered), "missing": sorted(missing),
            "share": round(len(covered) / max(len(covered) + len(missing), 1), 3)}
