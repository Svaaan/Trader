"""Daily price history, fetched once and kept on disk.

Every other part of this project reads prices through here, so there is exactly
one place that knows where the numbers come from and exactly one cache to clear
when they look wrong.

Two things this module is careful about, because both are ways a trading
project quietly lies to itself:

**Adjusted prices.** A stock that splits four-for-one drops 75% overnight in raw
prices, and a model trained on that learns that enormous crashes are routine and
recoverable. auto_adjust divides the history back through splits and dividends,
so a price series reflects what a holder actually experienced.

**Today is not a finished day.** The last row of an intraday fetch is a partial
bar that keeps changing until the close. Training on it means training on a
number that was not knowable, and a signal generated from it is generated from
the future. Rows are dropped unless the session they describe has ended.

**A cache is only valid for the period it was fetched for.** This one drew
blood. A short fetch of AAPL was cached early on, and every run afterwards asked
for ten years, got two, and never noticed -- the symbol quietly fell out of the
panel because it had no rows before the cut date, and nine names trained where
ten were reported. So the cache records what it covers and is refetched when it
covers less than it is asked for.

**And a cache is only valid until the next session closes.** Checking the span
is not enough: a ten-year history ending twelve days ago still spans ten years.
Measured -- the panel was fetched once, and a fortnight later the page was
still generating "today's signal" from bars dated 2026-09-01, labelled honestly
as `as_of` and read by nobody. For a daily signal that is the whole product
quietly two weeks out of date, so a cache whose last bar is older than a few
days is refetched whatever its span.

**But asking again does not make the provider know more.** Both rules above
assume a refetch can fix what they find, and for some symbols it cannot. ^VIX3M
stopped updating at the provider on 2026-07-17, and DSFIR.AS listed in 2023, so
neither will ever satisfy a ten-year, up-to-date check. Every load refetched
them, got back exactly what was cached, and refetched again on the next panel
assembly -- nine times in one search. So every fetch is now recorded, and a
cache that fails the checks but came from the provider within the last
RECHECK_HOURS is used as it is, with a warning naming what the provider
actually has.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
from concurrent import futures
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# yfinance returns these; the rest of the project assumes exactly these.
COLUMNS = ["open", "high", "low", "close", "volume"]

CACHE_DIR = os.environ.get(
    "TRADER_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "prices"),
)


class PriceError(Exception):
    """Raised when a symbol cannot be turned into a usable price history."""


# How much slack to allow before calling a cache short. Exchanges have holidays,
# listings begin mid-window, and a stock that IPO'd four years ago can never
# satisfy a ten-year request however many times it is refetched.
CACHE_TOLERANCE_DAYS = 45

# And how stale the *end* of a cache may be before it is refetched. Four days
# covers a weekend plus a public holiday; beyond that the series has genuinely
# missed sessions. A delisted symbol will be refetched every run under this
# rule and fail every time, which is the correct outcome -- it is not a usable
# price history and the run should keep saying so.
MAX_CACHE_AGE_DAYS = 4


# How long a provider's answer stands. A cache that fails the checks above but
# was fetched this recently is what the provider has: asking again within the
# hour returns the same rows. Long enough that one search does not ask nine
# times; short enough that a series which resumes updating is picked up the
# same day.
RECHECK_HOURS = 12

# Every fetch, by symbol: when, for what period, and the first and last dates it
# returned. One small file beside the CSVs rather than a file per symbol.
FETCH_LOG = "_fetched.json"

_LOG_LOCK = threading.Lock()
_WARNED: set = set()


def _period_days(period: str) -> float | None:
    """Roughly how many calendar days `period` asks for, or None if open-ended.

    Only needs to be close. It exists to tell two years from ten, not to be a
    calendar.
    """
    text = str(period).strip().lower()
    if text in ("max", "ytd", ""):
        return None

    match = re.fullmatch(r"(\d+)\s*(d|wk|mo|y)", text)
    if not match:
        return None

    count = int(match.group(1))
    return count * {"d": 1.0, "wk": 7.0, "mo": 30.44, "y": 365.25}[match.group(2)]


def _is_current(frame: pd.DataFrame, today: dt.date | None = None) -> bool:
    """Does this cache reach the present, or did it stop a fortnight ago?

    Separate from `_covers_period` because they fail differently and only one
    of them was being checked. A history can span exactly what was asked for and
    still end before the last session anybody traded.
    """
    if frame.empty:
        return False

    today = today or dt.datetime.now(dt.timezone.utc).date()
    return (today - frame.index[-1].date()).days <= MAX_CACHE_AGE_DAYS


def _covers_period(frame: pd.DataFrame, period: str) -> bool:
    """Does this cached frame actually span what was asked for?

    A frame that starts later than requested is either a short fetch that got
    cached (refetch it) or a symbol that did not exist yet (refetching will not
    help, and the caller finds out either way). Answering "no" costs one HTTP
    request; answering "yes" wrongly costs a symbol, silently.
    """
    wanted = _period_days(period)
    if wanted is None or frame.empty:
        return True

    span = (frame.index[-1] - frame.index[0]).days
    return span >= wanted - CACHE_TOLERANCE_DAYS


def _cache_path(symbol: str) -> str:
    # CSV rather than parquet: it needs no extra engine, and a cache you can
    # open and read is worth more here than one that loads a few milliseconds
    # faster. When a feature looks wrong the first question is always what the
    # prices actually were.
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(os.path.abspath(CACHE_DIR), f"{safe}.csv")


def _fetch_log_path() -> str:
    return os.path.join(os.path.abspath(CACHE_DIR), FETCH_LOG)


def _read_fetch_log() -> dict:
    try:
        with open(_fetch_log_path(), encoding="utf-8") as handle:
            log = json.load(handle)
        return log if isinstance(log, dict) else {}
    except (OSError, ValueError):
        return {}


def _record_fetch(symbol: str, period: str, frame: pd.DataFrame,
                  now: dt.datetime | None = None) -> None:
    """Note what the provider returned, and when.

    Written whole to a temporary file and swapped in, under a lock, because
    load_many fetches on several threads at once. Two processes can still race
    and lose an entry; that costs one redundant fetch, never a wrong answer.
    And a log that cannot be written must not stop prices loading -- it only
    means the next load asks again.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    entry = {"at": now.isoformat(timespec="seconds"), "period": str(period),
             "first": str(frame.index[0].date()) if len(frame) else None,
             "last": str(frame.index[-1].date()) if len(frame) else None}
    path = _fetch_log_path()
    with _LOG_LOCK:
        try:
            log = _read_fetch_log()
            log[symbol] = entry
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(log, handle, indent=1, sort_keys=True)
            os.replace(temporary, path)
        except OSError as exc:
            logger.debug("Could not record the fetch of %s: %s", symbol, exc)


def _vouched_for(symbol: str, cached: pd.DataFrame, period: str,
                 now: dt.datetime | None = None) -> dict | None:
    """The fetch that produced this cache, if it was recent enough to stand.

    Three conditions, each closing a way to trust a cache wrongly: the fetch was
    within RECHECK_HOURS; it asked for at least this much history, so a 2y fetch
    cannot vouch for a 10y request; and its first and last dates are the ones in
    the file, so a CSV replaced by hand since is not taken on the log's word.
    """
    entry = _read_fetch_log().get(symbol)
    if not isinstance(entry, dict) or cached.empty:
        return None

    try:
        at = dt.datetime.fromisoformat(entry["at"])
    except (KeyError, TypeError, ValueError):
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    if now - at > dt.timedelta(hours=RECHECK_HOURS):
        return None

    asked, wanted = _period_days(entry.get("period", "")), _period_days(period)
    if wanted is None:
        if asked is not None:
            return None
    elif asked is not None and asked < wanted:
        return None

    if (entry.get("first") != str(cached.index[0].date())
            or entry.get("last") != str(cached.index[-1].date())):
        return None
    return entry


def _normalise(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One shape, whatever yfinance felt like returning."""
    if frame is None or frame.empty:
        raise PriceError(f"No price history came back for {symbol}.")

    # A single-symbol download sometimes arrives with a MultiIndex anyway.
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(1, axis=1)

    frame = frame.rename(columns=str.lower)

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise PriceError(f"{symbol} is missing {missing}; got {list(frame.columns)}")

    frame = frame[COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"

    # Rows with no close are holidays and half-days that leaked in.
    frame = frame[frame["close"].notna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    return _drop_unfinished_session(frame)


def _drop_unfinished_session(frame: pd.DataFrame,
                             today: dt.date | None = None) -> pd.DataFrame:
    """Remove a bar for a session that has not closed yet.

    yfinance happily returns today's partial bar mid-session. Its close is
    whatever the price happens to be at the moment of the request, and it will
    be a different number in an hour. A model trained on it has been shown a
    figure nobody could have known; a signal generated from it was generated
    from information that did not exist at the time it claims to apply to.

    Being a day conservative costs one row. Being a day optimistic invalidates
    every backtest built on top.
    """
    if frame.empty:
        return frame

    today = today or dt.datetime.now(dt.timezone.utc).date()
    last = frame.index[-1].date()

    if last >= today:
        logger.info("Dropping today's unfinished bar (%s)", last)
        return frame.iloc[:-1]
    return frame


def load(symbol: str, *, period: str = "10y", refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV for one symbol, split- and dividend-adjusted.

    Cached to disk so that iterating on features does not re-download, and so
    that a run is reproducible from one hour to the next.
    """
    path = _cache_path(symbol)

    if not refresh and os.path.exists(path):
        try:
            cached = pd.read_csv(path, index_col="date", parse_dates=["date"])
            short = not _covers_period(cached, period)
            stale = not short and not _is_current(cached)
            if not (short or stale):
                logger.debug("%s: %d rows from cache", symbol, len(cached))
                return _drop_unfinished_session(cached)

            vouched = _vouched_for(symbol, cached, period)
            if vouched is not None:
                if symbol not in _WARNED:
                    _WARNED.add(symbol)
                    # Judged separately: a frozen end also shortens the span,
                    # so "short" alone would hide the problem that matters.
                    ended = not _is_current(cached)
                    problems = ([] if _covers_period(cached, period)
                                else [f"short of {period}"])
                    if ended:
                        age = (dt.datetime.now(dt.timezone.utc).date()
                               - cached.index[-1].date()).days
                        problems.append(f"{age} days out of date")
                    logger.warning(
                        "%s: the provider only has %s..%s (asked at %s), which "
                        "is %s; using it as it is and not asking again for %d "
                        "hours.%s",
                        symbol, vouched["first"], vouched["last"], vouched["at"],
                        " and ".join(problems), RECHECK_HOURS,
                        (f" Anything that aligns this series to a calendar and "
                         f"forward-fills it will carry {vouched['last']} past "
                         f"that date." if ended else ""))
                return _drop_unfinished_session(cached)

            if short:
                logger.info(
                    "Cache for %s covers %s..%s, which is short of %s; refetching",
                    symbol, cached.index[0].date(), cached.index[-1].date(),
                    period)
            else:
                logger.info(
                    "Cache for %s ends %s, which is stale; refetching",
                    symbol, cached.index[-1].date())
        except Exception as exc:                       # noqa: BLE001
            logger.warning("Cache for %s unreadable (%s); refetching", symbol, exc)

    import yfinance                                    # imported late: it is slow

    logger.info("Fetching %s (%s)", symbol, period)
    raw = yfinance.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=True,       # see the module docstring
        actions=False,
    )

    frame = _normalise(raw, symbol)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path)
    _record_fetch(symbol, period, frame)

    return frame


# How many symbols to fetch at once. These are I/O-bound waits on somebody
# else's server, so a few in flight is most of the win; a large pool mostly
# buys rate limiting. Cached symbols never reach the pool at all.
FETCH_WORKERS = int(os.environ.get("TRADER_FETCH_WORKERS", "6"))


def load_many(symbols: Iterable[str], *, period: str = "10y",
              refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Prices for several symbols, skipping the ones that fail.

    One delisted ticker in a watchlist should not stop the run; it should be
    reported and left out, because a silent gap in a universe is the kind of
    thing that turns into a mystery three steps later.

    Fetched a few at a time and with progress logged. A few hundred symbols
    one at a time is several silent minutes on the first run, which is
    indistinguishable from being hung.
    """
    symbols = list(symbols)
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    def one(symbol: str):
        return symbol, load(symbol, period=period, refresh=refresh)

    with futures.ThreadPoolExecutor(max_workers=max(FETCH_WORKERS, 1)) as pool:
        pending = [pool.submit(one, symbol) for symbol in symbols]
        for done, future in enumerate(futures.as_completed(pending), start=1):
            try:
                symbol, frame = future.result()
                out[symbol] = frame
            except Exception as exc:                   # noqa: BLE001
                failed.append(str(exc))
                logger.warning("Skipping a symbol: %s", exc)

            if len(symbols) > 20 and done % 25 == 0:
                logger.info("Prices: %d/%d", done, len(symbols))

    missing = sorted(set(symbols) - set(out))
    if missing:
        logger.warning("No price history for %d symbol(s): %s",
                       len(missing), missing)

    return out
