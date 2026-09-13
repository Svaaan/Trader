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
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
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
            if _covers_period(cached, period):
                logger.debug("%s: %d rows from cache", symbol, len(cached))
                return _drop_unfinished_session(cached)
            logger.info(
                "Cache for %s covers %s..%s, which is short of %s; refetching",
                symbol, cached.index[0].date(), cached.index[-1].date(), period)
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
