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
"""

from __future__ import annotations

import datetime as dt
import logging
import os
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
            logger.debug("%s: %d rows from cache", symbol, len(cached))
            return _drop_unfinished_session(cached)
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


def load_many(symbols: Iterable[str], *, period: str = "10y",
              refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Prices for several symbols, skipping the ones that fail.

    One delisted ticker in a watchlist should not stop the run; it should be
    reported and left out, because a silent gap in a universe is the kind of
    thing that turns into a mystery three steps later.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            out[symbol] = load(symbol, period=period, refresh=refresh)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("Skipping %s: %s", symbol, exc)
    return out
