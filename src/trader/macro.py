"""The state of the world, as a handful of numbers that move every day.

This is the answer to "can we add economic news", and it is deliberately not
news. Scraping CPI headlines gives about twenty observations across a two-year
test set -- twelve prints a year, identical across every symbol on the day they
land. Nothing can be learned from twenty points and nothing can be graded on
them either; the trust gate in explain.py refuses to look at fewer than a few
hundred. A macro *event* feature is the smallest dataset in the project wearing
the costume of the largest.

The same information arrives continuously through prices. The curve steepens
before the print and after it; the dollar moves every session; credit widens
while the story is still forming. Those are 2,500 observations rather than
twenty, they are the mechanism by which macro actually reaches equity prices,
and they cost one HTTP request each.

They have a second advantage that matters more than it sounds. **Market data is
never revised.** GDP for a quarter is restated twice, employment is restated
every month, and FRED serves you the current revision -- so a model trained on
"what GDP was in March 2024" is trained on a number that did not exist in March
2024. Getting that right needs vintage data (ALFRED, not FRED) and getting it
wrong is invisible. A yield close is a yield close forever.

**Everything here is lagged one session, and that is not conservatism.** The
panel holds US and European names. The S&P closes at 16:00 New York; Nestle
closes at 17:30 Zurich, which is 11:30 New York. Today's US macro close is
*five and a half hours after* the European bar it would be attached to. Even
within the US it is wrong: the VIX settles at 16:15 ET, fifteen minutes after
the equity close. So today's macro is tomorrow's input, uniformly, for every
symbol. It costs a day of freshness and removes an entire class of leak that
would otherwise be invisible and would flatter every European name in the panel.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import prices as prices_mod

logger = logging.getLogger(__name__)

# The series, and why each one is here. Every one is a traded instrument or an
# index derived from traded instruments, so every one is unrevised.
SERIES = {
    "vix": "^VIX",              # equity volatility, the fear gauge
    "vix3m": "^VIX3M",          # three-month vol, for the term structure
    "y10": "^TNX",              # ten-year Treasury yield
    "y3m": "^IRX",              # thirteen-week bill, the short end
    "dollar": "DX-Y.NYB",       # the dollar index; matters most for the EU names
    "hy": "HYG",                # high-yield credit
    "ig": "LQD",                # investment-grade credit
    "oil": "CL=F",              # crude; XOM is in the panel and oil is macro anyway
    "gold": "GC=F",             # the other safe haven
    "spx": "^GSPC",             # the US market
    "stoxx": "^STOXX50E",       # the European market
}

MACRO_NAMES = [
    "vix_log",
    "vix_change_5d",
    "vix_term_slope",
    "y10_level",
    "y10_change_20d",
    "curve_slope",
    "curve_change_20d",
    "dollar_return_20d",
    "credit_ratio_change_20d",
    "oil_return_20d",
    "gold_return_20d",
    "spx_return_5d",
    "spx_drawdown_252",
    "spx_vol_20d",
    "stoxx_return_5d",
    "stoxx_drawdown_252",
]

# How many sessions to hold macro back. See the module docstring: one is enough
# to put every series strictly before every close in the panel, and it is the
# same trade prices.py makes when it drops an unfinished bar.
LAG_SESSIONS = 1

# How many consecutive sessions a level may be carried across on the shared
# calendar. Long enough for a holiday one market keeps and another does not;
# measured on the cached ten-year histories, the longest such gap in any series
# here is two sessions (STOXX over Christmas 2018), so five never touches
# history. Short enough that a series the provider has stopped publishing --
# ^VIX3M, frozen since 2026-07-17 -- is not quietly read as unchanged for months.
MAX_FILL_SESSIONS = 5

_WARNED: set = set()


def _warn_once(key, message: str, *args) -> None:
    """A panel is assembled many times per search; say each thing once."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)

MAX_WARMUP = 252


def _closes(period: str = "10y", refresh: bool = False) -> pd.DataFrame:
    """Closing level of every series, on one index, not yet filled.

    A missing series is dropped with a warning rather than failing the run: a
    macro block that is one column short is worth more than no run. The gaps
    the shared calendar creates are left for `_bound` to judge.
    """
    frames = {}
    for name, ticker in SERIES.items():
        try:
            frames[name] = prices_mod.load(ticker, period=period,
                                           refresh=refresh)["close"]
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Macro series %s (%s) unavailable: %s", name, ticker, exc)

    if not frames:
        raise prices_mod.PriceError("no macro series could be fetched")

    return pd.DataFrame(frames).sort_index()


def _bound(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Forward-fill across a holiday, and never across an ending.

    A holiday means the level did not change because nothing traded, so
    carrying yesterday forward states exactly what was known -- for a day or
    two. Past MAX_FILL_SESSIONS it is no longer a holiday, and there are two
    different things it can be:

    **An ending.** The series stops and the others carry on. The whole series
    is dropped, and every feature built from it goes with it. Leaving it blank
    instead would be worse than it sounds: `build` drops any date with a
    missing column and the panel drops any row whose macro join is empty, so a
    blank ending deletes the most recent sessions -- including today -- for
    every symbol. Measured on the wide panel, that would have blanked 34
    sessions since ^VIX3M stopped and deleted 7,777 complete rows; dropping the
    series instead keeps all 536,221, and costs one column.

    **An outage.** The series resumes. The sessions past the limit are left
    blank and counted, because they were not known; those dates drop out of the
    panel, and so does anything a rolling window spans across them. That is the
    honest cost and it is reported rather than hidden. None of the series here
    has had one in ten years.

    Returns the filled closes and {"ended": {...}, "gaps": {...}} for the report.
    An ending is judged against the other series rather than the clock, so a
    block that is a weekend behind has not ended.
    """
    filled = raw.ffill(limit=MAX_FILL_SESSIONS)
    ended, gaps = {}, {}

    for name in list(raw.columns):
        last = raw[name].last_valid_index()
        if last is None:
            filled = filled.drop(columns=name)
            continue

        behind = int((raw.index > last).sum())
        if behind > MAX_FILL_SESSIONS:
            ended[name] = {"ticker": SERIES.get(name, name),
                           "last": str(last.date()),
                           "sessions_behind": behind}
            filled = filled.drop(columns=name)
            _warn_once(("ended", name, str(last.date())),
                       "Macro series %s (%s) stopped on %s, %d sessions before "
                       "the others; dropping it and the features built from it "
                       "rather than carrying its last close forward",
                       name, SERIES.get(name, name), last.date(), behind)
            continue

        first = raw[name].first_valid_index()
        blank = int(filled.loc[first:last, name].isna().sum())
        if blank:
            gaps[name] = blank
            _warn_once(("gap", name, blank),
                       "Macro series %s (%s) has %d session(s) left blank inside "
                       "its history, past the %d a holiday can explain; those "
                       "dates, and rolling windows across them, drop out",
                       name, SERIES.get(name, name), blank, MAX_FILL_SESSIONS)

    return filled, {"ended": ended, "gaps": gaps}


def build(period: str = "10y", refresh: bool = False,
          report: dict | None = None) -> pd.DataFrame:
    """Macro features by date, already lagged, ready to join on.

    The returned frame is indexed by the date the features may be *used* on,
    not the date they were measured on. That shift happens once, here, at the
    end -- so no caller can forget it and no caller has to remember it.

    Pass `report` to have it filled with the series that have ended and the
    blanks left inside the others -- see `_bound`.

    Every pct_change here passes fill_method=None. Its default pads, which on
    a series that has stopped reports a return of exactly zero every day after
    -- the same unbounded fill `_bound` removes, one step later.
    """
    raw, notes = _bound(_closes(period=period, refresh=refresh))
    if report is not None:
        report.update(notes)
    out = pd.DataFrame(index=raw.index)

    def series(name: str) -> pd.Series | None:
        return raw[name] if name in raw.columns else None

    vix, vix3m = series("vix"), series("vix3m")
    if vix is not None:
        # Log, because volatility is multiplicative: 40 from 20 is the same
        # event as 20 from 10, and a level in points says otherwise.
        out["vix_log"] = np.log(vix.clip(lower=1e-6))
        out["vix_change_5d"] = np.log(vix.clip(lower=1e-6)).diff(5)
        if vix3m is not None:
            # Three-month vol below one-month vol means the market is paying up
            # for protection right now rather than later. That inversion is one
            # of the cleaner stress signals there is, and it is a ratio of two
            # numbers both known at the same instant.
            out["vix_term_slope"] = vix3m / vix.replace(0.0, np.nan) - 1.0

    y10, y3m = series("y10"), series("y3m")
    if y10 is not None:
        out["y10_level"] = y10 / 100.0
        out["y10_change_20d"] = (y10 - y10.shift(20)) / 100.0
        if y3m is not None:
            # The slope of the curve, in percentage points. Inversion here has
            # preceded most of the recessions anyone has data for, which is not
            # a reason to trust it and is a reason to give the model the option.
            out["curve_slope"] = (y10 - y3m) / 100.0
            out["curve_change_20d"] = ((y10 - y3m) - (y10 - y3m).shift(20)) / 100.0

    dollar = series("dollar")
    if dollar is not None:
        out["dollar_return_20d"] = dollar.pct_change(20, fill_method=None)

    hy, ig = series("hy"), series("ig")
    if hy is not None and ig is not None:
        # High yield against investment grade. Both are bond funds of similar
        # duration, so the ratio strips out the rate move and leaves the credit
        # move -- risk appetite, more or less directly.
        ratio = hy / ig.replace(0.0, np.nan)
        out["credit_ratio_change_20d"] = ratio.pct_change(20, fill_method=None)

    oil, gold = series("oil"), series("gold")
    if oil is not None:
        out["oil_return_20d"] = oil.pct_change(20, fill_method=None)
    if gold is not None:
        out["gold_return_20d"] = gold.pct_change(20, fill_method=None)

    for tag, name in (("spx", "spx"), ("stoxx", "stoxx")):
        index = series(name)
        if index is None:
            continue
        out[f"{tag}_return_5d"] = index.pct_change(5, fill_method=None)
        # How far the market sits below its own year's high. This is the single
        # most useful regime variable in the block: nearly every relationship
        # between features and returns behaves differently in a drawdown, and
        # without it the model has to infer the regime from the names it holds.
        out[f"{tag}_drawdown_252"] = index / index.rolling(252).max() - 1.0
        if tag == "spx":
            out["spx_vol_20d"] = index.pct_change(fill_method=None).rolling(20).std()

    out = out.replace([np.inf, -np.inf], np.nan)

    # The shift that makes all of this legal. See the module docstring.
    out = out.shift(LAG_SESSIONS)

    # Keep the declared order, skipping any column a missing series cost us.
    present = [name for name in MACRO_NAMES if name in out.columns]
    missing = [name for name in MACRO_NAMES if name not in out.columns]
    if missing:
        _warn_once(("missing", tuple(missing)), "Macro block is missing %s", missing)

    return out[present].dropna()


def names(frame: pd.DataFrame) -> list[str]:
    """Whatever `build` actually produced, in declared order."""
    return [name for name in MACRO_NAMES if name in frame.columns]
