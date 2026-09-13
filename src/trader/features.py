"""Turning one symbol's price history into features, without ever looking forward.

Every column here is computable at the close of the day it sits on, from that
symbol's own OHLCV and nothing else. That is the whole discipline of this file,
and it is easy to break by accident: pandas will happily centre a rolling
window, or shift a series the wrong way, and the result is a model that scores
beautifully and cannot work, because it was shown the answer.

The specific rules:

  * rolling windows are trailing, never centred
  * nothing is shifted backwards (`shift(-n)`) -- that is the future
  * today's high, low and volume are allowed, because they are known at the
    close of today; tomorrow's are not
  * the label lives in labels.py, not here, so the two cannot be confused

Four other kinds of input live elsewhere, because none is derivable from one
symbol's prices and mixing them in here would blur the rule above:

  * cross.py    where this symbol sits among its peers today
  * macro.py    market-wide state, joined by date
  * events.py   how close the next scheduled announcement is
  * news.py     a point-in-time store, inert until it has history

tests/test_no_lookahead.py checks the property directly rather than trusting
the reading: it computes features on a truncated history and on the full one,
and every row they share has to match. A feature that peeks cannot pass that.

On the size of this list. Nine features became eighteen because the original
nine were nearly the same feature -- returns over three horizons, price against
two moving averages, all of them momentum wearing different hats. A model given
five correlated views of one thing has one input, not five. What was missing was
information of a genuinely different kind: where in the day's range it closed,
how much of the move happened overnight, how far it sits below its year's high,
what it would cost to trade. None of those is more momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered, because a model trained on columns in one order and asked to predict
# with them in another produces confident nonsense. The manifest that goes to
# HelloWorldAi carries these names so the pairing survives the round trip.
FEATURE_NAMES = [
    # --- momentum, over horizons that are actually different ---
    "return_1d",
    "return_5d",
    "return_20d",
    "momentum_12_1",
    # --- how violent, and whether that is changing ---
    "volatility_20d",
    "vol_ratio_5_20",
    "skew_20d",
    # --- position within the recent range ---
    "rsi_14",
    "close_vs_sma20",
    "sma20_vs_sma50",
    "close_vs_high_252",
    # --- participation ---
    "volume_vs_avg20",
    "volume_trend_5_20",
    "amihud_20",
    # --- the shape of today's session ---
    "range_vs_close",
    "close_in_range",
    "overnight_return",
    "intraday_return",
]

# The longest trailing window any feature uses. Rows before this are dropped,
# so it is also how much history a symbol needs before it is usable at all --
# worth knowing when a symbol goes missing from a panel.
MAX_WARMUP = 252


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index, Wilder's smoothing, trailing only.

    The warm-up has to come back NaN. An earlier version filled it with 100 --
    the value meaning "nothing but gains, as strong as it gets" -- because the
    guard meant to blank the warm-up tested `avg_gain.isna()`, which is true
    *during* the warm-up and so kept the fill instead of removing it. It went
    unnoticed because the 50-day moving average drops those rows anyway. A
    latent trap: shorten the window set and fourteen days of fictional maximum
    strength enter the front of every symbol's history.
    """
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)

    # ewm with adjust=False is the recursive form Wilder described, and it uses
    # only past values at every point.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # A stretch with no losses at all is genuinely "as strong as it gets" and
    # gets 100. A stretch that has not finished warming up gets nothing.
    warmed = avg_gain.notna() & avg_loss.notna()
    return rsi.where(avg_loss > 0.0, 100.0).where(warmed)


def build(prices: pd.DataFrame) -> pd.DataFrame:
    """Features for every day there is enough history to compute them.

    Returns a frame indexed by date with exactly FEATURE_NAMES as columns. Rows
    where any window is still filling are dropped, rather than filled with a
    guess: a zero in place of a missing 50-day average is a lie the model has no
    way to recognise.
    """
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    open_ = prices["open"]
    volume = prices["volume"]

    out = pd.DataFrame(index=prices.index)
    daily = close.pct_change()

    # --- momentum -----------------------------------------------------------
    # pct_change looks back by construction.
    out["return_1d"] = daily
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)

    # Twelve months of return, skipping the most recent one. The skip is the
    # point: the last month tends to reverse while the eleven before it tend to
    # persist, and blending them cancels both. This is the oldest surviving
    # cross-sectional anomaly in equities, and the original list had nothing
    # like it -- its longest horizon was twenty days.
    out["momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0

    # --- volatility ---------------------------------------------------------
    vol20 = daily.rolling(20).std()
    out["volatility_20d"] = vol20

    # Whether the recent past is calmer or wilder than the near past. A level
    # says what the regime is; a ratio says it is changing, which is the part
    # that tends to come before something.
    out["vol_ratio_5_20"] = daily.rolling(5).std() / vol20.replace(0.0, np.nan)

    # Asymmetry. A name that grinds up and gaps down is a different animal from
    # one that does the reverse, and neither a standard deviation nor any return
    # horizon can tell them apart.
    out["skew_20d"] = daily.rolling(20).skew()

    # --- where it sits ------------------------------------------------------
    # Scaled to roughly [-0.5, 0.5] so it sits in the same range as the rest;
    # a feature two orders of magnitude larger than its neighbours dominates
    # the first layer for no reason other than its units.
    out["rsi_14"] = (_rsi(close, 14) - 50.0) / 100.0

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["close_vs_sma20"] = close / sma20 - 1.0
    out["sma20_vs_sma50"] = sma20 / sma50 - 1.0

    # Distance below the year's high. Nearness to a 52-week high is a real and
    # well-documented effect -- it is where resistance to further buying tends
    # to concentrate -- and it is not the question any moving average asks.
    out["close_vs_high_252"] = close / close.rolling(252).max() - 1.0

    # --- participation ------------------------------------------------------
    avg_volume = volume.rolling(20).mean()
    # Log, because volume spikes are multiplicative: a day at ten times average
    # and a day at a tenth should be the same distance from normal.
    relative_volume = (volume / avg_volume.replace(0.0, np.nan)).clip(lower=0.01)
    out["volume_vs_avg20"] = np.log(relative_volume)

    # A slower version of the same question: is interest building or fading over
    # weeks, rather than spiking today.
    out["volume_trend_5_20"] = np.log(
        (volume.rolling(5).mean() / avg_volume.replace(0.0, np.nan)).clip(lower=0.01))

    # Amihud illiquidity: how far the price moves per unit of money traded. It
    # is a priced risk factor in its own right, and it is the closest thing in
    # this file to "would it cost anything to act on this".
    turnover = (close * volume).replace(0.0, np.nan)
    out["amihud_20"] = np.log1p((daily.abs() / turnover).rolling(20).mean() * 1e9)

    # --- the shape of today's session ---------------------------------------
    # All known at today's close.
    span = (high - low).replace(0.0, np.nan)
    out["range_vs_close"] = (high - low) / close

    # Where in the day's range it finished. Closing on the high after a wide
    # range is a different event from closing on the low, and a return alone
    # cannot distinguish them.
    out["close_in_range"] = (close - low) / span - 0.5

    # Overnight and intraday returns behave differently -- one absorbs news
    # released while the market was shut, the other is the session itself --
    # and return_1d is their sum, which hides the split.
    out["overnight_return"] = open_ / close.shift(1) - 1.0
    out["intraday_return"] = close / open_.replace(0.0, np.nan) - 1.0

    out = out[FEATURE_NAMES]

    # Infinities come from a zero denominator on a stale or broken series.
    out = out.replace([np.inf, -np.inf], np.nan)

    return out.dropna()
