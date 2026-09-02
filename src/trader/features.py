"""Turning a price history into features, without ever looking forward.

Every column here is computable at the close of the day it sits on. That is the
whole discipline of this file, and it is easy to break by accident: pandas will
happily centre a rolling window, or shift a series the wrong way, and the result
is a model that scores beautifully and cannot work, because it was shown the
answer.

The specific rules:

  * rolling windows are trailing, never centred
  * nothing is shifted backwards (`shift(-n)`) -- that is the future
  * today's high, low and volume are allowed, because they are known at the
    close of today; tomorrow's are not
  * the label lives in labels.py, not here, so the two cannot be confused

tests/test_no_lookahead.py checks the property directly rather than trusting
the reading: it computes features on a truncated history and on the full one,
and every row they share has to match. A feature that peeks cannot pass that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered, because a model trained on columns in one order and asked to predict
# with them in another produces confident nonsense. The manifest that goes to
# HelloWorldAi carries these names so the pairing survives the round trip.
FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "close_vs_sma20",
    "sma20_vs_sma50",
    "volume_vs_avg20",
    "range_vs_close",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index, Wilder's smoothing, trailing only."""
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)

    # ewm with adjust=False is the recursive form Wilder described, and it uses
    # only past values at every point.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # A stretch with no losses at all is genuinely "as strong as it gets".
    return rsi.fillna(100.0).where(avg_loss.notna() | avg_gain.isna())


def build(prices: pd.DataFrame) -> pd.DataFrame:
    """Features for every day there is enough history to compute them.

    Returns a frame indexed by date with exactly FEATURE_NAMES as columns. Rows
    where any window is still filling are dropped, rather than filled with a
    guess: a zero in place of a missing 50-day average is a lie the model has no
    way to recognise.
    """
    close = prices["close"]
    out = pd.DataFrame(index=prices.index)

    # Returns over a few horizons. pct_change looks back by construction.
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)

    # How violent the recent past has been.
    out["volatility_20d"] = close.pct_change().rolling(20).std()

    # Scaled to roughly [-0.5, 0.5] so it sits in the same range as the rest;
    # a feature two orders of magnitude larger than its neighbours dominates
    # the first layer for no reason other than its units.
    out["rsi_14"] = (_rsi(close, 14) - 50.0) / 100.0

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["close_vs_sma20"] = close / sma20 - 1.0
    out["sma20_vs_sma50"] = sma20 / sma50 - 1.0

    volume = prices["volume"]
    avg_volume = volume.rolling(20).mean()
    # Log, because volume spikes are multiplicative: a day at ten times average
    # and a day at a tenth should be the same distance from normal.
    out["volume_vs_avg20"] = np.log(
        (volume / avg_volume.replace(0.0, np.nan)).clip(lower=0.01))

    # Today's trading range, known at today's close.
    out["range_vs_close"] = (prices["high"] - prices["low"]) / close

    out = out[FEATURE_NAMES]

    # Infinities come from a zero denominator on a stale or broken series.
    out = out.replace([np.inf, -np.inf], np.nan)

    return out.dropna()
