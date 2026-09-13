"""Where this symbol sits among its peers today, rather than what it did alone.

The single largest thing moving any equity on any day is the market. Two thirds
or more of a typical large cap's daily variance is the market's variance wearing
that company's name. So a model asked "will this go up tomorrow" spends almost
all of its capacity on a question about the index, gets the same answer for
every symbol on a given day, and -- because the index rises about 52% of the
time -- discovers that answering "up" always is a local minimum it cannot climb
out of. That is exactly the collapse this project kept measuring: up-rate 0.98,
accuracy equal to the class balance, every feature influence near zero.

Ranking fixes the question rather than the model. "Is this name in the top half
of its peers today" has an answer that is 50/50 by construction, cannot be
guessed by a constant, and is the part of the return the company is actually
responsible for. It is also what a long/short book is made of.

Two families come out of here:

**Ranks.** Each symbol's position among the panel on that date, as a percentile.
Relative momentum is the oldest factor in the literature; relative volatility
and relative attention are how a name's own history gets normalised against a
market that was calm in 2017 and is not now.

**Panel state.** Breadth and dispersion -- how many names went up, and how far
apart they moved. These are macro features derived from the panel itself, and
they say something the index level cannot: a 1% index day where everything rose
together is a different market from a 1% day where half the names fell.

**Lagged one session, for the same reason macro is.** The panel holds US and
European names. Ranking Nestle against Apple on the same calendar date compares
a 17:30 Zurich close with a 16:00 New York close five and a half hours later,
so the rank a European name gets is built partly from its own future. One
session of lag removes that. A finer version would group the panel by trading
session and rank within each; that is worth doing when the universe is wide
enough for the groups to be big, and it is noted in the README as such.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Which per-symbol features are worth asking a relative question about. Not all
# of them: ranking eighteen features would double the input width to say the
# same things twice, and these six are the ones where "compared to what" changes
# the meaning rather than just the units.
RANKED = [
    "return_1d",          # relative short-term reversal
    "return_20d",         # relative medium momentum
    "momentum_12_1",      # the classic cross-sectional momentum factor
    "volatility_20d",     # relative risk
    "rsi_14",             # relative positioning
    "volume_vs_avg20",    # relative attention
]

CROSS_NAMES = [f"rank_{name}" for name in RANKED] + [
    "panel_breadth",
    "panel_dispersion",
    "relative_return_1d",
]

# See the module docstring. Same trade as macro.LAG_SESSIONS, same reason.
LAG_SESSIONS = 1

# Below this many symbols a percentile is not a percentile, it is an opinion
# about three numbers. Panels smaller than this get no cross-sectional block.
MIN_SYMBOLS = 5


def usable(frames: dict) -> bool:
    """Whether this panel is wide enough for a rank to mean anything."""
    return len(frames) >= MIN_SYMBOLS


def build(feature_frames: dict) -> dict:
    """Cross-sectional features for every symbol, keyed the same way.

    `feature_frames` maps symbol to that symbol's per-symbol feature frame, as
    features.build returns. The result maps symbol to a frame with CROSS_NAMES
    as columns, on the same dates, already lagged.

    Returns empty frames when the panel is too narrow, so a caller can join
    unconditionally and get nothing rather than get nonsense.
    """
    if not usable(feature_frames):
        logger.info("Panel of %d is below %d symbols; no cross-sectional block",
                    len(feature_frames), MIN_SYMBOLS)
        return {symbol: pd.DataFrame(index=frame.index)
                for symbol, frame in feature_frames.items()}

    symbols = sorted(feature_frames)

    # One date x symbol panel per ranked feature, then a percentile across the
    # row. `pct=True` handles days when a symbol is missing -- the rank is taken
    # among whoever actually traded, which is the honest comparison.
    ranks: dict[str, pd.DataFrame] = {}
    for name in RANKED:
        panel = pd.DataFrame(
            {symbol: feature_frames[symbol][name] for symbol in symbols
             if name in feature_frames[symbol].columns})
        if panel.empty:
            continue
        # Centred on zero so "middle of the pack" is the same neutral value the
        # scaler produces for everything else.
        ranks[f"rank_{name}"] = panel.rank(axis=1, pct=True) - 0.5

    returns = pd.DataFrame(
        {symbol: feature_frames[symbol]["return_1d"] for symbol in symbols
         if "return_1d" in feature_frames[symbol].columns})

    # How many names rose, and how far apart they moved. Both are single series
    # broadcast to every symbol -- panel-level state, not per-symbol.
    breadth = (returns > 0).sum(axis=1) / returns.notna().sum(axis=1) - 0.5
    dispersion = returns.std(axis=1)
    # This name's move against the panel's, which is the return the company is
    # responsible for rather than the one the market handed it.
    relative = returns.sub(returns.mean(axis=1), axis=0)

    out = {}
    for symbol in symbols:
        index = feature_frames[symbol].index
        frame = pd.DataFrame(index=index)

        for name, panel in ranks.items():
            if symbol in panel.columns:
                frame[name] = panel[symbol]

        frame["panel_breadth"] = breadth
        frame["panel_dispersion"] = dispersion
        if symbol in relative.columns:
            frame["relative_return_1d"] = relative[symbol]

        frame = frame.replace([np.inf, -np.inf], np.nan)

        # The lag that makes a mixed-session panel legal.
        frame = frame.shift(LAG_SESSIONS)

        out[symbol] = frame[[n for n in CROSS_NAMES if n in frame.columns]]

    return out


def names(frames: dict) -> list[str]:
    """Whatever `build` actually produced, in declared order."""
    for frame in frames.values():
        if len(frame.columns):
            return [name for name in CROSS_NAMES if name in frame.columns]
    return []
