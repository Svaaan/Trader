"""What the model is asked to predict.

Kept apart from features.py on purpose. Features may only look backwards and
labels must look forwards, so they are the two halves of the one mistake that
matters here -- putting them in separate files means the shift that creates a
label can never be mistaken for a rolling window that creates a feature.

The label on day t describes what happened *after* day t. It is not knowable on
day t, which is the point: that is what there would be value in predicting.

There are two targets here, and choosing between them is the most consequential
decision in the project.

**Absolute direction** -- will this close higher tomorrow -- is the obvious one
and it is close to unanswerable. Roughly 52% of daily moves in a large-cap panel
are up, so a model that learns nothing and answers "up" every time scores 52%,
and gradient descent finds that constant long before it finds anything subtle.
Measured here, repeatedly: up-rate 0.98, accuracy equal to the class balance,
every feature influence under 0.01. The model was not failing to learn. It had
learned the only thing reliably there, which is the drift.

There is a second distinction here that matters as much and is easier to miss:
the return the label *describes* is not the return anybody can *hold*. The label
runs close to close, and a signal built from today's close cannot be acted on
until the next open. `forward_return` is the first; `executable_return` is the
second; every evaluation reports both because on this panel they disagree about
whether there is a strategy at all.

**Relative direction** -- will this name finish in the top half of its peers --
removes the drift by construction. The classes are 50/50 on every single date,
so no constant answer can score above chance, and the market factor that
dominates absolute returns cancels out of the target entirely. What is left is
the part the company is responsible for, which is the part the features describe
and the part a long/short book is paid for.

That does not conjure a signal that is not there. It makes the absence of one
legible instead of hiding it behind 52%, and it makes any signal that does exist
reachable instead of drowned. Both targets are kept, because the comparison
between them is itself informative and the honest reading of this project is
still that neither has produced an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN = 1, 0
CLASS_NAMES = ["down", "up"]        # index == label value

ABSOLUTE = "absolute"
RELATIVE = "relative"
TARGETS = (ABSOLUTE, RELATIVE)


# --- one symbol at a time --------------------------------------------------

def direction(prices: pd.DataFrame, *, horizon: int = 1,
              threshold: float = 0.0) -> pd.Series:
    """1 if the close rises over the next `horizon` sessions, else 0.

    `threshold` is a dead band: with 0.002, a move of less than 0.2% counts as
    down rather than up. Raising it makes the classes less balanced and the
    remaining "up" days more decisive, which is sometimes what you want and is
    never free -- there are fewer of them to learn from.

    The last `horizon` rows come back as NaN, because their future has not
    happened yet. Callers drop them. They are also, not coincidentally, the rows
    a live signal is generated for: today has features and no label, which is
    exactly the situation prediction exists for.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 session")

    close = prices["close"]

    # shift(-horizon) is the only forward-looking operation in this project.
    future = close.shift(-horizon)
    change = future / close - 1.0

    label = (change > threshold).astype("float64")
    label[change.isna()] = float("nan")

    return label.rename("label")


def forward_return(prices: pd.DataFrame, *, horizon: int = 1) -> pd.Series:
    """The actual return the label is derived from.

    Kept because evaluation needs the size of a move, not just its sign: a
    strategy that is right about small moves and wrong about large ones loses
    money while looking accurate.

    Close to close, which is what the label describes and **not** what anybody
    could hold -- see `executable_return`.
    """
    close = prices["close"]
    return (close.shift(-horizon) / close - 1.0).rename("forward_return")


def executable_return(prices: pd.DataFrame, *, horizon: int = 1) -> pd.Series:
    """The part of that move somebody could actually have captured.

    The features on day t are computed from day t's close. Nobody knows that
    close until the session has ended, so nobody can be positioned *at* it on
    the strength of it. The earliest a signal derived from today's close can be
    acted on is the next open.

    So this measures open(t+1) -> close(t+horizon): enter at the first price
    available after the signal exists, exit where the label does.

    The difference between this and `forward_return` is the overnight gap, and
    on this project's own panel the gap is not a rounding error -- it is the
    entire gross edge. Measured, per row, on the logistic control:

        close(t) -> close(t+1)    +2.96 bp   daily Sharpe +1.67
          the overnight gap       +3.84 bp   (more than all of it)
          open(t+1) -> close      -0.80 bp   daily Sharpe -0.40

    A backtest graded on the first line describes a trade nobody can place. The
    third line is the one that decides whether there is a strategy, which is why
    evaluate.py reports both and the gate reads this one.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 session")

    # shift(-1) on the open: the next session's opening price, known the instant
    # that session begins and not before.
    entry = prices["open"].shift(-1)
    exit_price = prices["close"].shift(-horizon)
    return (exit_price / entry - 1.0).rename("executable_return")


# --- the whole panel at once -----------------------------------------------
#
# A relative label cannot be computed one symbol at a time: "top half of its
# peers" needs the peers. These take the panel and hand back date x symbol
# frames that dataset.py splits up again.

def forward_return_panel(frames: dict, *, horizon: int = 1) -> pd.DataFrame:
    """Every symbol's forward return, on one date index."""
    return pd.DataFrame({
        symbol: forward_return(prices, horizon=horizon)
        for symbol, prices in frames.items()
    }).sort_index()


def executable_return_panel(frames: dict, *, horizon: int = 1) -> pd.DataFrame:
    """The same, for the window that could actually be held."""
    return pd.DataFrame({
        symbol: executable_return(prices, horizon=horizon)
        for symbol, prices in frames.items()
    }).sort_index()


def relative_forward_return(panel: pd.DataFrame) -> pd.DataFrame:
    """Each symbol's forward return with the panel's average taken out.

    This is what a market-neutral book actually earns: long the names expected
    to lead, short the ones expected to lag, and the index move cancels. It is
    also the return that pairs with the relative label, and evaluating a
    relative model against absolute returns would credit it for market drift it
    never predicted.
    """
    return panel.sub(panel.mean(axis=1), axis=0)


def relative_direction(panel: pd.DataFrame, *,
                       neutral_band: float = 0.0) -> pd.DataFrame:
    """1 if this symbol finishes in the top half of the panel, else 0.

    Ranked per date, so the classes are balanced on every date rather than on
    average -- which matters, because a target that is 50/50 overall but 70/30
    inside the test window hands back exactly the baseline problem this was
    meant to remove.

    `neutral_band` drops the middle of the cross-section: at 0.1, names ranking
    between 0.45 and 0.55 become NaN and are dropped. The days a name lands in
    the middle of its peers are the days there was nothing to know, and training
    on them teaches the model to reproduce noise. It costs rows, so it is off by
    default and worth trying rather than assuming.
    """
    if not 0.0 <= neutral_band < 1.0:
        raise ValueError("neutral_band must be in [0, 1)")

    # rank over columns: only same-date values, no time axis involved at all.
    ranked = panel.rank(axis=1, pct=True)

    # A date with too few names to rank is not a cross-section.
    ranked = ranked.where(panel.notna().sum(axis=1) >= 3)

    label = (ranked > 0.5).astype("float64")
    label[ranked.isna()] = np.nan

    if neutral_band > 0.0:
        low, high = 0.5 - neutral_band / 2.0, 0.5 + neutral_band / 2.0
        label[(ranked > low) & (ranked < high)] = np.nan

    return label


def build_panel_labels(frames: dict, *, horizon: int = 1, target: str = RELATIVE,
                       threshold: float = 0.0, neutral_band: float = 0.0
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(labels, returns-to-grade-on, returns-that-could-be-held).

    One entry point so that a caller cannot pair a relative label with absolute
    returns, which would be a quiet and very flattering mistake -- and so that
    the executable series is demeaned exactly the way the graded one is. A
    market-neutral book earns market-relative returns whichever window it is
    held over, and demeaning one and not the other would make the comparison
    between them meaningless.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")

    panel = forward_return_panel(frames, horizon=horizon)
    reachable = executable_return_panel(frames, horizon=horizon)

    if target == RELATIVE:
        return (relative_direction(panel, neutral_band=neutral_band),
                relative_forward_return(panel),
                relative_forward_return(reachable))

    labels = pd.DataFrame({
        symbol: direction(prices, horizon=horizon, threshold=threshold)
        for symbol, prices in frames.items()
    }).sort_index()
    return labels, panel, reachable
