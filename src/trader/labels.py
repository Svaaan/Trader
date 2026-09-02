"""What the model is asked to predict.

Kept apart from features.py on purpose. Features may only look backwards and
labels must look forwards, so they are the two halves of the one mistake that
matters here -- putting them in separate files means the shift that creates a
label can never be mistaken for a rolling window that creates a feature.

The label on day t describes what happened *after* day t. It is not knowable on
day t, which is the point: that is what there would be value in predicting.
"""

from __future__ import annotations

import pandas as pd

UP, DOWN = 1, 0
CLASS_NAMES = ["down", "up"]        # index == label value


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
    """
    close = prices["close"]
    return (close.shift(-horizon) / close - 1.0).rename("forward_return")
