"""Building the training set, and splitting it the way time actually runs.

Two decisions here are the difference between a number that means something and
a number that does not.

**The split is chronological.** A random split of a price series lets the model
train on Tuesday and Thursday and be tested on Wednesday, with both neighbours
memorised. Markets are autocorrelated enough that this alone can lift accuracy
several points, and every one of them is fictional. Train is the earlier part of
history, test is the later part, and nothing crosses.

This matters twice over, because HelloWorldAi holds back a *random* 20% for its
own verification. That score is a real check on whether training worked -- it
catches a node returning untrained or corrupted weights, which is what it is
for -- but for time-series data it is optimistic as a measure of skill. So this
project keeps its own out-of-time test set, never sends it anywhere, and the UI
shows both numbers side by side rather than picking the flattering one.

**The scaler is fitted on training rows only.** Standardising with the mean and
standard deviation of the whole series tells the model, in a small but real way,
what the test period looked like. The statistics are computed on train, applied
to both, and written into the artifact so that inference months later uses the
same numbers rather than re-deriving them from whatever data is at hand.
"""

from __future__ import annotations

import dataclasses
import io
import json
from typing import Sequence

import numpy as np
import pandas as pd

from . import features as features_mod
from . import labels as labels_mod


@dataclasses.dataclass
class Split:
    """One symbol's data, cut in time."""

    symbol: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    forward_returns_test: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.y_train) + len(self.y_test)


@dataclasses.dataclass
class Scaler:
    """Mean and standard deviation per feature, fitted on training rows only."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    def apply(self, x: np.ndarray) -> np.ndarray:
        # A feature that never moves has std 0; dividing by it produces inf,
        # and the model then sees a column of infinities instead of a constant.
        safe = np.where(self.std > 1e-12, self.std, 1.0)
        return (x - self.mean) / safe

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Scaler":
        return cls(
            mean=np.asarray(raw["mean"], dtype=np.float32),
            std=np.asarray(raw["std"], dtype=np.float32),
            feature_names=list(raw["feature_names"]),
        )


def choose_cut_date(frames: dict, *, horizon: int = 1, threshold: float = 0.0,
                    test_fraction: float = 0.2) -> pd.Timestamp:
    """One date, before which everything trains and after which everything tests.

    Splitting each symbol at its own 80% mark looks chronological per symbol and
    is not chronological at all across the pool: symbols have different amounts
    of history, so Apple's training rows can run years past the start of SAP's
    test rows. These markets move together, so that is the same leak as a random
    split, wearing a different hat -- the model sees 2025 in one name and is
    graded on 2025 in another.

    Measured on the first real panel built here: per-symbol splits gave a
    training window ending 2026-04-21 and a test window starting 2024-09-09,
    a twenty-month overlap.

    So the cut is a single date, chosen so that roughly `test_fraction` of all
    rows across all symbols fall after it.
    """
    all_dates = []
    for symbol, prices in frames.items():
        x_frame = features_mod.build(prices)
        y_series = labels_mod.direction(prices, horizon=horizon, threshold=threshold)
        joined = x_frame.join(y_series, how="inner").dropna()
        all_dates.append(pd.DatetimeIndex(joined.index))

    if not all_dates:
        raise ValueError("no symbols to choose a cut date from")

    pooled = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates])).sort_values()
    position = int(len(pooled) * (1.0 - test_fraction))
    position = min(max(position, 1), len(pooled) - 1)

    return pooled[position]


def build_one(symbol: str, prices: pd.DataFrame, *, horizon: int = 1,
              threshold: float = 0.0, test_fraction: float = 0.2,
              cut_date: pd.Timestamp | None = None) -> Split:
    """Features, labels and a chronological split for a single symbol.

    `cut_date` splits at a fixed date rather than a fraction, which is what
    pooling several symbols requires -- see choose_cut_date.
    """
    x_frame = features_mod.build(prices)
    y_series = labels_mod.direction(prices, horizon=horizon, threshold=threshold)
    fwd = labels_mod.forward_return(prices, horizon=horizon)

    # Keep only dates that have both a full set of features and a known future.
    frame = x_frame.join(y_series, how="inner").join(fwd, how="inner").dropna()
    if frame.empty:
        raise ValueError(f"{symbol}: no rows survive feature and label alignment")

    x = frame[features_mod.FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    returns = frame["forward_return"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index)

    if cut_date is not None:
        cut = int((dates < cut_date).sum())
    else:
        cut = int(len(frame) * (1.0 - test_fraction))

    if cut < 1 or cut >= len(frame):
        raise ValueError(
            f"{symbol}: {len(frame)} rows leave nothing on one side of the split "
            f"(cut at {cut})")

    return Split(
        symbol=symbol,
        x_train=x[:cut], y_train=y[:cut],
        x_test=x[cut:], y_test=y[cut:],
        train_dates=dates[:cut], test_dates=dates[cut:],
        forward_returns_test=returns[cut:],
    )


def combine(splits: Sequence[Split]) -> tuple[np.ndarray, np.ndarray, Scaler]:
    """Pool several symbols into one training set, scaled on train rows only.

    Pooling is deliberate. One symbol gives a few thousand rows, which is not
    much to learn from, and a pattern that only exists in one company's history
    is more likely to be that company's last few years than anything general.
    """
    if not splits:
        raise ValueError("nothing to combine")

    # Sorted by date, not stacked by symbol.
    #
    # Training does not care -- batches are drawn at random, so the order of the
    # rows makes no difference to what is learned. The coordinator's holdout
    # does care. Telling it the rows are in time order and then handing it all
    # of Apple followed by all of SAP means "hold back the last 20%" holds back
    # the tail of the last company rather than the most recent period, which is
    # not the question anybody meant to ask.
    #
    # Measured: declaring time order on symbol-stacked rows gave a holdout score
    # of 54.9%, no better than the random slice it replaced.
    x_parts = np.concatenate([s.x_train for s in splits])
    y_parts = np.concatenate([s.y_train for s in splits])
    dates = np.concatenate([s.train_dates.values for s in splits])

    # Stable, so rows sharing a date keep a deterministic order between runs.
    order = np.argsort(dates, kind="stable")
    x_train = x_parts[order]
    y_train = y_parts[order]

    scaler = Scaler(
        mean=x_train.mean(axis=0),
        std=x_train.std(axis=0),
        feature_names=list(features_mod.FEATURE_NAMES),
    )

    return scaler.apply(x_train).astype(np.float32), y_train, scaler


def pack_for_helloworld(x: np.ndarray, y: np.ndarray) -> bytes:
    """The .npz shape HelloWorldAi's artifact loader accepts.

    Its loader refuses anything it would have to unpickle, which rules out a
    CSV or an object array -- the arrays go in as plain numeric types under the
    names it expects.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} feature rows against {y.shape[0]} labels")
    if x.shape[0] == 0:
        raise ValueError("refusing to send an empty dataset")

    buffer = io.BytesIO()
    np.savez(buffer, x=x.astype(np.float32), y=y.astype(np.int64))
    return buffer.getvalue()


def describe(splits: Sequence[Split], scaler: Scaler) -> dict:
    """Everything the UI needs to say what this dataset is.

    Written out rather than summarised into a single "quality" number, because
    the things that make a dataset misleading -- one class dominating, a test
    period that is all one market regime, too few rows -- are all visible in the
    detail and invisible in an average.
    """
    y_train = np.concatenate([s.y_train for s in splits])
    y_test = np.concatenate([s.y_test for s in splits])

    def balance(y: np.ndarray) -> dict:
        if len(y) == 0:
            return {"rows": 0, "up": 0, "down": 0, "up_share": None}
        up = int((y == labels_mod.UP).sum())
        return {
            "rows": int(len(y)),
            "up": up,
            "down": int(len(y) - up),
            # The number to beat. A model that always says "up" scores this, and
            # a headline accuracy is meaningless without it beside it.
            "up_share": round(float(up) / len(y), 4),
        }

    return {
        "symbols": [s.symbol for s in splits],
        "feature_names": list(scaler.feature_names),
        "train": {
            **balance(y_train),
            "from": min(s.train_dates[0] for s in splits).date().isoformat(),
            "to": max(s.train_dates[-1] for s in splits).date().isoformat(),
        },
        "test": {
            **balance(y_test),
            "from": min(s.test_dates[0] for s in splits).date().isoformat(),
            "to": max(s.test_dates[-1] for s in splits).date().isoformat(),
        },
        "scaler": scaler.to_dict(),
    }


def save_description(path: str, description: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(description, handle, indent=2)


def build_panel(frames: dict, *, horizon: int = 1, threshold: float = 0.0,
                test_fraction: float = 0.2) -> tuple[list[Split], pd.Timestamp]:
    """Every symbol, split at one shared date.

    The only correct way to pool: see choose_cut_date for what happens when
    each symbol picks its own.
    """
    if not frames:
        raise ValueError("no price data to build from")

    cut = choose_cut_date(frames, horizon=horizon, threshold=threshold,
                          test_fraction=test_fraction)

    splits = []
    for symbol, prices in frames.items():
        try:
            splits.append(build_one(symbol, prices, horizon=horizon,
                                    threshold=threshold, cut_date=cut))
        except ValueError:
            # A symbol with too little history to sit on both sides of the cut
            # is left out rather than split somewhere else, which would put it
            # back in the overlap this function exists to prevent.
            continue

    if not splits:
        raise ValueError(f"no symbol has data on both sides of {cut.date()}")

    return splits, cut
