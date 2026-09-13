"""Building the training set, and splitting it the way time actually runs.

Four decisions here are the difference between a number that means something
and a number that does not.

**The split is chronological, at one date across the whole panel.** A random
split of a price series lets the model train on Tuesday and Thursday and be
tested on Wednesday, with both neighbours memorised. Splitting each symbol at
its own 80% mark looks chronological and is not: symbols have different amounts
of history, so one name's training rows can run years past the start of
another's test rows, and these markets move together. One date, everything
before it trains, everything after it tests.

**The cut date is chosen once and then carried, never re-derived.** It used to
be recomputed at scoring time from a stored row-count, which reproduced the
original split only by luck: measured, it moved from 2024-11-06 to 2024-12-31,
graded 4,187 rows while the run advertised 4,100, and silently added a symbol to
the test set that had been absent from training. Worse, the drift is not
directional -- a single failed price fetch at scoring time changes the pool,
which moves the date, which can move it *earlier* and put trained rows into the
score this project exists to keep clean. So `cut_date` is an argument, it is
stored with the run, and scoring passes back the one that was used.

**Training labels may not reach across the cut.** A row on the last training
day is labelled with a return that realises after it -- inside the test period.
At a one-day horizon that is one row per symbol and nearly harmless; at a
twenty-day horizon it is twenty rows of direct leakage. The last `horizon` rows
before the cut are dropped from training. This is the standard purge and it
costs almost nothing.

**The scaler is fitted on training rows only.** Standardising with the mean and
standard deviation of the whole series tells the model, in a small but real way,
what the test period looked like. The statistics are computed on train, applied
to both, and written into the artifact so that inference months later uses the
same numbers rather than re-deriving them from whatever data is at hand.

A note on what is assembled here. Features arrive from five places -- the
symbol's own prices, its position among its peers, the state of the market, the
distance to its next announcement, and the point-in-time news store -- and each
is optional so that the contribution of each can be measured rather than
assumed. The order of the columns is fixed and recorded, because the model
treats them positionally and a reordering is silent and total.

The news block is the one that can be asked for and refused. It is built
forwards and is worthless until it has history, so asking for it over a young
store gets a warning and no columns rather than a column of zeros for every
historical row and a real number for today.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from . import cross as cross_mod
from . import events as events_mod
from . import features as features_mod
from . import labels as labels_mod
from . import macro as macro_mod
from . import news as news_mod

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Spec:
    """Which blocks go in, and what is being predicted.

    Held together so that a run records exactly what it was built from. Two runs
    whose numbers differ are only comparable if this is identical, and it is
    stored with both.
    """

    target: str = labels_mod.RELATIVE
    horizon: int = 1
    threshold: float = 0.0
    neutral_band: float = 0.0
    use_macro: bool = True
    use_cross: bool = True
    use_events: bool = True
    # Asking for the news block is not the same as getting it. The store is
    # built forwards and is worthless until it has history, so `assemble`
    # consults news.readiness and refuses the block until it clears -- loudly,
    # in the report, rather than by handing the model a column of zeros for
    # every historical row and a real number for today.
    use_news: bool = False
    test_fraction: float = 0.2

    @property
    def embargo(self) -> int:
        """Training rows dropped before the cut so no label reaches across it."""
        return self.horizon

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["embargo"] = self.embargo
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "Spec":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in fields})


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
    # The same rows measured over the window somebody could actually hold --
    # open(t+1) to close(t+horizon). Carried beside the graded return rather
    # than instead of it, because the pair is what says whether the strategy
    # exists. See labels.executable_return.
    executable_returns_test: np.ndarray
    # Sessions each row holds. Carried by the data rather than passed alongside
    # it, because every statistic past one session has to know, and a keyword
    # threaded through six call sites is a keyword one of them forgets.
    horizon: int = 1

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


# --- assembling the inputs -------------------------------------------------

def assemble(frames: dict, spec: Spec | None = None, *,
             refresh: bool = False) -> tuple[dict, list[str], dict]:
    """Every feature block, joined per symbol.

    Returns (per-symbol feature frames, ordered column names, a report of what
    was and was not available). The report is not decoration: a panel where the
    macro block failed to download, or where half the names have no earnings
    calendar, produces perfectly plausible numbers and the only way to know is
    to have written it down.
    """
    spec = spec or Spec()

    own = {}
    for symbol, prices in frames.items():
        try:
            frame = features_mod.build(prices)
            if not frame.empty:
                own[symbol] = frame
        except Exception as exc:                        # noqa: BLE001
            logger.warning("No features for %s: %s", symbol, exc)

    if not own:
        raise ValueError("no symbol produced any features")

    report = {"symbols_with_features": sorted(own)}
    names = list(features_mod.FEATURE_NAMES)

    # --- cross-sectional ---------------------------------------------------
    cross_frames = {}
    if spec.use_cross:
        cross_frames = cross_mod.build(own)
        cross_names = cross_mod.names(cross_frames)
        names += cross_names
        report["cross"] = {"used": bool(cross_names), "columns": cross_names,
                           "panel_width": len(own)}
    else:
        report["cross"] = {"used": False, "columns": []}

    # --- macro -------------------------------------------------------------
    macro_frame = None
    if spec.use_macro:
        try:
            macro_frame = macro_mod.build(refresh=refresh)
            macro_names = macro_mod.names(macro_frame)
            names += macro_names
            report["macro"] = {"used": True, "columns": macro_names,
                               "rows": len(macro_frame)}
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Macro block unavailable, continuing without it: %s", exc)
            report["macro"] = {"used": False, "columns": [], "error": str(exc)}
    else:
        report["macro"] = {"used": False, "columns": []}

    # --- events ------------------------------------------------------------
    if spec.use_events:
        names += list(events_mod.EVENT_NAMES)
        report["events"] = {"used": True,
                            "columns": list(events_mod.EVENT_NAMES),
                            "coverage": events_mod.coverage(sorted(own),
                                                            refresh=refresh)}
    else:
        report["events"] = {"used": False, "columns": []}

    # --- news --------------------------------------------------------------
    use_news = False
    if spec.use_news:
        state = news_mod.readiness(sorted(own))
        use_news = bool(state.get("ready"))
        if use_news:
            names += list(news_mod.NEWS_NAMES)
        else:
            logger.warning(
                "News block asked for but the store has %d days of %d; "
                "leaving it out rather than feeding zeros",
                state.get("history_days", 0), news_mod.MIN_HISTORY_DAYS)
        report["news"] = {"used": use_news,
                          "columns": list(news_mod.NEWS_NAMES) if use_news else [],
                          "readiness": state}
    else:
        report["news"] = {"used": False, "columns": []}

    # --- join --------------------------------------------------------------
    assembled = {}
    for symbol, frame in own.items():
        joined = frame

        if spec.use_cross and symbol in cross_frames:
            joined = joined.join(cross_frames[symbol], how="left")

        if macro_frame is not None:
            joined = joined.join(macro_frame, how="left")

        if spec.use_events:
            joined = joined.join(
                events_mod.build(symbol, joined.index, refresh=refresh), how="left")

        if use_news:
            joined = joined.join(news_mod.build(symbol, joined.index), how="left")

        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if not joined.empty:
            assembled[symbol] = joined[[n for n in names if n in joined.columns]]

    if not assembled:
        raise ValueError("no symbol survived joining the feature blocks")

    # Whatever actually made it through, in the declared order.
    present = [n for n in names if n in next(iter(assembled.values())).columns]
    report["feature_names"] = present
    report["feature_count"] = len(present)

    return assembled, present, report


# --- splitting -------------------------------------------------------------

def choose_cut_date(assembled: dict, label_frame: pd.DataFrame, *,
                    test_fraction: float = 0.2) -> pd.Timestamp:
    """One date, before which everything trains and after which everything tests.

    Chosen so that roughly `test_fraction` of all usable rows across all symbols
    fall after it. Call this once per run and then carry the answer -- see the
    module docstring for what re-deriving it costs.
    """
    pooled = []
    for symbol, frame in assembled.items():
        if symbol not in label_frame.columns:
            continue
        usable = frame.index.intersection(label_frame[symbol].dropna().index)
        pooled.append(pd.DatetimeIndex(usable))

    if not pooled:
        raise ValueError("no symbol has both features and labels")

    dates = pd.DatetimeIndex(np.concatenate([d.values for d in pooled])).sort_values()
    position = int(len(dates) * (1.0 - test_fraction))
    position = min(max(position, 1), len(dates) - 1)

    return dates[position]


def build_one(symbol: str, frame: pd.DataFrame, label: pd.Series,
              graded: pd.Series, executable: pd.Series, *,
              cut_date: pd.Timestamp, feature_names: Sequence[str],
              embargo: int = 1, horizon: int = 1) -> Split:
    """Features, labels and a chronological split for a single symbol.

    Two return series, not one: what the label describes (close to close) and
    what could actually be held (open after the signal, to the same exit). They
    are joined together so a row survives only if both exist -- grading a model
    on rows where one window is missing and the other is not would make the
    comparison between them a comparison of different rows.
    """
    joined = frame.join(label.rename("label"), how="inner") \
                  .join(graded.rename("graded"), how="inner") \
                  .join(executable.rename("executable"), how="inner").dropna()
    if joined.empty:
        raise ValueError(f"{symbol}: no rows survive feature and label alignment")

    x = joined[list(feature_names)].to_numpy(dtype=np.float32)
    y = joined["label"].to_numpy(dtype=np.int64)
    returns = joined["graded"].to_numpy(dtype=np.float32)
    reachable = joined["executable"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(joined.index)

    cut = int((dates < cut_date).sum())

    # The purge. A row this close to the cut is labelled with a return that
    # realises on the far side of it.
    train_end = cut - max(embargo, 0)

    if train_end < 1 or cut >= len(joined):
        raise ValueError(
            f"{symbol}: {len(joined)} rows leave nothing on one side of "
            f"{pd.Timestamp(cut_date).date()} (train ends at {train_end})")

    return Split(
        symbol=symbol,
        x_train=x[:train_end], y_train=y[:train_end],
        x_test=x[cut:], y_test=y[cut:],
        train_dates=dates[:train_end], test_dates=dates[cut:],
        forward_returns_test=returns[cut:],
        executable_returns_test=reachable[cut:],
        horizon=horizon,
    )


@dataclasses.dataclass
class Prepared:
    """Everything assembled and labelled, before any date has been chosen.

    Separated from splitting so that walk-forward validation can cut the same
    assembled panel at a dozen dates without rebuilding features a dozen times.
    """

    assembled: dict
    label_frame: pd.DataFrame
    graded_frame: pd.DataFrame
    executable_frame: pd.DataFrame
    feature_names: list
    report: dict
    spec: Spec


def prepare(frames: dict, spec: Spec | None = None, *,
            refresh: bool = False) -> Prepared:
    """Features and labels for the whole panel, not yet split."""
    spec = spec or Spec()
    if not frames:
        raise ValueError("no price data to build from")

    assembled, feature_names, report = assemble(frames, spec, refresh=refresh)

    label_frame, graded_frame, executable_frame = labels_mod.build_panel_labels(
        {s: frames[s] for s in assembled},
        horizon=spec.horizon, target=spec.target,
        threshold=spec.threshold, neutral_band=spec.neutral_band)

    report["requested"] = sorted(frames)
    return Prepared(assembled=assembled, label_frame=label_frame,
                    graded_frame=graded_frame,
                    executable_frame=executable_frame,
                    feature_names=feature_names, report=report, spec=spec)


def split_at(prepared: Prepared, cut_date: pd.Timestamp) -> tuple[list, dict]:
    """Cut an assembled panel at one date. The only place a split is made."""
    spec = prepared.spec
    assembled = prepared.assembled
    label_frame = prepared.label_frame
    graded_frame = prepared.graded_frame
    executable_frame = prepared.executable_frame
    feature_names = prepared.feature_names
    report = dict(prepared.report)
    frames = assembled
    cut_date = pd.Timestamp(cut_date)

    splits, excluded = [], {}
    for symbol in sorted(assembled):
        if symbol not in label_frame.columns:
            excluded[symbol] = "no labels"
            continue
        try:
            splits.append(build_one(
                symbol, assembled[symbol], label_frame[symbol],
                graded_frame[symbol], executable_frame[symbol],
                cut_date=cut_date, feature_names=feature_names,
                embargo=spec.embargo, horizon=spec.horizon))
        except ValueError as exc:
            # Left out rather than split somewhere else, which would put it back
            # in the overlap the single cut date exists to prevent. Named, so
            # that a watchlist of ten training as nine is a line in the run
            # rather than something nobody notices for a month.
            excluded[symbol] = str(exc)

    if not splits:
        raise ValueError(f"no symbol has data on both sides of {cut_date.date()}")

    dropped = sorted(set(report.get("requested", frames)) - {s.symbol for s in splits})
    for symbol in dropped:
        excluded.setdefault(symbol, "no usable features")

    if excluded:
        logger.warning("Excluded from the panel: %s", sorted(excluded))

    report["excluded"] = excluded
    report["included"] = [s.symbol for s in splits]

    return splits, report


def build_panel(frames: dict, spec: Spec | None = None, *,
                cut_date: pd.Timestamp | None = None,
                refresh: bool = False) -> tuple[list[Split], pd.Timestamp, dict]:
    """Every symbol, split at one date, with everything that happened recorded.

    Pass `cut_date` to reproduce an existing split exactly; leave it out to
    choose one. Scoring a trained model must always pass the one its run
    recorded -- see the module docstring for what re-deriving it costs.
    """
    prepared = prepare(frames, spec, refresh=refresh)

    if cut_date is None:
        cut_date = choose_cut_date(prepared.assembled, prepared.label_frame,
                                   test_fraction=prepared.spec.test_fraction)
    cut_date = pd.Timestamp(cut_date)

    splits, report = split_at(prepared, cut_date)
    return splits, cut_date, report


# --- pooling ---------------------------------------------------------------

def combine(splits: Sequence[Split],
            feature_names: Sequence[str] | None = None
            ) -> tuple[np.ndarray, np.ndarray, Scaler]:
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
        feature_names=list(feature_names or features_mod.FEATURE_NAMES),
    )

    return scaler.apply(x_train).astype(np.float32), y_train, scaler


@dataclasses.dataclass
class TestSet:
    """The pooled test rows, named rather than positional.

    This was a tuple of four and then a tuple of a tuple of four and a fifth,
    which is the shape an argument list takes just before somebody passes the
    returns where the dates go. Six things with names cost nothing.
    """

    x: np.ndarray
    y: np.ndarray
    returns: np.ndarray             # close(t) -> close(t+h), what the label means
    executable: np.ndarray          # open(t+1) -> close(t+h), what can be held
    dates: pd.DatetimeIndex
    symbols: np.ndarray
    horizon: int = 1                # sessions each row holds; evaluate needs it

    def __len__(self) -> int:
        return len(self.y)


def test_matrix(splits: Sequence[Split], scaler: Scaler) -> TestSet:
    """The pooled test set, scaled, in date order, with its dates.

    Date order rather than symbol order, because everything that reads this --
    turnover, drawdown, any statistic that treats consecutive rows as consecutive
    -- is wrong on rows stacked by symbol.
    """
    x = np.concatenate([s.x_test for s in splits])
    y = np.concatenate([s.y_test for s in splits])
    returns = np.concatenate([s.forward_returns_test for s in splits])
    reachable = np.concatenate([s.executable_returns_test for s in splits])
    dates = np.concatenate([s.test_dates.values for s in splits])
    symbols = np.concatenate([np.full(len(s.y_test), s.symbol) for s in splits])

    horizons = {s.horizon for s in splits}
    if len(horizons) != 1:
        raise ValueError(f"splits disagree about their horizon: {sorted(horizons)}")

    order = np.argsort(dates, kind="stable")
    return TestSet(
        x=scaler.apply(x[order]).astype(np.float32),
        y=y[order],
        returns=returns[order],
        executable=reachable[order],
        dates=pd.DatetimeIndex(dates[order]),
        symbols=symbols[order],
        horizon=horizons.pop(),
    )


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


def describe(splits: Sequence[Split], scaler: Scaler, spec: Spec,
             report: dict, cut_date: pd.Timestamp) -> dict:
    """Everything the UI needs to say what this dataset is.

    Written out rather than summarised into a single "quality" number, because
    the things that make a dataset misleading -- one class dominating, a test
    period that is all one market regime, too few rows, a block that silently
    failed to download -- are all visible in the detail and invisible in an
    average.
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
        "spec": spec.to_dict(),
        "cut_date": pd.Timestamp(cut_date).date().isoformat(),
        "symbols": [s.symbol for s in splits],
        "feature_names": list(scaler.feature_names),
        "blocks": {k: report.get(k) for k in ("cross", "macro", "events", "news")},
        "excluded": report.get("excluded", {}),
        "requested": report.get("requested", []),
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
