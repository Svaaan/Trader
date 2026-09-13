"""Searching over configurations without spending the one number that matters.

Compute stopped being the constraint some time ago. A model trains in six
seconds and a card can train sixty-four at once, so an afternoon is a few
thousand configurations. The constraint now is statistical, and it is sharp:

    **How many times can you ask the test set a question before its answer
    stops meaning anything?**

The answer is roughly once. A gate set at two standard errors lets noise
through about one time in forty; run four hundred configurations and keep the
best, and something clears it with near-certainty on a panel of pure noise.
Every safeguard this project has -- the noise floor, the walk-forward, the
design-effect discount, the executable-return hurdle -- sits *downstream* of a
test set that has already been mined, and none of them can detect it.

So the data is cut three ways instead of two:

    train        the model fits on this
    validation   the search is scored on this, as often as you like
    test         opened once, for the configuration you commit to

Validation exists to be spent; that is its job. The test set is sealed behind
`open_test_set`, which refuses to run unless a configuration has been committed
and writes down that it happened. And every configuration tried is recorded in a
ledger, so that when a result is finally taken to the test set the page can say
"the best of 1,847" -- because a 1.08-point edge means something very different
at trial 1 than at trial 1,847.

None of this makes a search safe on its own. It makes the cost of one visible,
which is the most any harness can do.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
import logging
import math
import os

import numpy as np
import pandas as pd

from . import baseline as baseline_mod
from . import dataset as dataset_mod
from . import evaluate as evaluate_mod
from . import labels as labels_mod

logger = logging.getLogger(__name__)

SEARCH_DIR = os.environ.get(
    "TRADER_SEARCH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "search"),
)

TRIALS_FILE = "trials.jsonl"
SEAL_FILE = "seal.json"

# How much of the *training* period is held back to score the search on. Taken
# off the end, chronologically, so it rehearses the test set rather than
# interleaving with the rows the model fits.
VALIDATION_FRACTION = 0.2


@dataclasses.dataclass
class Cut:
    """Two dates, three periods. The only place a three-way split is defined."""

    validation_start: pd.Timestamp
    test_start: pd.Timestamp

    def to_dict(self) -> dict:
        return {"validation_start": self.validation_start.date().isoformat(),
                "test_start": self.test_start.date().isoformat()}


def _paths() -> tuple:
    root = os.path.abspath(SEARCH_DIR)
    return os.path.join(root, TRIALS_FILE), os.path.join(root, SEAL_FILE)


def three_way_cut(prepared, *, test_fraction: float = 0.2,
                  validation_fraction: float = VALIDATION_FRACTION) -> Cut:
    """Where to cut, so that the search never touches the last period.

    Both boundaries are chosen from the pooled row dates, and both are chosen
    once. The test boundary is the same one `dataset.choose_cut_date` would
    pick, so a sealed run and an ordinary one are graded on identical rows --
    otherwise the sealed number could not be compared to anything.
    """
    pooled = []
    for symbol, frame in prepared.assembled.items():
        if symbol in prepared.label_frame.columns:
            pooled.append(pd.DatetimeIndex(frame.index.intersection(
                prepared.label_frame[symbol].dropna().index)))
    if not pooled:
        raise ValueError("no symbol has both features and labels")

    dates = pd.DatetimeIndex(
        np.concatenate([d.values for d in pooled])).sort_values()

    test_at = int(len(dates) * (1.0 - test_fraction))
    # The validation slice comes out of what is left, not out of the test set.
    validation_at = int(test_at * (1.0 - validation_fraction))

    return Cut(validation_start=dates[max(validation_at, 1)],
               test_start=dates[min(max(test_at, 2), len(dates) - 1)])


def _score_on(prepared, cut: Cut, spec, *, period: str,
              seed: int = 0) -> dict:
    """Fit on the training period and score on validation or test.

    The model never sees rows at or after whichever boundary it is being scored
    against, which is the only thing that makes either number mean anything.
    """
    if period == "validation":
        splits, _ = dataset_mod.split_at(prepared, cut.validation_start)
        graded = [_truncate_before(s, cut.test_start) for s in splits]
    else:
        splits, _ = dataset_mod.split_at(prepared, cut.test_start)
        graded = splits

    graded = [s for s in graded if len(s.y_test) > 0 and len(s.y_train) > 0]
    if not graded:
        raise ValueError(f"nothing left to score on the {period} period")

    x_train, y_train, scaler = dataset_mod.combine(
        graded, prepared.feature_names)
    test = dataset_mod.test_matrix(graded, scaler)

    model = baseline_mod.fit_mlp(x_train, y_train, steps=20_000, seed=seed)
    probabilities = model.probabilities(test.x)

    result = evaluate_mod.evaluate(
        probabilities, test.y, test.returns, test.dates, test.symbols,
        executable_returns=test.executable,
        train_up_share=float(y_train.mean()), horizon=test.horizon)

    return result.to_dict()


def _truncate_before(split, stop: pd.Timestamp):
    """Keep only the test rows before `stop` -- the validation window."""
    return baseline_mod._truncate(split, stop)


# --- the ledger -------------------------------------------------------------

def record_trial(spec, scores: dict, *, note: str = "") -> dict:
    """Append one configuration and what it scored on validation.

    Append-only. The count is the whole point: it is what turns "1.08 points
    above baseline" into "the best 1.08 points out of 1,847 tries", which are
    very different claims and only one of them is usually true.
    """
    path, _ = _paths()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {
        "trial": count_trials() + 1,
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec": spec.to_dict() if hasattr(spec, "to_dict") else dict(spec),
        "note": note,
        # Deliberately only the validation numbers. A trial that recorded a
        # test score would be a trial that had opened the test set.
        "validation": {
            "accuracy": scores.get("accuracy"),
            "baseline": scores.get("baseline_accuracy"),
            "edge": scores.get("edge"),
            "executable_sharpe": scores.get("executable_sharpe"),
            "executable_tstat": scores.get("executable_tstat"),
            "rows": scores.get("rows"),
            # The bar is set by this, not by rows: two hundred symbols on the
            # same day are nowhere near two hundred independent observations.
            "effective_rows": scores.get("effective_rows"),
            "days": scores.get("days"),
        },
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_trials() -> list:
    path, _ = _paths()
    if not os.path.exists(path):
        return []

    trials = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping an unparseable trial line")
    return trials


def count_trials() -> int:
    return len(read_trials())


def leaderboard(limit: int = 20, by: str = "executable_sharpe") -> list:
    """The best configurations so far, on validation only."""
    trials = [t for t in read_trials()
              if (t.get("validation") or {}).get(by) is not None]
    trials.sort(key=lambda t: t["validation"][by], reverse=True)
    return trials[:limit]


# --- what a search costs ----------------------------------------------------

def corrected_threshold(trials: int, baseline: float, effective_rows: int,
                        alpha: float = 0.05) -> dict:
    """How large an edge has to be once you have looked this many times.

    A two-standard-error bar admits noise about one time in twenty. Look `n`
    times and keep the best, and the chance of at least one false pass is
    1 - (1 - alpha)^n, which reaches near-certainty within a hundred trials. The
    Sidak correction puts that back: test each trial at alpha / n instead.

    This is not a formality. At 400 trials the honest bar is roughly twice the
    naive one, and a result that clears 2 SE after a long search has not cleared
    anything.
    """
    trials = max(int(trials), 1)
    naive_z = 1.959964

    # Sidak: per-trial alpha such that the family-wise rate stays at `alpha`.
    per_trial = 1.0 - (1.0 - alpha) ** (1.0 / trials)
    # Two-sided z for that tail, via an inverse-normal approximation good to
    # about 4e-4 -- ample here, where the question is "roughly how much worse".
    z = _inverse_normal(1.0 - per_trial / 2.0)

    error = math.sqrt(max(baseline * (1.0 - baseline), 1e-9)
                      / max(effective_rows, 1))

    return {
        "trials": trials,
        "naive": round(naive_z * error, 5),
        "corrected": round(z * error, 5),
        "inflation": round(z / naive_z, 3),
        "family_wise_risk_uncorrected": round(
            1.0 - (1.0 - alpha) ** trials, 4),
    }


def _inverse_normal(p: float) -> float:
    """Acklam's approximation. Good enough to size a bar, not to publish."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    low, high = 0.02425, 1 - 0.02425

    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --- the seal ---------------------------------------------------------------

def seal_state() -> dict:
    _, path = _paths()
    if not os.path.exists(path):
        return {"opened": [], "committed": None}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:                                   # noqa: BLE001
        return {"opened": [], "committed": None}


def commit(spec, *, why: str) -> dict:
    """Declare the configuration you are taking to the test set, and why.

    Required before `open_test_set` will do anything. The point is not
    bureaucracy -- it is that committing *before* looking is the whole
    difference between a test and a search. Writing down the reason is what
    stops the commit being decided by the answer.
    """
    _, path = _paths()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = seal_state()
    state["committed"] = {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec": spec.to_dict() if hasattr(spec, "to_dict") else dict(spec),
        "why": why,
        "after_trials": count_trials(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    logger.info("Committed a configuration after %d trials",
                state["committed"]["after_trials"])
    return state["committed"]


def open_test_set(prepared, cut: Cut, *, seed: int = 0) -> dict:
    """Score the committed configuration on the sealed period. Once.

    Refuses without a commit, and records every opening. A second opening is
    allowed -- forbidding it would just mean people delete the file -- but it is
    counted and reported, because the number of times a test set has been looked
    at is exactly the thing that determines what its answer is worth.
    """
    state = seal_state()
    committed = state.get("committed")
    if not committed:
        raise RuntimeError(
            "Nothing has been committed. Choose a configuration on the "
            "validation period and call commit() with the reason, then open "
            "the test set -- deciding after looking is not a test.")

    spec = dataset_mod.Spec.from_dict(committed["spec"])
    scores = _score_on(prepared, cut, spec, period="test", seed=seed)

    trials = committed.get("after_trials", 0)
    correction = corrected_threshold(
        trials,
        baseline=float(scores.get("baseline_accuracy") or 0.5),
        effective_rows=int(scores.get("effective_rows") or scores.get("rows") or 1))

    opening = {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "opening_number": len(state.get("opened", [])) + 1,
        "after_trials": trials,
        "spec": committed["spec"],
        "why": committed["why"],
        "scores": scores,
        "correction": correction,
        "reading": _reading(scores, correction,
                            len(state.get("opened", [])) + 1),
    }

    state.setdefault("opened", []).append(opening)
    _, path = _paths()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    logger.warning("Test set opened (opening %d, after %d trials)",
                   opening["opening_number"], trials)
    return opening


def _reading(scores: dict, correction: dict, opening_number: int) -> str:
    """What the sealed number is worth, given how it was arrived at."""
    edge = float(scores.get("edge") or 0.0)
    tstat = float(scores.get("executable_tstat") or 0.0)
    trials = correction["trials"]

    lines = []
    if opening_number > 1:
        lines.append(
            f"This is opening number {opening_number}. The first was the test; "
            f"the rest are a slower search.")

    if trials > 1:
        lines.append(
            f"Best of {trials} configurations. Chance alone clears the naive "
            f"bar {correction['family_wise_risk_uncorrected']:.0%} of the time "
            f"over that many tries, so the bar is "
            f"{correction['inflation']:.1f}x higher: "
            f"{correction['corrected'] * 100:.2f} points rather than "
            f"{correction['naive'] * 100:.2f}.")

    if edge < correction["corrected"]:
        lines.append(
            f"The measured edge is {edge * 100:+.2f} points, which does not "
            f"clear it.")
    elif tstat < 2.0:
        lines.append(
            f"The edge clears the corrected bar, but held from the first open "
            f"after the signal the return has a t-statistic of {tstat:.2f}. "
            f"Accurate and unprofitable is still unprofitable.")
    else:
        lines.append(
            f"{edge * 100:+.2f} points past a bar corrected for "
            f"{trials} trials, and a return t-statistic of {tstat:.2f}. This "
            f"is the strongest thing this project can say, and the next step "
            f"is forward paper trading, not another search.")

    return " ".join(lines)


# --- running a search -------------------------------------------------------

def grid(**options) -> list:
    """Every combination of the options given, as Spec overrides."""
    keys = list(options)
    return [dict(zip(keys, values))
            for values in itertools.product(*(options[k] for k in keys))]


def run_search(frames: dict, combinations: list, *, base: dataset_mod.Spec | None = None,
               test_fraction: float = 0.2, seed: int = 0,
               note: str = "") -> dict:
    """Score every configuration on validation, and never on test.

    The panel is assembled once per distinct feature-block combination rather
    than once per trial: assembling is minutes and fitting is seconds, so doing
    it per trial would spend the entire budget on rebuilding the same columns.
    """
    base = base or dataset_mod.Spec()
    results = []

    # Group by the things that change the assembled columns, so the expensive
    # part happens once per group instead of once per trial.
    def assembly_key(override: dict) -> tuple:
        merged = {**dataclasses.asdict(base), **override}
        return (merged["use_macro"], merged["use_cross"], merged["use_events"],
                merged["use_news"], merged["horizon"], merged["target"],
                merged["threshold"], merged["neutral_band"])

    combinations = sorted(combinations, key=assembly_key)
    for key, group in itertools.groupby(combinations, key=assembly_key):
        group = list(group)
        spec = dataset_mod.Spec(**{**dataclasses.asdict(base), **group[0]})
        try:
            prepared = dataset_mod.prepare(frames, spec)
            cut = three_way_cut(prepared, test_fraction=test_fraction)
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Could not prepare %s: %s", key, exc)
            continue

        for override in group:
            trial_spec = dataset_mod.Spec(
                **{**dataclasses.asdict(base), **override})
            try:
                scores = _score_on(prepared, cut, trial_spec,
                                   period="validation", seed=seed)
            except Exception as exc:                    # noqa: BLE001
                logger.warning("Trial failed (%s): %s", override, exc)
                continue

            entry = record_trial(trial_spec, scores, note=note)
            results.append(entry)
            logger.info("Trial %d: edge %+.4f, executable sharpe %+.2f  %s",
                        entry["trial"], scores.get("edge") or 0.0,
                        scores.get("executable_sharpe") or 0.0, override)

    return {
        "ran": len(results),
        "total_trials": count_trials(),
        "best": leaderboard(5),
    }
