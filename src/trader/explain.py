"""Why the model said what it said, and whether it has earned the right to say it.

Two separate questions, and the second one comes first.

**Has this model shown any skill?** Out of time, against the baseline of always
guessing the commoner direction, with enough days for the difference to mean
something. Almost every model of this kind fails that, including every one this
project has trained so far. A page that prints "Buy now" over a model with no
measured edge is not a trading tool, it is a random number generator with
confident typography.

So the verdict is gated. If the model has not beaten the baseline by more than
chance would produce, every symbol reads "Still collecting data" no matter what
the probability says. That is not the page being coy -- it is the only honest
thing it can say, and it is what "still collecting data" is for.

**Why this symbol, today?** By ablation: set one feature to its training average
and see how far the answer moves. Because the scaler standardises to mean zero,
"average" is exactly zero in scaled space, so this is a precise question rather
than an approximation -- if RSI had been ordinary instead of what it is, the
model would have said this much less.

That is a real attribution and a modest one. It says what moved this decision,
not what the feature means, and it is worth nothing at all if the model has no
edge -- which is why it is only ever shown underneath the gate.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# What the page can conclude.
BUY = "buy"
NO_BUY = "no_buy"
UNSURE = "unsure"

# How far past a coin flip a probability has to be before it is called at all.
# Two thirds of the way is deliberately demanding: on a target this noisy,
# anything less is the model rounding.
CALL_THRESHOLD = 0.58

# Below this many graded days, an edge is not distinguishable from luck however
# large it looks.
MIN_DAYS = 250


@dataclasses.dataclass
class Trust:
    """Whether the model's opinions are worth printing."""

    trusted: bool
    reason: str
    edge: float
    needed: float
    days: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def assess(evaluation: dict) -> Trust:
    """Decide whether this model has earned an opinion.

    The bar is that its accuracy beat the baseline by more than two standard
    errors -- computed from the number of days it was graded on rather than
    picked, because the honest threshold depends on how much evidence there is.
    On 4,000 days that is about 1.6 percentage points; on 400 it is about five.
    """
    days = int(evaluation.get("rows") or 0)
    accuracy = float(evaluation.get("accuracy") or 0.0)
    baseline = float(evaluation.get("baseline_accuracy") or 0.0)
    up_rate = float(evaluation.get("up_rate") or 0.0)
    edge = accuracy - baseline

    if days < MIN_DAYS:
        return Trust(False,
                     f"Only {days} days graded. Below about {MIN_DAYS} an edge "
                     f"cannot be told apart from luck.",
                     edge, float("nan"), days)

    # The spread of a proportion measured over `days` samples. Two of these is
    # the usual bar for "probably not chance".
    standard_error = math.sqrt(max(baseline * (1.0 - baseline), 1e-9) / days)
    needed = 2.0 * standard_error

    if up_rate > 0.97 or up_rate < 0.03:
        return Trust(False,
                     "The model answers the same way almost every day, so its "
                     "accuracy is just the class balance. It has learned nothing.",
                     edge, needed, days)

    if edge <= 0:
        return Trust(False,
                     "It does no better than always guessing the commoner "
                     "direction, on days it never saw.",
                     edge, needed, days)

    if edge < needed:
        return Trust(False,
                     f"It is {edge * 100:.2f} points above the baseline, and "
                     f"chance alone produces about {needed * 100:.2f} over "
                     f"{days:,} days. Not enough to act on.",
                     edge, needed, days)

    return Trust(True,
                 f"{edge * 100:.2f} points above the baseline over {days:,} "
                 f"days it never saw, which is past what chance produces "
                 f"({needed * 100:.2f}).",
                 edge, needed, days)


def rank(probability: float, trust: Trust) -> tuple[str, str]:
    """The call, and the sentence explaining it.

    Returns (verdict, because). The gate comes first: an untrusted model has no
    opinion worth ranking, whatever number came out of it.
    """
    if not trust.trusted:
        return UNSURE, (
            "Still collecting data — this model has not shown an edge yet, so "
            "its confidence here means nothing.")

    if probability >= CALL_THRESHOLD:
        return BUY, (
            f"It puts {probability:.0%} on this rising, past the {CALL_THRESHOLD:.0%} "
            f"it needs to say anything, from a model that has beaten the baseline.")

    if probability <= 1.0 - CALL_THRESHOLD:
        return NO_BUY, (
            f"It puts only {probability:.0%} on this rising, which is a call "
            f"against rather than an absence of one.")

    return UNSURE, (
        f"At {probability:.0%} it is too close to a coin flip to call, which is "
        f"the honest answer most days.")


def contributions(model, scaler, row: np.ndarray) -> list[dict]:
    """How much each feature moved today's answer, by setting it to average.

    The scaler standardises to mean zero, so replacing a scaled feature with 0
    is exactly "if this had been an ordinary day for this measure". The
    difference in the resulting probability is what that feature was worth.

    Signed: positive means the feature pushed towards up.
    """
    scaled = scaler.apply(np.asarray(row, dtype=np.float32).reshape(1, -1))
    base = float(model.probabilities(scaled)[0, 1])

    out = []
    for index, name in enumerate(scaler.feature_names):
        muted = scaled.copy()
        muted[0, index] = 0.0                       # the training average
        without = float(model.probabilities(muted)[0, 1])

        out.append({
            "feature": name,
            # What the model would have said without this feature's deviation.
            "without": round(without, 4),
            "effect": round(base - without, 4),
            "raw": round(float(np.asarray(row).ravel()[index]), 6),
            # How unusual today's value is, in standard deviations. The reason
            # a feature matters is usually that it is far from normal.
            "z": round(float(scaled[0, index]), 3),
        })

    out.sort(key=lambda item: abs(item["effect"]), reverse=True)
    return out


def in_words(contribution: dict) -> str:
    """One line a person can read, for one feature."""
    name = contribution["feature"].replace("_", " ")
    effect = contribution["effect"]
    z = contribution["z"]

    unusual = ("unusually high" if z > 1.5 else
               "unusually low" if z < -1.5 else
               "a little high" if z > 0.5 else
               "a little low" if z < -0.5 else
               "about average")

    if abs(effect) < 0.002:
        return f"{name} is {unusual} and made almost no difference."

    direction = "towards up" if effect > 0 else "towards down"
    return (f"{name} is {unusual}, and pushed the answer {direction} by "
            f"{abs(effect) * 100:.1f} points.")


def what_it_learnt(model, scaler, rows: np.ndarray, sample: int = 400) -> list[dict]:
    """Which features move this model at all, across many days.

    A per-day attribution says what mattered today. This says what the model
    pays attention to in general, which is the more useful thing to know about
    it -- and it is how you notice a model that is ignoring everything, which is
    what a network that has collapsed to predicting one class looks like from
    the inside.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    if len(rows) > sample:
        # Evenly spaced rather than random, so the answer is the same twice.
        rows = rows[np.linspace(0, len(rows) - 1, sample).astype(int)]

    scaled = scaler.apply(rows)
    base = model.probabilities(scaled)[:, 1]

    summary = []
    for index, name in enumerate(scaler.feature_names):
        muted = scaled.copy()
        muted[:, index] = 0.0
        without = model.probabilities(muted)[:, 1]
        shift = base - without

        summary.append({
            "feature": name,
            # Average size of the effect, ignoring direction: how much the model
            # uses this input at all.
            "influence": round(float(np.abs(shift).mean()), 5),
            # And which way it usually leans, which is the interpretable half.
            "leans": round(float(shift.mean()), 5),
        })

    summary.sort(key=lambda item: item["influence"], reverse=True)
    return summary
