"""Why the model said what it said, and whether it has earned the right to say it.

Two separate questions, and the second one comes first.

**Has this model shown any skill?** Out of time, against the class that
dominated training, with enough evidence for the difference to mean something.
Almost every model of this kind fails that, including every one this project has
trained so far. A page that prints "Buy now" over a model with no measured edge
is not a trading tool, it is a random number generator with confident typography.

So the verdict is gated. If the model has not beaten the baseline by more than
chance would produce, every symbol reads "Still collecting data" no matter what
the probability says. That is not the page being coy -- it is the only honest
thing it can say, and it is what "still collecting data" is for.

The gate now has to clear five hurdles rather than three, and the two new ones
came from watching the old one nearly pass things it should not have:

**The sample size is the effective one.** Ten symbols on the same day are not
ten independent verdicts. Measured on this panel the design effect is about 1.7,
so the honest bar is a third higher than the naive one. A gate that is a third
too permissive is exactly the kind that lets a marginal result through.

**The edge has to beat the noise floor.** The same configuration trained three
times with different seeds produced accuracies spanning 1.8 points. Any "edge"
smaller than the spread between identical models is a statement about which seed
came up, and nothing else. This one number would have killed several results the
project once printed.

**And it has to survive more than one window.** A single test period is a single
draw. Walk-forward grades the same features on six consecutive stretches, and an
edge that appears in one of them is what a coin looks like. When walk-forward
results are available the gate requires most folds to agree.

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

# Below this many effective observations an edge is not distinguishable from
# luck however large it looks.
MIN_EFFECTIVE_ROWS = 250

# And below this many distinct dates, no amount of symbols helps: a wide panel
# over three weeks is still three weeks.
MIN_DAYS = 120

# How much of the walk-forward has to agree before a single window is believed.
MIN_FOLD_AGREEMENT = 0.7

# And how far the money has to be from zero -- measured over the window that
# could actually be held, not the one the label describes. Same two-standard-
# errors bar the accuracy hurdles use, applied to the thing somebody would
# actually act on.
#
# This hurdle exists because the others were not enough, which was found by
# running them. On the 238-symbol relative panel the logistic control clears
# every accuracy test there is -- 1.08 points over the baseline on 25,520
# independent rows, past chance, past the seed spread, positive in six of six
# walk-forward windows -- and loses 3.2% a year. It is right more often about
# small moves and wrong about large ones, which is a real and well-known way to
# be accurate and broke. Without this check the page would have printed "Buy
# now" over it.
MIN_RETURN_TSTAT = 2.0


@dataclasses.dataclass
class Trust:
    """Whether the model's opinions are worth printing."""

    trusted: bool
    reason: str
    edge: float
    needed: float
    days: int
    effective_rows: int
    noise_floor: float
    checks: list

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def assess(evaluation: dict, *, controls: dict | None = None,
           walk_forward: dict | None = None) -> Trust:
    """Decide whether this model has earned an opinion.

    Every hurdle is recorded in `checks` whether it passed or not, so the page
    can show what was tested rather than only the conclusion. A gate that says
    no without saying which question it failed teaches nobody anything.
    """
    rows = int(evaluation.get("rows") or 0)
    days = int(evaluation.get("days") or 0)
    effective = int(evaluation.get("effective_rows") or rows)
    accuracy = float(evaluation.get("accuracy") or 0.0)
    baseline = float(evaluation.get("baseline_accuracy") or 0.0)
    up_rate = float(evaluation.get("up_rate") or 0.0)
    edge = accuracy - baseline

    noise = 0.0
    if controls:
        noise = float((controls.get("noise_floor") or {}).get("spread") or 0.0)

    # The spread of a proportion measured over `effective` independent samples.
    # Two of these is the usual bar for "probably not chance".
    standard_error = math.sqrt(max(baseline * (1.0 - baseline), 1e-9)
                               / max(effective, 1))
    needed = 2.0 * standard_error

    checks: list = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def refuse(reason: str) -> Trust:
        return Trust(False, reason, edge, needed, days, effective, noise, checks)

    # --- enough evidence at all -------------------------------------------
    enough_days = days >= MIN_DAYS
    record("enough_days", enough_days,
           f"{days} distinct dates graded; {MIN_DAYS} is the floor.")
    if not enough_days:
        return refuse(f"Only {days} distinct days graded. Below about {MIN_DAYS} "
                      f"an edge cannot be told apart from luck, however many "
                      f"symbols are stacked on top of them.")

    enough_rows = effective >= MIN_EFFECTIVE_ROWS
    record("enough_effective_rows", enough_rows,
           f"{rows} rows are worth {effective} independent ones after "
           f"discounting for how much the panel moves together.")
    if not enough_rows:
        return refuse(f"{rows} rows, but only about {effective} of them are "
                      f"independent once the panel's shared movement is taken "
                      f"out. Not enough to measure anything.")

    # --- is it actually answering the question? ---------------------------
    varies = 0.03 <= up_rate <= 0.97
    record("model_varies", varies,
           f"It answers 'up' on {up_rate:.1%} of days.")
    if not varies:
        return refuse("The model answers the same way almost every day, so its "
                      "accuracy is just the class balance. It has learned nothing.")

    # --- does it beat the floor? ------------------------------------------
    positive = edge > 0
    record("beats_baseline", positive,
           f"{accuracy:.4f} against a baseline of {baseline:.4f} "
           f"({edge * 100:+.2f} points).")
    if not positive:
        return refuse("It does no better than always guessing the class that "
                      "dominated training, on days it never saw.")

    beats_chance = edge >= needed
    record("beats_chance", beats_chance,
           f"Chance alone produces about {needed * 100:.2f} points over "
           f"{effective:,} independent rows.")
    if not beats_chance:
        return refuse(f"It is {edge * 100:.2f} points above the baseline, and "
                      f"chance alone produces about {needed * 100:.2f}. "
                      f"Not enough to act on.")

    # --- is it bigger than the difference between identical models? -------
    beats_noise = noise <= 0.0 or edge > noise
    record("beats_noise_floor", beats_noise,
           f"Identical configurations trained on different seeds spanned "
           f"{noise * 100:.2f} points."
           if noise > 0 else "No noise floor was measured for this run.")
    if not beats_noise:
        return refuse(f"Its {edge * 100:.2f} point edge is smaller than the "
                      f"{noise * 100:.2f} points that separate identical models "
                      f"trained with different random seeds. That is a fact "
                      f"about the seed, not about the market.")

    # --- does it hold up in more than one window? -------------------------
    if walk_forward and walk_forward.get("folds_run"):
        run = int(walk_forward["folds_run"])
        good = int(walk_forward.get("folds_positive") or 0)
        share = good / max(run, 1)
        consistent = share >= MIN_FOLD_AGREEMENT
        record("consistent_across_time", consistent,
               f"Positive in {good} of {run} walk-forward windows.")
        if not consistent:
            return refuse(f"It only beats the baseline in {good} of {run} "
                          f"consecutive test windows. An edge that appears in "
                          f"one period and not the others is what a coin looks "
                          f"like.")
    else:
        record("consistent_across_time", True,
               "No walk-forward was run, so consistency over time is untested.")

    # --- and does being right actually pay, in a window you could hold? ----
    #
    # The executable series, not the graded one. The graded series runs close to
    # close and the signal is computed from that close, so nothing can be
    # positioned in time to collect it. Measured on this panel the two disagree
    # completely -- graded Sharpe +1.67, executable -0.40 -- because the whole
    # gross edge is the overnight gap. Gating on the graded number would open
    # the page over a trade nobody can place.
    tstat = float(evaluation.get("executable_tstat")
                  if evaluation.get("executable_tstat") is not None
                  else evaluation.get("strategy_tstat") or 0.0)
    sharpe = float(evaluation.get("executable_sharpe")
                   if evaluation.get("executable_sharpe") is not None
                   else evaluation.get("strategy_sharpe") or 0.0)
    gap = evaluation.get("execution_gap")

    pays = tstat >= MIN_RETURN_TSTAT
    detail = (f"Held from the first open after the signal, the strategy returns "
              f"a t-statistic of {tstat:.2f} (Sharpe {sharpe:.2f}); "
              f"{MIN_RETURN_TSTAT:.0f} is the bar.")
    if gap:
        detail += (f" The close-to-close version scores {gap:+.2f} Sharpe "
                   f"higher and cannot be traded.")
    record("makes_money", pays, detail)

    if not pays:
        return refuse(
            f"It is more accurate than the baseline, but held from the first "
            f"open after the signal exists its return has a t-statistic of "
            f"{tstat:.2f}. Being right more often is not the same as making "
            f"money, and a close-to-close backtest is not a trade anybody can "
            f"place.")

    return Trust(True,
                 f"{edge * 100:.2f} points above the baseline over {effective:,} "
                 f"independent rows across {days} days it never saw, past what "
                 f"chance produces ({needed * 100:.2f}) and past the "
                 f"{noise * 100:.2f} that separates identical models.",
                 edge, needed, days, effective, noise, checks)


def rank(probability: float, trust: Trust) -> tuple:
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


def contributions(model, scaler, row: np.ndarray) -> list:
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


def what_it_learnt(model, scaler, rows: np.ndarray, sample: int = 400) -> list:
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
