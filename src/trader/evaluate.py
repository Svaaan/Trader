"""Grading the model on data it has never seen, without flattering it.

Accuracy alone is close to meaningless here, and reporting it alone is how
these projects mislead the person who built them. Three things go beside it,
always:

**The baseline.** Roughly 52% of daily moves in this panel are up. A model that
answers "up" every time scores 52%. A headline of 53% is not a result, it is
noise with a decimal point, and the only way to see that is to have the baseline
printed next to it.

**What it does when it commits.** A model that is 51% accurate overall but 60%
accurate on the 5% of days it is most confident about is worth something. One
that is uniformly 51% is not. Splitting by confidence separates them.

**Money, after costs.** Accuracy counts days; returns count size. Being right
about nine small moves and wrong about one large one is a losing week that reads
as 90% accurate. And every trade pays spread and commission, which a backtest
that ignores them turns into profit that does not exist.

Nothing here is a recommendation to trade. It is a measurement of whether a
model has any edge at all on days it did not see, and the honest answer for
most models of this kind is that it does not.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# What a round trip costs, as a fraction of position value. Deliberately not
# zero: a strategy tested at zero cost is a strategy tested in a market that
# does not exist. Five basis points is optimistic for a retail account and
# generous enough that anything failing at this level fails for real.
DEFAULT_COST = 0.0005


@dataclasses.dataclass
class Evaluation:
    """Everything the UI needs to say how the model did, and how sure to be."""

    rows: int
    accuracy: float
    baseline_accuracy: float
    edge: float                     # accuracy - baseline; the only number that matters
    up_rate: float                  # how often it says up, which catches a stuck model
    by_confidence: list[dict]
    strategy_daily: float           # mean net return per day held
    hold_daily: float               # the same for simply holding
    strategy_annualised: float
    hold_annualised: float
    trades: int
    cost_per_trade: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _bucket(probabilities: np.ndarray, correct: np.ndarray,
            returns: np.ndarray, low: float, high: float) -> dict:
    """How the model did on the days its confidence fell in a band."""
    confidence = np.abs(probabilities - 0.5) * 2.0        # 0 = coin flip, 1 = certain
    chosen = (confidence >= low) & (confidence < high)
    count = int(chosen.sum())

    if count == 0:
        return {"from": round(low, 2), "to": round(high, 2), "days": 0,
                "accuracy": None, "mean_return": None}

    return {
        "from": round(low, 2),
        "to": round(high, 2),
        "days": count,
        "accuracy": round(float(correct[chosen].mean()), 4),
        # Signed by the direction taken, so it reads as what the position earned.
        "mean_return": round(float(
            (np.where(probabilities[chosen] > 0.5, 1.0, -1.0) * returns[chosen]).mean()
        ), 6),
    }


def evaluate(probabilities: np.ndarray, actual: np.ndarray,
             forward_returns: np.ndarray, *,
             cost: float = DEFAULT_COST) -> Evaluation:
    """Score predictions against what actually happened next.

    `probabilities` is P(up) per row, `actual` is the realised 1/0 label, and
    `forward_returns` is the move those rows are graded on.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    actual = np.asarray(actual).ravel()
    forward_returns = np.asarray(forward_returns, dtype=np.float64).ravel()

    if not (len(probabilities) == len(actual) == len(forward_returns)):
        raise ValueError("predictions, labels and returns must line up")
    if len(probabilities) == 0:
        raise ValueError("nothing to evaluate")

    predicted = (probabilities > 0.5).astype(int)
    correct = (predicted == actual).astype(float)

    accuracy = float(correct.mean())

    # The score of always guessing the more common class -- the real floor.
    up_share = float((actual == 1).mean())
    baseline = max(up_share, 1.0 - up_share)

    # Long when it says up, short when it says down. Every row is a position, so
    # every row pays the cost; a strategy that trades less would pay less, and
    # that is a different strategy than the one being measured.
    direction = np.where(predicted == 1, 1.0, -1.0)
    gross = direction * forward_returns
    net = gross - cost

    buckets = [_bucket(probabilities, correct, forward_returns, low, high)
               for low, high in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]]

    # Per day, not compounded along the whole array.
    #
    # The rows are ten symbols stacked one after another, so compounding through
    # them describes holding Apple for two years, then ASML for the same two
    # years, then Nestle -- which is not a strategy anybody ran, and it produced
    # a headline "buy and hold" of +1165% on the first real run. A mean daily
    # return is the same measurement without the fiction, and the annualised
    # figure beside it is that mean at 252 trading days.
    strategy_daily = float(net.mean())
    hold_daily = float(forward_returns.mean())

    def annualise(daily: float) -> float:
        return float((1.0 + daily) ** 252 - 1.0)

    return Evaluation(
        rows=len(probabilities),
        accuracy=round(accuracy, 4),
        baseline_accuracy=round(baseline, 4),
        edge=round(accuracy - baseline, 4),
        up_rate=round(float(predicted.mean()), 4),
        by_confidence=buckets,
        strategy_daily=round(strategy_daily, 6),
        hold_daily=round(hold_daily, 6),
        strategy_annualised=round(annualise(strategy_daily), 4),
        hold_annualised=round(annualise(hold_daily), 4),
        trades=int(len(predicted)),
        cost_per_trade=cost,
    )


def verdict(evaluation: Evaluation) -> str:
    """One sentence, erring towards saying there is nothing here.

    The failure mode of a tool like this is that it is built by the person it
    is meant to inform, who would like it to work. So the wording leans the
    other way, and "no edge" is the default rather than the exception.
    """
    if evaluation.rows < 250:
        return ("Too few days to say anything. A couple of hundred rows can "
                "show any pattern you like.")

    if evaluation.up_rate > 0.97 or evaluation.up_rate < 0.03:
        return ("The model answers the same way almost every day, so its "
                "accuracy is just the class balance. It has learned nothing.")

    if evaluation.edge <= 0.0:
        return ("No edge: it does no better than always guessing the more "
                "common direction. This is the usual outcome and the honest "
                "reading is that there is no signal here.")

    if evaluation.edge < 0.02:
        return (f"{evaluation.edge:+.1%} over the baseline, which is within the "
                f"range chance produces over a period this length. Not evidence "
                f"of an edge.")

    if evaluation.strategy_daily <= evaluation.hold_daily:
        return (f"{evaluation.edge:+.1%} over the baseline, but after costs it "
                f"still does no better than holding. Being right more often is "
                f"not the same as making money.")

    return (f"{evaluation.edge:+.1%} over the baseline and ahead of holding "
            f"after costs, on {evaluation.rows} days it never saw. Worth another "
            f"look, on a different period, before believing it.")
