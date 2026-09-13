"""Grading the model on data it has never seen, without flattering it.

Accuracy alone is close to meaningless here, and reporting it alone is how these
projects mislead the person who built them. Several things go beside it, always:

**The baseline, fixed before the test.** Roughly 52% of daily moves in this panel
are up, so a model that answers "up" every time scores 52%. The version of this
that took the *test* period's own majority was measuring against a number it
could only know afterwards. The class to beat is now whichever one dominated
training, which is the only one anybody actually had in advance.

**What it does when it commits.** A model that is 51% accurate overall but 60%
accurate on the 5% of days it is most confident about is worth something. One
that is uniformly 51% is not. Splitting by confidence separates them -- with a
minimum sample per bucket, because "100% accurate on its five most confident
days" is the single most misleading line a page like this can print, and this
one printed it.

**Money, after costs that reflect the trades actually made.** The earlier
version charged a round trip on every row. The model held long on 99% of days
and changed position 113 times in 4,187 rows, so it was charged 4,187 round
trips for 113 -- 14.9 points of annual return against a true 0.85. It made a
strategy that roughly matched buy-and-hold look like one that destroyed a
sixth of the account every year. Cost is now charged on the change in position,
per symbol, which is the definition of what a spread is paid on.

**Uncertainty, on the returns too.** The trust gate computes a standard error
for accuracy and refuses to speak without it, and then the return figures used
to be printed bare -- the one number a person would actually act on was the only
one with no error bar. Every return here carries a t-statistic, a Sharpe and a
worst drawdown, and the annualised figure is compounded from a real daily series
rather than from a mean.

Nothing here is a recommendation to trade. It is a measurement of whether a
model has any edge at all on days it did not see, and the honest answer for most
models of this kind is that it does not.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

# What a round trip costs, as a fraction of position value. Deliberately not
# zero: a strategy tested at zero cost is a strategy tested in a market that
# does not exist. Five basis points is optimistic for a retail account and
# generous enough that anything failing at this level fails for real.
DEFAULT_COST = 0.0005

TRADING_DAYS = 252

# Below this many rows a bucket's accuracy is an anecdote. It is still counted
# and still shown, but its accuracy is withheld rather than printed as though
# five days meant something.
MIN_BUCKET_ROWS = 30


@dataclasses.dataclass
class Evaluation:
    """Everything the UI needs to say how the model did, and how sure to be."""

    rows: int
    days: int                       # distinct dates
    symbols: int
    effective_rows: int             # rows discounted for cross-sectional correlation
    design_effect: float
    row_correlation: float
    accuracy: float
    baseline_accuracy: float        # the training-period majority, known in advance
    baseline_source: str
    test_majority: float            # for reference only; not knowable in advance
    edge: float
    up_rate: float                  # how often it says up, which catches a stuck model
    by_confidence: list

    # Money, from a real daily portfolio series.
    strategy_daily: float
    strategy_annualised: float
    strategy_sharpe: float
    strategy_tstat: float
    strategy_max_drawdown: float
    hold_daily: float
    hold_annualised: float
    hold_sharpe: float

    turnover_daily: float           # fraction of the book replaced per day
    position_changes: int
    cost_per_trade: float
    cost_drag_annualised: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _wilson(correct: int, total: int, z: float = 1.96) -> tuple:
    """A confidence interval for a proportion that behaves at small n.

    The textbook interval goes outside [0, 1] and collapses to zero width at
    0 or 100%, which is exactly where a confidence bucket lands. Wilson does
    neither, so an accuracy of 1.0 on nine rows reads as the very wide interval
    it is rather than as certainty.
    """
    if total == 0:
        return (None, None)
    phat = correct / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _bucket(probabilities: np.ndarray, correct: np.ndarray, net: np.ndarray,
            low: float, high: float) -> dict:
    """How the model did on the rows its confidence fell in a band."""
    confidence = np.abs(probabilities - 0.5) * 2.0        # 0 = coin flip, 1 = certain
    chosen = (confidence >= low) & (confidence < high)
    count = int(chosen.sum())

    out = {"from": round(low, 2), "to": round(high, 2), "rows": count,
           "accuracy": None, "accuracy_interval": (None, None),
           "mean_net_return": None, "enough": count >= MIN_BUCKET_ROWS}

    if count == 0:
        return out

    hits = int(correct[chosen].sum())
    # Net, matching the headline. Reporting gross here while the headline was
    # net invited comparing two different quantities as though they agreed.
    out["mean_net_return"] = round(float(net[chosen].mean()), 6)
    out["accuracy_interval"] = _wilson(hits, count)

    if count >= MIN_BUCKET_ROWS:
        out["accuracy"] = round(hits / count, 4)

    return out


def _portfolio(positions: pd.DataFrame, returns: pd.DataFrame,
               cost: float) -> tuple:
    """A real daily return series, with cost charged on the change in position.

    `positions` and `returns` are date x symbol. Position is -1, 0 or +1 and the
    book is equal-weighted across whatever it holds that day, so the series is
    what an account would have done rather than an average over stacked rows.
    """
    held = positions.notna() & (positions != 0)
    width = held.sum(axis=1).replace(0, np.nan)

    # Equal weight across the names held that day.
    weights = positions.fillna(0.0).div(width, axis=0).fillna(0.0)

    gross = (weights * returns.fillna(0.0)).sum(axis=1)

    # Turnover is how much of the book changed hands, which is what pays the
    # spread. A position carried unchanged from yesterday costs nothing.
    traded = weights.diff()
    traded.iloc[0] = weights.iloc[0]
    turnover = traded.abs().sum(axis=1)

    net = gross - turnover * cost
    changes = int((positions.fillna(0.0).diff().abs() > 0).sum().sum())

    return net, gross, turnover, changes


def _effective_rows(correct: np.ndarray, dates, symbols) -> tuple:
    """How many independent observations 4,000 correlated rows are really worth.

    Ten symbols on the same day are not ten independent verdicts on the model.
    They share a market, so when it is wrong about one it is more likely wrong
    about the others, and a standard error computed as though they were
    independent is too small -- the gate would then ask for less evidence than
    it thinks it is asking for.

    The correction is the usual design effect for clustered samples,
    1 + (m - 1) * rho, with rho measured from the panel rather than assumed.
    Measured on this project's own test set it comes out near 1.7, so the bar
    rises by about a third: not the difference between a signal and none, but
    exactly the sort of quiet third that gets a marginal result believed.
    """
    frame = pd.DataFrame({"date": pd.DatetimeIndex(dates),
                          "symbol": np.asarray(symbols).ravel(),
                          "correct": correct})
    panel = frame.pivot_table(index="date", columns="symbol", values="correct",
                              aggfunc="last")

    rows = len(frame)
    if panel.shape[1] < 2:
        return rows, 1.0, 0.0

    matrix = panel.corr().to_numpy()
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    # A symbol the model was uniformly right or wrong about has no variance, so
    # its correlations are undefined. That is not a correlation of zero, it is
    # an absent measurement, and averaging over an all-NaN slice warns and
    # returns NaN -- so the pairs that exist are counted and the rest ignored.
    usable = off_diagonal[np.isfinite(off_diagonal)]
    rho = float(usable.mean()) if usable.size else 0.0

    width = float(panel.notna().sum(axis=1).mean())
    # A negative average correlation would shrink the design effect below one,
    # which would be claiming the panel carries more information than it has.
    design = max(1.0 + (width - 1.0) * max(rho, 0.0), 1.0)

    return int(round(rows / design)), design, rho


def _drawdown(daily: pd.Series) -> float:
    """Worst peak-to-trough fall of the compounded series."""
    if daily.empty:
        return 0.0
    equity = (1.0 + daily).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(daily: pd.Series) -> float:
    spread = float(daily.std())
    if not spread or math.isnan(spread):
        return 0.0
    return float(daily.mean() / spread * math.sqrt(TRADING_DAYS))


def _tstat(daily: pd.Series) -> float:
    spread = float(daily.std())
    if not spread or math.isnan(spread) or len(daily) < 2:
        return 0.0
    return float(daily.mean() / (spread / math.sqrt(len(daily))))


def evaluate(probabilities, actual, forward_returns, dates, symbols, *,
             train_up_share: float | None = None,
             cost: float = DEFAULT_COST) -> Evaluation:
    """Score predictions against what actually happened next.

    `probabilities` is P(up) per row; `actual` the realised 1/0 label;
    `forward_returns` the move those rows are graded on -- market-relative when
    the target was relative, so that a market-neutral model is not credited with
    drift it never predicted. `dates` and `symbols` are what make a portfolio
    out of a pile of rows, and without them turnover cannot be known.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    actual = np.asarray(actual).ravel()
    forward_returns = np.asarray(forward_returns, dtype=np.float64).ravel()
    dates = pd.DatetimeIndex(np.asarray(dates).ravel())
    symbols = np.asarray(symbols).ravel()

    lengths = {len(probabilities), len(actual), len(forward_returns),
               len(dates), len(symbols)}
    if len(lengths) != 1:
        raise ValueError("predictions, labels, returns, dates and symbols must line up")
    if len(probabilities) == 0:
        raise ValueError("nothing to evaluate")

    predicted = (probabilities > 0.5).astype(int)
    correct = (predicted == actual).astype(float)
    accuracy = float(correct.mean())

    # The baseline that was knowable in advance: whichever class dominated the
    # training period. Falling back to the test period's own majority is the
    # old behaviour and is marked as such wherever it is used.
    test_majority = float(max((actual == 1).mean(), 1.0 - (actual == 1).mean()))
    if train_up_share is None:
        baseline = test_majority
        source = "test period majority (not knowable in advance)"
    else:
        always = 1 if train_up_share >= 0.5 else 0
        baseline = float((actual == always).mean())
        source = f"always '{'up' if always else 'down'}', the training majority"

    frame = pd.DataFrame({"date": dates, "symbol": symbols,
                          "position": np.where(predicted == 1, 1.0, -1.0),
                          "ret": forward_returns})
    positions = frame.pivot_table(index="date", columns="symbol",
                                  values="position", aggfunc="last")
    returns = frame.pivot_table(index="date", columns="symbol",
                                values="ret", aggfunc="last")

    net, gross, turnover, changes = _portfolio(positions, returns, cost)
    hold = returns.mean(axis=1).fillna(0.0)

    cost_drag = float((1.0 + gross.mean()) ** TRADING_DAYS
                      - (1.0 + net.mean()) ** TRADING_DAYS)

    # Per-row cost charged on that symbol's own change in position, for the
    # same reason the headline is: a flat charge per row bills a position that
    # was simply carried, and doing it in the buckets while fixing it in the
    # headline would leave the page comparing two different quantities again.
    traded_rows = positions.diff().abs()
    traded_rows.iloc[0] = positions.iloc[0].abs()
    row_cost = (traded_rows.stack(future_stack=True)
                .rename("cost").reset_index()
                .rename(columns={"level_0": "date", "level_1": "symbol"}))
    charged = frame.merge(row_cost, on=["date", "symbol"], how="left")
    net_rows = (charged["ret"].to_numpy() * charged["position"].to_numpy()
                - charged["cost"].fillna(0.0).to_numpy() * cost)

    buckets = [_bucket(probabilities, correct, net_rows, low, high)
               for low, high in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]]

    def annualise(series: pd.Series) -> float:
        """Compounded from the real series, not from its mean.

        A mean daily return raised to the 252nd power ignores volatility drag
        and reads high. This is what the account would have done.
        """
        if series.empty:
            return 0.0
        total = float((1.0 + series).prod())
        years = len(series) / TRADING_DAYS
        return float(total ** (1.0 / years) - 1.0) if years > 0 else 0.0

    effective, design, rho = _effective_rows(correct, dates, symbols)

    return Evaluation(
        rows=len(probabilities),
        days=int(len(positions.index)),
        symbols=int(len(set(symbols))),
        effective_rows=effective,
        design_effect=round(design, 3),
        row_correlation=round(rho, 4),
        accuracy=round(accuracy, 4),
        baseline_accuracy=round(baseline, 4),
        baseline_source=source,
        test_majority=round(test_majority, 4),
        edge=round(accuracy - baseline, 4),
        up_rate=round(float(predicted.mean()), 4),
        by_confidence=buckets,
        strategy_daily=round(float(net.mean()), 6),
        strategy_annualised=round(annualise(net), 4),
        strategy_sharpe=round(_sharpe(net), 3),
        strategy_tstat=round(_tstat(net), 3),
        strategy_max_drawdown=round(_drawdown(net), 4),
        hold_daily=round(float(hold.mean()), 6),
        hold_annualised=round(annualise(hold), 4),
        hold_sharpe=round(_sharpe(hold), 3),
        turnover_daily=round(float(turnover.mean()), 4),
        position_changes=changes,
        cost_per_trade=cost,
        cost_drag_annualised=round(cost_drag, 4),
    )


def verdict(evaluation: Evaluation) -> str:
    """One sentence, erring towards saying there is nothing here.

    The failure mode of a tool like this is that it is built by the person it is
    meant to inform, who would like it to work. So the wording leans the other
    way, and "no edge" is the default rather than the exception.
    """
    if evaluation.days < 60:
        return ("Too few distinct days to say anything. A couple of months can "
                "show any pattern you like.")

    if evaluation.up_rate > 0.97 or evaluation.up_rate < 0.03:
        return ("The model answers the same way almost every day, so its "
                "accuracy is just the class balance. It has learned nothing.")

    if evaluation.edge <= 0.0:
        return ("No edge: it does no better than always guessing the class that "
                "dominated training. This is the usual outcome and the honest "
                "reading is that there is no signal here.")

    if evaluation.edge < 0.02:
        return (f"{evaluation.edge:+.1%} over the baseline, which is within the "
                f"range chance produces over a period this length. Not evidence "
                f"of an edge.")

    if evaluation.strategy_tstat < 2.0:
        return (f"{evaluation.edge:+.1%} over the baseline, but the return has a "
                f"t-statistic of {evaluation.strategy_tstat:.1f}. Being right "
                f"more often has not turned into money you could distinguish "
                f"from luck.")

    return (f"{evaluation.edge:+.1%} over the baseline on {evaluation.days} days "
            f"it never saw, Sharpe {evaluation.strategy_sharpe:.2f}, t "
            f"{evaluation.strategy_tstat:.1f}. Worth another look, on a "
            f"different period, before believing it.")
