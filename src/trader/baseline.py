"""Models that run here, in under a second, so the expensive one has something
to be compared against.

This is the piece the project was missing, and its absence was expensive. Nine
models were trained on a GPU across a network, each a round trip of minutes to
hours, and not one of them was ever compared with anything but the class
balance. A logistic regression on the same rows takes four tenths of a second
and scores the same. That does not mean the distributed training is broken; it
means nobody could have told if it were.

So three controls run locally before every submission, and their scores are
stored with the run:

**Majority.** Answer the training majority every time. The floor. If the trained
model does not clear this it has learned nothing, and this is the number the
trust gate measures against.

**Logistic regression.** A linear model on the same features, same split, same
scaler. If the network does not beat this, the non-linearity is not buying
anything and the honest description of the result is "a linear model, slowly".

**A local MLP of the same shape.** Same width and depth as the job that goes to
HelloWorldAi. If the remote model scores meaningfully *worse* than this, the
problem is in the round trip -- placement, hyperparameters, the holdout it
carves out -- rather than in the data. That is a question worth being able to
answer, and before this there was no way to ask it.

And two things the controls make possible that a single model cannot:

**Walk-forward.** One test period is one draw. The same feature set graded on
six consecutive periods says whether an edge persists or whether one window
happened to be kind. It is far too expensive to do this on a GPU round trip and
nearly free to do it here.

**A noise floor.** The same configuration trained several times with different
seeds. Whatever spread that produces is the smallest difference between two
models that means anything at all, and on this project's own numbers -- three
identical runs scoring 51.44, 51.64 and 51.78 -- it was wider than every edge
ever measured.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from . import dataset as dataset_mod
from . import evaluate as evaluate_mod

logger = logging.getLogger(__name__)

# Total gradient steps the local MLP control may spend, across all seeds. It is
# a budget rather than a match because a wide panel earns tens of thousands of
# steps and three seeds of that in numpy is ten minutes of somebody waiting --
# which is how controls get switched off, which is how this project ended up
# with nine models and nothing to compare them to.
LOCAL_STEP_BUDGET = 36_000


# --- the models ------------------------------------------------------------

def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so neither branch overflows: exp of a large positive number
    # is inf, and inf/inf is nan, which propagates silently through everything.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclasses.dataclass
class Logistic:
    """A linear model, fitted by gradient descent with an L2 penalty."""

    weights: np.ndarray

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(_add_bias(np.asarray(x, dtype=np.float64)) @ self.weights)


def fit_logistic(x: np.ndarray, y: np.ndarray, *, epochs: int = 600,
                 lr: float = 0.5, l2: float = 1e-4) -> Logistic:
    """Full-batch gradient descent. The objective is convex, so this is enough."""
    features = _add_bias(np.asarray(x, dtype=np.float64))
    target = np.asarray(y, dtype=np.float64)
    weights = np.zeros(features.shape[1])

    for _ in range(epochs):
        error = _sigmoid(features @ weights) - target
        gradient = features.T @ error / len(features)
        gradient[:-1] += l2 * weights[:-1]          # the bias is not penalised
        weights -= lr * gradient

    return Logistic(weights=weights)


@dataclasses.dataclass
class MLP:
    """The same shape as the job that goes to HelloWorldAi, run here."""

    layers: list

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        activation = np.asarray(x, dtype=np.float64)
        for index, (weight, bias) in enumerate(self.layers):
            activation = activation @ weight + bias
            if index < len(self.layers) - 1:
                activation = np.maximum(activation, 0.0)
        return _sigmoid(activation.ravel())


def fit_mlp(x: np.ndarray, y: np.ndarray, *, hidden: int = 64, depth: int = 2,
            steps: int = 4000, batch: int = 64, lr: float = 0.01,
            seed: int = 0) -> MLP:
    """A small ReLU network trained with Adam, in numpy.

    Counted in **gradient steps, not epochs**, because that is how the job
    submitted to HelloWorldAi is counted -- `steps=4000, batch_size=64`. Passing
    the same numbers means the local control does the same amount of learning as
    the remote model rather than an amount that happens to depend on how many
    symbols are in the panel. Two consequences, both wanted: the comparison is
    fair, and a run on 240 symbols costs the same as one on 10 instead of
    twenty-four times as much. Epoch-counting made the controls take minutes on
    a wide panel, which is long enough that somebody would turn them off.

    Deliberately not torch. The project refuses that dependency to keep every
    arithmetic step readable, and a control that needed two gigabytes of CUDA to
    run would not be much of a control.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    sizes = [x.shape[1]] + [hidden] * depth + [1]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        # He initialisation, which is what ReLU wants.
        layers.append([rng.normal(0.0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)])

    moment1 = [[np.zeros_like(w), np.zeros_like(b)] for w, b in layers]
    moment2 = [[np.zeros_like(w), np.zeros_like(b)] for w, b in layers]

    order = rng.permutation(len(x))
    cursor = 0

    for step in range(1, max(steps, 1) + 1):
        # Draw the next minibatch, reshuffling when the pass runs out. Sampling
        # without replacement within a pass keeps the gradient estimates less
        # correlated than drawing independently every time.
        if cursor + batch > len(order):
            order = rng.permutation(len(x))
            cursor = 0
        index = order[cursor:cursor + batch]
        cursor += batch

        xb, yb = x[index], y[index]

        activations = [xb]
        for depth_index, (weight, bias) in enumerate(layers):
            z = activations[-1] @ weight + bias
            activations.append(z if depth_index == len(layers) - 1
                               else np.maximum(z, 0.0))

        # Cross-entropy through a sigmoid differentiates to exactly this.
        delta = (_sigmoid(activations[-1]) - yb) / len(xb)

        for depth_index in range(len(layers) - 1, -1, -1):
            weight, bias = layers[depth_index]
            previous = activations[depth_index]

            grad_w = previous.T @ delta
            grad_b = delta.sum(axis=0)

            if depth_index > 0:
                delta = (delta @ weight.T) * (activations[depth_index] > 0)

            for slot, (grad, param) in enumerate(
                    ((grad_w, weight), (grad_b, bias))):
                moment1[depth_index][slot] = (
                    0.9 * moment1[depth_index][slot] + 0.1 * grad)
                moment2[depth_index][slot] = (
                    0.999 * moment2[depth_index][slot] + 0.001 * grad * grad)
                corrected1 = moment1[depth_index][slot] / (1 - 0.9 ** step)
                corrected2 = moment2[depth_index][slot] / (1 - 0.999 ** step)
                param -= lr * corrected1 / (np.sqrt(corrected2) + 1e-8)

    return MLP(layers=layers)


@dataclasses.dataclass
class Majority:
    """Answer the training majority, every time. The floor."""

    answer: int

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(x)), 0.99 if self.answer else 0.01)


# --- running them ----------------------------------------------------------

def _score(probabilities, splits, scaler, train_up_share, cost) -> dict:
    (x, y, returns, dates), symbols = dataset_mod.test_matrix(splits, scaler)
    result = evaluate_mod.evaluate(probabilities, y, returns, dates, symbols,
                                   train_up_share=train_up_share, cost=cost)
    return result.to_dict()


def run_controls(splits, feature_names, *, hidden: int = 64, depth: int = 2,
                 steps: int = 4000, batch: int = 64,
                 cost: float = evaluate_mod.DEFAULT_COST,
                 seeds: int = 3) -> dict:
    """Fit every control on the training half and score it out of time.

    `hidden`, `depth`, `steps` and `batch` should be exactly what is being sent
    to HelloWorldAi -- pipeline passes them through. A control trained for a
    different length than the model it is a control for is not answering the
    question.

    Returns a dictionary the run stores verbatim. `seeds` refits the MLP that
    many times to measure the noise floor -- the spread between identical
    configurations, which is the smallest difference between two models that
    means anything.
    """
    x_train, y_train, scaler = dataset_mod.combine(splits, feature_names)
    (x_test, _, _, _), _ = dataset_mod.test_matrix(splits, scaler)
    train_up_share = float(y_train.mean())

    out: dict = {"train_up_share": round(train_up_share, 4),
                 "train_rows": int(len(y_train)),
                 "hyperparameters": {"hidden": hidden, "depth": depth,
                                     "steps": steps, "batch": batch}}

    majority = Majority(answer=1 if train_up_share >= 0.5 else 0)
    out["majority"] = _score(majority.probabilities(x_test), splits, scaler,
                             train_up_share, cost)

    logistic = fit_logistic(x_train, y_train)
    out["logistic"] = _score(logistic.probabilities(x_test), splits, scaler,
                             train_up_share, cost)

    # Matching the submission exactly is the ideal and is not always affordable:
    # a wide panel earns 80,000 steps, and three seeds of that in numpy is ten
    # minutes of somebody waiting. So the local work has a budget, and when it
    # binds the run records how much of the submission's training the control
    # actually got -- a control that saw a quarter of the steps is still worth
    # having and is not the same claim as one that saw all of them.
    seeds = max(seeds, 1)
    local_steps = min(steps, max(LOCAL_STEP_BUDGET // seeds, 1))
    out["local_steps"] = local_steps
    out["local_step_share"] = round(local_steps / max(steps, 1), 3)

    accuracies = []
    for seed in range(seeds):
        model = fit_mlp(x_train, y_train, hidden=hidden, depth=depth,
                        steps=local_steps, batch=batch, seed=seed)
        scored = _score(model.probabilities(x_test), splits, scaler,
                        train_up_share, cost)
        accuracies.append(scored["accuracy"])
        if seed == 0:
            out["local_mlp"] = scored

    # The number that says how much of any "edge" is just which seed came up.
    out["noise_floor"] = {
        "seeds": len(accuracies),
        "steps_each": local_steps,
        "accuracies": accuracies,
        "spread": round(float(max(accuracies) - min(accuracies)), 4),
        "sd": round(float(np.std(accuracies)), 4),
    }

    return out


def walk_forward(frames: dict, spec, *, folds: int = 6,
                 cost: float = evaluate_mod.DEFAULT_COST,
                 model: str = "logistic") -> dict:
    """Grade the same feature set on several consecutive out-of-time periods.

    Expanding window: each fold trains on everything before its cut and is
    graded on the stretch after it, up to the next cut. One test period is one
    draw, and a single flattering window is the most common way a strategy that
    does not work comes to look as though it does.

    Runs on the local models only. Six GPU round trips to answer this would take
    a day; this takes seconds, and it is answering a question about the *data*
    rather than about any particular trained network.
    """
    prepared = dataset_mod.prepare(frames, spec)

    pooled = []
    for symbol, frame in prepared.assembled.items():
        if symbol in prepared.label_frame.columns:
            pooled.append(pd.DatetimeIndex(
                frame.index.intersection(
                    prepared.label_frame[symbol].dropna().index)))
    dates = pd.DatetimeIndex(np.concatenate([d.values for d in pooled])).sort_values()

    # Cuts spread across the back half, so every fold trains on a decent history.
    quantiles = np.linspace(0.5, 1.0, folds + 1)[:-1]
    cuts = [dates[int(len(dates) * q)] for q in quantiles]

    results = []
    for index, cut in enumerate(cuts):
        stop = cuts[index + 1] if index + 1 < len(cuts) else None
        try:
            splits, _ = dataset_mod.split_at(prepared, cut)
            if stop is not None:
                splits = [_truncate(s, stop) for s in splits]
                splits = [s for s in splits if len(s.y_test) > 0]
            if not splits:
                continue

            x_train, y_train, scaler = dataset_mod.combine(
                splits, prepared.feature_names)
            (x_test, _, _, _), _ = dataset_mod.test_matrix(splits, scaler)

            fitted = (fit_logistic(x_train, y_train) if model == "logistic"
                      else fit_mlp(x_train, y_train, seed=0))
            scored = _score(fitted.probabilities(x_test), splits, scaler,
                            float(y_train.mean()), cost)

            results.append({
                "fold": index + 1,
                "cut": pd.Timestamp(cut).date().isoformat(),
                "test_from": min(s.test_dates[0] for s in splits).date().isoformat(),
                "test_to": max(s.test_dates[-1] for s in splits).date().isoformat(),
                "rows": scored["rows"],
                "accuracy": scored["accuracy"],
                "baseline": scored["baseline_accuracy"],
                "edge": scored["edge"],
                "sharpe": scored["strategy_sharpe"],
                "annualised": scored["strategy_annualised"],
            })
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Walk-forward fold %d failed: %s", index + 1, exc)

    edges = [r["edge"] for r in results]
    return {
        "model": model,
        "folds": results,
        # The summary that matters: an edge that is real shows up in most
        # windows, not in one. A mean with a sign and nothing else is how one
        # good fold gets mistaken for a strategy.
        "mean_edge": round(float(np.mean(edges)), 4) if edges else None,
        "sd_edge": round(float(np.std(edges)), 4) if edges else None,
        "folds_positive": int(sum(e > 0 for e in edges)),
        "folds_run": len(edges),
    }


def _truncate(split, stop: pd.Timestamp):
    """Keep only the test rows before `stop`, for one walk-forward fold."""
    keep = split.test_dates < pd.Timestamp(stop)
    return dataset_mod.Split(
        symbol=split.symbol,
        x_train=split.x_train, y_train=split.y_train,
        x_test=split.x_test[keep], y_test=split.y_test[keep],
        train_dates=split.train_dates, test_dates=split.test_dates[keep],
        forward_returns_test=split.forward_returns_test[keep],
    )
