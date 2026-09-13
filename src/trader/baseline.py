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
from typing import Sequence

import numpy as np
import pandas as pd

from . import arrays as arrays_mod
from .arrays import CPU, GPU
from . import dataset as dataset_mod
from . import evaluate as evaluate_mod

logger = logging.getLogger(__name__)

# Total gradient steps the local MLP control may spend, across all seeds. It is
# a budget rather than a match because a wide panel earns tens of thousands of
# steps and three seeds of that in numpy is ten minutes of somebody waiting --
# which is how controls get switched off, which is how this project ended up
# with nine models and nothing to compare them to.
LOCAL_STEP_BUDGET = 36_000

# Where the GPU starts to win, measured rather than assumed. Training the same
# 7,233-parameter network at batch 64, on this machine:
#
#     models      cpu       gpu    gpu s/model
#          1     1.1s     12.5s         12.53s
#          4     3.4s     12.4s          3.10s
#         16    13.3s     13.2s          0.82s
#         64    95.7s     15.0s          0.23s
#
# The card's time is flat -- it is waiting between kernel launches, not
# computing -- so every model after the first is nearly free, while the CPU pays
# for each one. Below sixteen the launches cost more than the arithmetic saves.
#
# This is why the answer to "use the GPU" is "train many models at once" rather
# than "train one model faster". One model on the card is ten times slower.
GPU_MODEL_CROSSOVER = 16

# And the same question for the logistic fit, which is one big matmul rather
# than many small ones. Measured, 600 passes:
#
#     rows        cpu      gpu
#      3,000     0.12s    1.23s    <- transfer and warm-up dominate
#     20,000     0.76s    0.35s
#    428,100    23.36s    2.30s
#
# Routing on "is there a card" rather than "is the problem big enough" made the
# test suite take nine minutes instead of one, because every small fit paid a
# transfer it could not earn back. Size is the question, not availability.
GPU_ROW_CROSSOVER = 20_000


# --- the models ------------------------------------------------------------

def _add_bias(x, xp=np):
    return xp.concatenate([x, xp.ones((len(x), 1), dtype=x.dtype)], axis=1)


def _sigmoid(z, xp=np):
    """Numerically stable, and written so it runs on either array module.

    Boolean-mask assignment is slow on a GPU and fine on a CPU, so this uses
    `where` on the clipped input instead: same arithmetic, no branch, no
    scatter. exp of a large positive number is inf and inf/inf is nan, which
    propagates silently through everything, so the clip is doing real work.
    """
    clipped = xp.clip(z, -60.0, 60.0)
    return xp.where(clipped >= 0,
                    1.0 / (1.0 + xp.exp(-clipped)),
                    xp.exp(clipped) / (1.0 + xp.exp(clipped)))


@dataclasses.dataclass
class Logistic:
    """A linear model, fitted by gradient descent with an L2 penalty."""

    weights: np.ndarray

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(_add_bias(np.asarray(x, dtype=np.float64)) @ self.weights)



def fit_logistic(x: np.ndarray, y: np.ndarray, *, epochs: int = 600,
                 lr: float = 0.5, l2: float = 1e-4,
                 device: str | None = None) -> Logistic:
    """Full-batch gradient descent. The objective is convex, so this is enough.

    This is the slowest thing in the project and the best case for a card: six
    hundred passes over the whole training matrix, which on a wide panel is
    600 x (428,000 x 47) twice per pass. Measured at 21.7 seconds on the CPU,
    and walk-forward runs it six times.

    It is also the safest thing to move, because there is no randomness in it at
    all -- the weights start at zero and the objective is convex, so CPU and GPU
    reach the same answer rather than two answers that have to be argued about.
    Measured: 20.8 seconds on the CPU, 3.6 on the card, weights agreeing to
    2e-17.
    """
    if device is None:
        # Unlike the MLP, this is a handful of enormous matmuls rather than
        # thousands of tiny ones, so the card wins -- but only once there is
        # enough matrix to pay for moving it there.
        device = (GPU if len(x) >= GPU_ROW_CROSSOVER and arrays_mod.available()
                  else CPU)
    module = arrays_mod.xp(device)

    features = _add_bias(module.asarray(x, dtype=module.float64), module)
    target = module.asarray(y, dtype=module.float64)
    weights = module.zeros(features.shape[1], dtype=module.float64)

    # The penalty skips the bias. Built once as a mask rather than by slicing
    # each pass, because in-place slice assignment is the one numpy idiom that
    # does not carry cleanly to a device array.
    penalised = module.ones(features.shape[1], dtype=module.float64)
    penalised[-1] = 0.0

    for _ in range(epochs):
        error = _sigmoid(features @ weights, module) - target
        gradient = features.T @ error / len(features)
        gradient = gradient + l2 * weights * penalised
        weights = weights - lr * gradient

    # Back to numpy on the way out: everything downstream is pandas, safetensors
    # and scoring, none of which knows what a device array is.
    return Logistic(weights=arrays_mod.to_cpu(weights))


@dataclasses.dataclass
class MLP:
    """The same shape as the job that goes to HelloWorldAi, run here."""

    layers: list
    # Where early stopping settled, and what it settled on. Recorded so that a
    # run says how much training it actually used rather than how much it was
    # allowed.
    stopped_at: int = 0
    validation_loss: float | None = None

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        activation = np.asarray(x, dtype=np.float64)
        for index, (weight, bias) in enumerate(self.layers):
            activation = activation @ weight + bias
            if index < len(self.layers) - 1:
                activation = np.maximum(activation, 0.0)
        return _sigmoid(activation.ravel())


def network_health(network: MLP, x: np.ndarray, sample: int = 20000) -> dict:
    """Whether this network is still capable of answering different things.

    A ReLU whose pre-activation is negative for every input in the data outputs
    zero for every input, and its gradient is zero too -- so it never comes back.
    When a whole layer dies the network collapses to its output bias: one
    probability, the same for every row, forever. Training longer makes it
    worse, not better, because there is nothing left to update.

    This is not a hypothetical. At `lr=0.01` -- the default this project shipped
    with, on both backends -- all sixty-four units of the second hidden layer
    died on every seed tried, and the model returned a constant 0.516992. The
    gate then reported "it has learned nothing", which reads as a statement
    about the market and was a statement about the optimiser.
    """
    rows = np.asarray(x, dtype=np.float64)[:sample]
    if not len(rows):
        return {"checked_rows": 0}

    activation = rows
    dead_per_layer = []
    for weight, bias in network.layers[:-1]:
        z = activation @ weight + bias
        alive = (z > 0).any(axis=0)
        dead_per_layer.append(int((~alive).sum()))
        activation = np.maximum(z, 0.0)

    probabilities = network.probabilities(rows)
    width = [len(w[1]) for w in [(None, b) for _, b in network.layers[:-1]]]

    return {
        "checked_rows": int(len(rows)),
        "dead_units": dead_per_layer,
        "hidden_width": width,
        # A layer entirely dead means the network cannot vary at all.
        "collapsed": any(d == w for d, w in zip(dead_per_layer, width)),
        "probability_std": round(float(probabilities.std()), 8),
        "probability_range": [round(float(probabilities.min()), 6),
                              round(float(probabilities.max()), 6)],
    }


def _cross_entropy(probabilities, target, xp=np) -> float:
    safe = xp.clip(probabilities, 1e-12, 1 - 1e-12)
    loss = -(target * xp.log(safe) + (1 - target) * xp.log(1 - safe)).mean()
    # A scalar, on the host, so the training loop can compare it to a float.
    return float(arrays_mod.to_cpu(loss))


def fit_mlp(x: np.ndarray, y: np.ndarray, *, hidden: int = 64, depth: int = 2,
            steps: int = 4000, batch: int = 64, lr: float = 0.001,
            seed: int = 0, validation: float = 0.15,
            patience: int = 8, check_every: int = 1000,
            device: str | None = None) -> MLP:
    """A small ReLU network trained with Adam, in numpy.

    The learning rate is Adam's own default and was not always. At 0.01 this
    network dies: see `network_health` for what that means and how it was
    found. Raising it is not a free knob -- check the health report.

    Counted in **gradient steps, not epochs**, because that is how the job
    submitted to HelloWorldAi is counted -- `steps=4000, batch_size=64`. Passing
    the same numbers means the local control does the same amount of learning as
    the remote model rather than an amount that happens to depend on how many
    symbols are in the panel. Two consequences, both wanted: the comparison is
    fair, and a run on 240 symbols costs the same as one on 10 instead of
    twenty-four times as much. Epoch-counting made the controls take minutes on
    a wide panel, which is long enough that somebody would turn them off.

    **It stops itself.** `steps` is a ceiling, not an instruction. The last
    `validation` fraction of the training rows is held back -- chronologically,
    because `combine` leaves them in date order, so the slice is the most recent
    part of the training period and the nearest thing available to a rehearsal
    of the test set. Training checks against it every `check_every` steps, keeps
    the weights that scored best, and gives up after `patience` checks without
    improvement.

    That is not a refinement, it is the difference between a model and a
    memorised table. Measured on the wide panel at a fixed step count:

        steps    passes   train acc   test acc
        5,000       0.7      0.5194     0.5118
        20,000      3.0      0.5337     0.5108
        80,268     12.0      0.5564     0.5066
        200,000    29.9      0.5690     0.5029

    Training accuracy climbs the whole way and out-of-sample accuracy falls the
    whole way. More training makes it strictly worse, so "train for longer" and
    "loop until it is smarter" both have the same answer, and it is no.

    The stopping point has to be chosen on the validation slice rather than on
    the test set, even though the test set would pick a better one. Reading the
    test set to decide a hyperparameter is how the one number that has to stay
    clean gets spent.

    Deliberately not torch. The project refuses that dependency to keep every
    arithmetic step readable, and a control that needed two gigabytes of CUDA to
    run would not be much of a control.
    """
    # The random parts stay on numpy on purpose -- the initial weights and the
    # shuffling order. CuPy's generator produces a different stream for the same
    # seed, so drawing them on the device would make a GPU run and a CPU run of
    # the same seed two different experiments, and the noise floor would be
    # measuring the backend as much as the seed. Drawn here, transferred once,
    # identical either way.
    # One network at batch 64 belongs on the CPU and the measurements are not
    # close: 7.4 seconds here against 67 on the card, because twenty thousand
    # steps of thirty kernel launches each cost more than the arithmetic saves.
    #
    # This has to be an explicit default rather than "auto", which resolves to
    # the card whenever one exists. Leaving it on auto made a single unit test
    # take 133 seconds instead of 8, and it would have made every ordinary run
    # slower on the machine that had just been given a GPU to speed it up.
    # `fit_mlp_ensemble` is where a card earns its place.
    module = arrays_mod.xp(device or CPU)
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    # The validation slice comes off the end, which is the most recent stretch
    # of the training period. Taking it at random would let the model choose its
    # stopping point using days it had memorised the neighbours of.
    holdout = int(len(x) * max(min(validation, 0.5), 0.0))
    if holdout > 50:
        x_fit, y_fit = x[:-holdout], y[:-holdout]
        x_val, y_val = x[-holdout:], y[-holdout:]
    else:
        x_fit, y_fit, x_val, y_val = x, y, None, None

    sizes = [x_fit.shape[1]] + [hidden] * depth + [1]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        # He initialisation, which is what ReLU wants. Drawn on numpy, then
        # moved, so the starting point does not depend on where it runs.
        layers.append([module.asarray(rng.normal(0.0, np.sqrt(2.0 / a), (a, b))),
                       module.zeros(b, dtype=module.float64)])

    # Everything the training loop touches lives on the device for the duration.
    x_fit = module.asarray(x_fit)
    y_fit = module.asarray(y_fit)
    if x_val is not None:
        x_val = module.asarray(x_val)
        y_val = module.asarray(y_val)

    moment1 = [[module.zeros_like(w), module.zeros_like(b)] for w, b in layers]
    moment2 = [[module.zeros_like(w), module.zeros_like(b)] for w, b in layers]

    order = module.asarray(rng.permutation(len(x_fit)))
    cursor = 0

    best_loss, best_layers, best_step, stale = np.inf, None, 0, 0

    for step in range(1, max(steps, 1) + 1):
        # Draw the next minibatch, reshuffling when the pass runs out. Sampling
        # without replacement within a pass keeps the gradient estimates less
        # correlated than drawing independently every time.
        if cursor + batch > len(order):
            order = module.asarray(rng.permutation(len(x_fit)))
            cursor = 0
        index = order[cursor:cursor + batch]
        cursor += batch

        xb, yb = x_fit[index], y_fit[index]

        activations = [xb]
        for depth_index, (weight, bias) in enumerate(layers):
            z = activations[-1] @ weight + bias
            activations.append(z if depth_index == len(layers) - 1
                               else module.maximum(z, 0.0))

        # Cross-entropy through a sigmoid differentiates to exactly this.
        delta = (_sigmoid(activations[-1], module) - yb) / len(xb)

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
                param -= lr * corrected1 / (module.sqrt(corrected2) + 1e-8)

        # --- stop when it stops helping ---------------------------------
        if x_val is not None and (step % check_every == 0 or step == steps):
            activation = x_val
            for depth_index, (weight, bias) in enumerate(layers):
                activation = activation @ weight + bias
                if depth_index < len(layers) - 1:
                    activation = module.maximum(activation, 0.0)
            loss = _cross_entropy(_sigmoid(activation.ravel(), module),
                                  y_val.ravel(), module)

            if loss < best_loss - 1e-6:
                best_loss, best_step, stale = loss, step, 0
                # Copied, because `layers` keeps being updated in place.
                best_layers = [(w.copy(), b.copy()) for w, b in layers]
            else:
                stale += 1
                if stale >= patience:
                    logger.info(
                        "Stopped at %d of %d steps; best validation loss "
                        "%.5f at step %d", step, steps, best_loss, best_step)
                    break

    # The weights that scored best, not the ones it happened to end on.
    if best_layers is not None:
        layers = best_layers

    # Home before anything downstream sees them.
    layers = [(arrays_mod.to_cpu(w), arrays_mod.to_cpu(b)) for w, b in layers]

    return MLP(layers=layers,
               stopped_at=best_step or steps,
               validation_loss=(None if best_loss == np.inf
                                else round(best_loss, 6)))


@dataclasses.dataclass
class Majority:
    """Answer the training majority, every time. The floor."""

    answer: int

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(x)), 0.99 if self.answer else 0.01)


# --- running them ----------------------------------------------------------

def _score(probabilities, splits, scaler, train_up_share, cost) -> dict:
    test = dataset_mod.test_matrix(splits, scaler)
    result = evaluate_mod.evaluate(
        probabilities, test.y, test.returns, test.dates, test.symbols,
        executable_returns=test.executable,
        train_up_share=train_up_share, cost=cost, horizon=test.horizon)
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
    x_test = dataset_mod.test_matrix(splits, scaler).x
    train_up_share = float(y_train.mean())

    out: dict = {"train_up_share": round(train_up_share, 4),
                 "train_rows": int(len(y_train)),
                 # Where the arithmetic actually ran. Two runs whose numbers
                 # differ are only comparable if this matches, and a card that
                 # silently fell back to the CPU is worth knowing about.
                 "device": arrays_mod.device_name(),
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

    # The seeds are trained together rather than one after another. At three
    # models that is a small win on the CPU; the reason it is written this way
    # is that a search over dozens of configurations is the same call with a
    # longer list, and there it is the difference between a queue and one pass.
    models = fit_mlp_ensemble(x_train, y_train, seeds=list(range(seeds)),
                              hidden=hidden, depth=depth, steps=local_steps,
                              batch=batch)

    accuracies = []
    for seed, model in enumerate(models):
        scored = _score(model.probabilities(x_test), splits, scaler,
                        train_up_share, cost)
        accuracies.append(scored["accuracy"])
        if seed == 0:
            out["local_mlp"] = scored
            # Recorded so that a dead network is diagnosable from the run file
            # rather than only from its suspiciously round accuracy.
            out["local_mlp_health"] = network_health(model, x_train)

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
            x_test = dataset_mod.test_matrix(splits, scaler).x

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
                "executable_sharpe": scored.get("executable_sharpe"),
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
    # Replaced rather than rebuilt field by field, so that a field added to
    # Split later -- as the horizon was -- survives the fold instead of quietly
    # falling back to its default.
    return dataclasses.replace(
        split,
        x_test=split.x_test[keep], y_test=split.y_test[keep],
        test_dates=split.test_dates[keep],
        forward_returns_test=split.forward_returns_test[keep],
        executable_returns_test=split.executable_returns_test[keep],
    )


# --- many models at once ----------------------------------------------------

def fit_mlp_ensemble(x: np.ndarray, y: np.ndarray, *, seeds: Sequence[int],
                     hidden: int = 64, depth: int = 2, steps: int = 4000,
                     batch: int = 64, lr: float = 0.001,
                     validation: float = 0.15, patience: int = 8,
                     check_every: int = 1000,
                     device: str | None = None) -> list:
    """Train several networks simultaneously, as one batched computation.

    This is the only shape of this problem a GPU is actually good at, and the
    measurements say so plainly. One network at batch 64 is 20,000 steps of
    about thirty kernel launches each, and the launches cost more than the
    arithmetic: on this machine a single `fit_mlp` takes 7.4 seconds on the CPU
    and 67 on the card. The card is not slow, it is idle -- waiting between
    tiny instructions.

    Give it a leading model axis and nothing changes except the amount of work
    per launch. Every network gets its own initialisation, its own shuffle and
    its own minibatch; the forward pass becomes one batched matmul of
    (models, batch, in) by (models, in, out); the launch count stays where it
    was. Thirty-two models cost barely more than one.

    That is what makes an hour of searching worth running. A hundred
    configurations trained one after another is a queue; trained together it is
    one pass, and the difference is the whole reason to have a card in this
    project at all.

    Returns one MLP per seed, in the order given, each already back on the host.
    """
    count = len(seeds)
    if device is None:
        # Routed on the measured crossover, not on whether a card exists.
        device = (GPU if count >= GPU_MODEL_CROSSOVER and arrays_mod.available()
                  else CPU)
    module = arrays_mod.xp(device)
    if not count:
        return []

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    holdout = int(len(x) * max(min(validation, 0.5), 0.0))
    if holdout > 50:
        x_fit, y_fit = x[:-holdout], y[:-holdout]
        x_val, y_val = module.asarray(x[-holdout:]), module.asarray(y[-holdout:])
    else:
        x_fit, y_fit, x_val, y_val = x, y, None, None

    generators = [np.random.default_rng(seed) for seed in seeds]
    sizes = [x_fit.shape[1]] + [hidden] * depth + [1]

    # One stack per layer: (models, in, out). Initialised per model on numpy so
    # a seed means the same thing here as it does in fit_mlp.
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        weights = np.stack([g.normal(0.0, np.sqrt(2.0 / a), (a, b))
                            for g in generators])
        layers.append([module.asarray(weights),
                       module.zeros((count, 1, b), dtype=module.float64)])

    x_fit_d = module.asarray(x_fit)
    y_fit_d = module.asarray(y_fit)

    moment1 = [[module.zeros_like(w), module.zeros_like(b)] for w, b in layers]
    moment2 = [[module.zeros_like(w), module.zeros_like(b)] for w, b in layers]

    orders = [g.permutation(len(x_fit)) for g in generators]
    cursor = 0

    best_loss = np.full(count, np.inf)
    best_layers = [None] * count
    best_step = np.zeros(count, dtype=int)
    stale = np.zeros(count, dtype=int)

    for step in range(1, max(steps, 1) + 1):
        if cursor + batch > len(x_fit):
            orders = [g.permutation(len(x_fit)) for g in generators]
            cursor = 0
        # (models, batch) indices -> (models, batch, features)
        index = module.asarray(
            np.stack([o[cursor:cursor + batch] for o in orders]))
        cursor += batch

        xb = x_fit_d[index]
        yb = y_fit_d[index]

        activations = [xb]
        for depth_index, (weight, bias) in enumerate(layers):
            z = module.matmul(activations[-1], weight) + bias
            activations.append(z if depth_index == len(layers) - 1
                               else module.maximum(z, 0.0))

        delta = (_sigmoid(activations[-1], module) - yb) / batch

        for depth_index in range(len(layers) - 1, -1, -1):
            weight, bias = layers[depth_index]
            previous = activations[depth_index]

            grad_w = module.matmul(module.swapaxes(previous, 1, 2), delta)
            grad_b = delta.sum(axis=1, keepdims=True)

            if depth_index > 0:
                delta = (module.matmul(delta, module.swapaxes(weight, 1, 2))
                         * (activations[depth_index] > 0))

            for slot, (grad, param) in enumerate(
                    ((grad_w, weight), (grad_b, bias))):
                moment1[depth_index][slot] = (
                    0.9 * moment1[depth_index][slot] + 0.1 * grad)
                moment2[depth_index][slot] = (
                    0.999 * moment2[depth_index][slot] + 0.001 * grad * grad)
                corrected1 = moment1[depth_index][slot] / (1 - 0.9 ** step)
                corrected2 = moment2[depth_index][slot] / (1 - 0.999 ** step)
                param -= lr * corrected1 / (module.sqrt(corrected2) + 1e-8)

        # --- stop each model when it stops helping, independently ---------
        if x_val is not None and (step % check_every == 0 or step == steps):
            activation = module.broadcast_to(
                x_val, (count,) + x_val.shape).copy()
            for depth_index, (weight, bias) in enumerate(layers):
                activation = module.matmul(activation, weight) + bias
                if depth_index < len(layers) - 1:
                    activation = module.maximum(activation, 0.0)

            probabilities = _sigmoid(activation[..., 0], module)
            safe = module.clip(probabilities, 1e-12, 1 - 1e-12)
            target = y_val.ravel()
            losses = arrays_mod.to_cpu(
                -(target * module.log(safe)
                  + (1 - target) * module.log(1 - safe)).mean(axis=1))

            for model in range(count):
                if losses[model] < best_loss[model] - 1e-6:
                    best_loss[model] = losses[model]
                    best_step[model] = step
                    stale[model] = 0
                    best_layers[model] = [
                        (arrays_mod.to_cpu(w[model]).copy(),
                         arrays_mod.to_cpu(b[model]).reshape(-1).copy())
                        for w, b in layers]
                else:
                    stale[model] += 1

            # Only stop when every model has given up; the batched step costs
            # the same whether one model still needs it or all of them do.
            if (stale >= patience).all():
                logger.info("Ensemble stopped at %d steps; best losses %s",
                            step, np.round(best_loss, 5).tolist())
                break

    out = []
    for model in range(count):
        chosen = best_layers[model]
        if chosen is None:
            chosen = [(arrays_mod.to_cpu(w[model]).copy(),
                       arrays_mod.to_cpu(b[model]).reshape(-1).copy())
                      for w, b in layers]
        out.append(MLP(layers=chosen,
                       stopped_at=int(best_step[model]) or steps,
                       validation_loss=(None if not np.isfinite(best_loss[model])
                                        else round(float(best_loss[model]), 6))))
    return out
