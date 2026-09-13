"""Where a model gets trained, and why there are two of those.

The project started with one answer -- ship the rows to HelloWorldAi, wait, and
collect the weights. That is still here and still the interesting one, because
running a real workload through a distributed trainer is the point of having
built it. But it was the *only* answer, which meant the round trip could not be
checked against anything.

It turns out not to be a compute question at all. The model is 7,233 parameters,
and training it for the full step count takes about twenty-six seconds of numpy
on a laptop CPU. A GPU is idle at this size; the network was carrying thirty
seconds of arithmetic.

So there are two backends, and the useful thing is running **both**:

  local        trains here, in numpy, in about half a minute
  helloworld   uploads, submits, polls, downloads

Both produce the same artifact -- a zip holding `model.safetensors` and a
`config.json` manifest -- so both are loaded by the same `model.load_bundle`,
run through the same numpy forward pass, and scored by the same evaluator on the
same held-out rows. Nothing downstream knows or cares which one it got.

That identity is the whole value of the pair. Train the same rows both ways with
the same hyperparameters and any difference in the result is attributable to the
round trip itself -- placement, their trainer, the holdout they carve out, the
weight serialisation -- rather than to the data. Before this there was no way to
tell a broken distributed trainer from a dataset with nothing in it, and the
project spent nine runs unable to ask.

The comparison has a scale, too. Two models of the same shape on the same rows
differ by the seed alone, and `baseline` already measures how much that is worth
-- the noise floor. A gap inside it is two draws from one distribution. A gap
several times larger is the round trip doing something.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import zipfile

import numpy as np

from . import baseline as baseline_mod
from . import model as model_mod

logger = logging.getLogger(__name__)

LOCAL = "local"
HELLOWORLD = "helloworld"
BOTH = "both"
BACKENDS = (LOCAL, HELLOWORLD, BOTH)


@dataclasses.dataclass
class Hyperparameters:
    """What both backends are told, so that both are told the same thing."""

    hidden: int = 64
    depth: int = 2
    steps: int = 4000
    batch: int = 64
    learning_rate: float = 0.001
    seed: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def pack_bundle(layers, feature_names, *, class_names=("down", "up")) -> bytes:
    """A locally trained network, in the shape HelloWorldAi returns one.

    Two conversions happen here and both matter.

    **Layout.** `baseline.fit_mlp` holds each Linear as (in, out) because that is
    how the forward pass reads most naturally in numpy. Torch stores (out, in),
    the manifest describes it that way, and `model.py` multiplies by `W.T` on
    that assumption. Transposing here rather than special-casing the loader keeps
    one convention on the wire.

    **Two outputs, not one.** The local network ends in a single logit through a
    sigmoid; the bundle format ends in two class scores through a softmax. These
    agree only for a particular pair: softmax([a0, a1])[1] is sigmoid(a1 - a0),
    so setting the "down" row to zero and the "up" row to the trained weights
    reproduces the sigmoid exactly. Mirroring the weights instead -- [-w, +w] --
    is the obvious thing to write and is wrong by a factor of two in the logit,
    which does not raise, does not look wrong, and quietly sharpens every
    probability the model reports. There is a test for this.
    """
    state: dict = {}
    modules: list = []

    for index, (weight, bias) in enumerate(layers):
        last = index == len(layers) - 1

        if last:
            # One logit becomes two class scores. See the docstring.
            weight = np.concatenate([np.zeros_like(weight), weight], axis=1)
            bias = np.concatenate([np.zeros_like(bias), bias])

        state[f"net.{index * 2}.weight"] = np.ascontiguousarray(
            weight.T.astype(np.float32))
        state[f"net.{index * 2}.bias"] = bias.astype(np.float32)

        modules.append({"type": "Linear",
                        "in_features": int(weight.shape[0]),
                        "out_features": int(weight.shape[1])})
        if not last:
            modules.append({"type": "ReLU"})

    manifest = {
        "architecture": "mlp",
        "modules": modules,
        "class_names": list(class_names),
        # Under `input.names`, which is where model.py looks first and where
        # HelloWorldAi puts it. The pairing of column to meaning is positional
        # and survives the round trip only because it is written down.
        "input": {"names": list(feature_names),
                  "shape": [int(layers[0][0].shape[0])]},
        "trained_by": "trader-local",
    }

    from safetensors.numpy import save

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in ((model_mod.WEIGHTS_NAME, save(state)),
                              (model_mod.CONFIG_NAME,
                               json.dumps(manifest, indent=2).encode("utf-8"))):
            # A fixed timestamp, because `writestr` otherwise stamps the clock
            # into the archive and two identical trainings a second apart
            # produce different bytes. The same weights should give the same
            # artifact -- that is what makes a bundle comparable to another one
            # rather than merely similar to it.
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, payload)

    return buffer.getvalue()


def train_local(x: np.ndarray, y: np.ndarray, feature_names,
                hyper: Hyperparameters | None = None) -> bytes:
    """Train here and return a bundle, indistinguishable from a collected one.

    Deliberately the same `fit_mlp` the controls use. A local trainer that was a
    second implementation of the control would make the comparison between them
    meaningless -- they have to be the same arithmetic for "the remote model
    scored worse than the local one" to say anything about the remote model.
    """
    hyper = hyper or Hyperparameters()

    logger.info("Training locally: %d rows, %d steps, %d wide, %d deep, lr %g",
                len(y), hyper.steps, hyper.hidden, hyper.depth,
                hyper.learning_rate)

    network = baseline_mod.fit_mlp(
        x, y, hidden=hyper.hidden, depth=hyper.depth, steps=hyper.steps,
        batch=hyper.batch, lr=hyper.learning_rate, seed=hyper.seed)

    # A dead network is a training failure and has to be raised as one. Left to
    # run, it returns one constant probability, the gate calls it "the model
    # answers the same way almost every day, it has learned nothing", and a
    # broken optimiser gets recorded as a fact about the market. That is the
    # exact misattribution this project exists to prevent, and it happened.
    health = baseline_mod.network_health(network, x)
    if health.get("collapsed"):
        raise ValueError(
            f"training collapsed: {health['dead_units']} of "
            f"{health['hidden_width']} hidden units are dead, so the network "
            f"returns a constant {health['probability_range'][0]}. This is the "
            f"optimiser, not the data -- a learning rate of "
            f"{hyper.learning_rate:g} is too high for this network. Nothing "
            f"about the market can be concluded from it.")
    if health.get("probability_std", 1.0) < 1e-6:
        raise ValueError(
            f"training produced a constant model (probability std "
            f"{health['probability_std']:.2e}) without any layer being fully "
            f"dead. Still a training failure, not a result.")

    logger.info("  health: %s dead of %s, probability std %.5f",
                health["dead_units"], health["hidden_width"],
                health["probability_std"])

    bundle = pack_bundle(network.layers, feature_names)

    # Cheap insurance against the layout and sigmoid conversions above: load the
    # bundle back through the real loader and check it says what the network
    # said. A silent mismatch here would look exactly like a model that learned
    # something slightly different, which is unfalsifiable by inspection.
    check = model_mod.load_bundle(bundle)
    sample = x[:256]
    if len(sample):
        direct = network.probabilities(sample)
        loaded = check.probabilities(sample)[:, 1]
        drift = float(np.max(np.abs(direct - loaded)))
        if drift > 1e-4:
            raise ValueError(
                f"the packed bundle disagrees with the network that produced it "
                f"by {drift:.2e} -- the weight layout or the two-class "
                f"conversion is wrong")

    return bundle


def compare(local: dict, remote: dict, noise_floor: float | None = None) -> dict:
    """What the gap between the two backends means, in units of the noise floor.

    Both are the same architecture on the same rows, so the only thing that
    *should* separate them is the random seed -- and `baseline` measures how much
    that is worth. Reporting the gap as a multiple of that is the difference
    between "0.6 points" and "eight times what a seed can do".
    """
    if not local or not remote:
        return {}

    gap = float(remote.get("accuracy") or 0.0) - float(local.get("accuracy") or 0.0)
    out = {
        "local_accuracy": local.get("accuracy"),
        "remote_accuracy": remote.get("accuracy"),
        "gap": round(gap, 4),
        "noise_floor": noise_floor,
    }

    if not noise_floor:
        out["reading"] = (
            "No noise floor was measured, so there is nothing to judge the gap "
            "against.")
        return out

    multiples = abs(gap) / noise_floor
    out["multiples_of_noise"] = round(multiples, 2)

    if multiples <= 1.0:
        out["reading"] = (
            f"The two agree to within the seed spread ({noise_floor * 100:.2f} "
            f"points). The round trip is returning what training here returns, "
            f"which is what it should do.")
    elif gap < 0:
        out["reading"] = (
            f"The returned model is {abs(gap) * 100:.2f} points *worse* than the "
            f"same architecture trained here on the same rows -- "
            f"{multiples:.1f}x the seed spread. That is a question about the "
            f"round trip rather than about the data: placement, the "
            f"hyperparameters that actually arrived, or the weights that came "
            f"back.")
    else:
        out["reading"] = (
            f"The returned model is {gap * 100:.2f} points *better* than the "
            f"same architecture trained here on the same rows -- "
            f"{multiples:.1f}x the seed spread. Worth being suspicious of "
            f"rather than pleased about: both saw the same training rows, so a "
            f"real advantage has to come from somewhere, and the usual "
            f"somewhere is having seen more than it should.")

    return out
