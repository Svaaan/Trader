"""Running a model HelloWorldAi trained, without torch.

The bundle is a zip holding `model.safetensors` and a `config.json` manifest,
and that manifest lists the layers explicitly:

    [{"type": "Linear", "in_features": 9, "out_features": 64},
     {"type": "ReLU"},
     ...]

It is written that way so a loader can rebuild the network instead of
re-deriving it from depth and width and hoping the two agree. Taking it at its
word means inference here is a few matrix multiplications, which numpy does in
the twenty lines below.

The alternative was importing torch to run nine numbers through two hidden
layers: two gigabytes, a CUDA version to keep in step, and a black box in the
one place this project most needs to be legible. Every arithmetic operation
that turns a price into a signal is visible in this file.

It refuses anything it does not recognise rather than guessing. A transformer
bundle, or a layer type added later, raises instead of silently producing
numbers that look plausible.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from typing import Any

import numpy as np

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

SUPPORTED_ARCHITECTURES = {"mlp"}
SUPPORTED_LAYERS = {"Linear", "ReLU", "GELU", "Tanh"}


class ModelError(Exception):
    """Raised when a bundle cannot be turned into something runnable."""


@dataclasses.dataclass
class Model:
    """A loaded network, plus what it was trained to mean."""

    modules: list[dict]
    weights: dict[str, np.ndarray]
    manifest: dict

    @property
    def class_names(self) -> list[str]:
        names = self.manifest.get("class_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
        return ["down", "up"]

    @property
    def feature_names(self) -> list[str]:
        """What each input column meant, if the trainer recorded it.

        Under `input.names`, not at the top level -- the manifest describes the
        input as an object with a shape and optional names, and only the class
        names sit at the root. Guessing the wrong place returns an empty list
        rather than raising, so a UI would simply show unlabelled columns and
        nobody would notice the labels had been lost.
        """
        names = (self.manifest.get("input") or {}).get("names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]

        # Tolerated at the root too: an older or hand-made bundle may put it
        # there, and the pairing matters more than where it was written.
        names = self.manifest.get("feature_names")
        return [str(n) for n in names] if isinstance(names, list) else []

    @property
    def input_dim(self) -> int:
        for module in self.modules:
            if module.get("type") == "Linear":
                return int(module["in_features"])
        raise ModelError("the manifest describes no Linear layer")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Raw scores, one row per input row.

        Linear layers are stored as (out_features, in_features) -- the torch
        convention -- so the multiply is x @ W.T + b.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.input_dim:
            raise ModelError(
                f"the model expects {self.input_dim} features and was given "
                f"{x.shape[1]}. If the feature set changed, the model has to be "
                f"retrained -- the columns are positional.")

        activation = x
        linear_index = 0

        for module in self.modules:
            kind = module.get("type")

            if kind == "Linear":
                weight, bias = self._linear_parameters(linear_index)
                activation = activation @ weight.T + bias
                linear_index += 1
            elif kind == "ReLU":
                activation = np.maximum(activation, 0.0)
            elif kind == "GELU":
                activation = 0.5 * activation * (
                    1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                  * (activation + 0.044715 * activation ** 3)))
            elif kind == "Tanh":
                activation = np.tanh(activation)
            else:
                raise ModelError(
                    f"unsupported layer {kind!r}. This loader handles "
                    f"{sorted(SUPPORTED_LAYERS)}; rather than approximate it, "
                    f"it stops.")

        return activation

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """Softmax over the classes, numerically stable."""
        scores = self.forward(x)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _linear_parameters(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Find the nth Linear layer's weight and bias, whatever they are called.

        Different exporters name these differently -- `net.0.weight`,
        `layers.0.weight`, `0.weight`. Rather than guess one convention, the
        Linear tensors are taken in the order they appear.
        """
        weights = [(name, array) for name, array in self.weights.items()
                   if name.endswith("weight") and array.ndim == 2]
        biases = [(name, array) for name, array in self.weights.items()
                  if name.endswith("bias") and array.ndim == 1]

        weights.sort(key=lambda item: _order_key(item[0]))
        biases.sort(key=lambda item: _order_key(item[0]))

        if index >= len(weights):
            raise ModelError(
                f"the manifest describes more Linear layers than the weights "
                f"file holds ({len(weights)})")

        name, weight = weights[index]

        if index < len(biases):
            bias = biases[index][1]
        else:
            bias = np.zeros(weight.shape[0], dtype=np.float32)

        if bias.shape[0] != weight.shape[0]:
            raise ModelError(
                f"{name} has {weight.shape[0]} outputs but its bias has "
                f"{bias.shape[0]}")

        return weight.astype(np.float32), bias.astype(np.float32)


def _order_key(name: str) -> tuple:
    """Sort tensor names by the numbers in them, so `net.10` follows `net.9`."""
    parts = []
    number = ""
    for char in name:
        if char.isdigit():
            number += char
        elif number:
            parts.append(int(number))
            number = ""
    if number:
        parts.append(int(number))
    return (tuple(parts), name)


def load_bundle(blob: bytes) -> Model:
    """Open a downloaded bundle and return something that can predict."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise ModelError("that is not a model bundle (not a zip)") from exc

    names = set(archive.namelist())
    for required in (WEIGHTS_NAME, CONFIG_NAME):
        if required not in names:
            raise ModelError(
                f"the bundle has no {required}; it holds {sorted(names)}")

    manifest = json.loads(archive.read(CONFIG_NAME).decode("utf-8"))

    architecture = str(manifest.get("architecture", "")).lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ModelError(
            f"this loader runs {sorted(SUPPORTED_ARCHITECTURES)} and the bundle "
            f"is {architecture!r}. Running it anyway would produce numbers, and "
            f"they would not mean anything.")

    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ModelError("the manifest lists no layers")

    unknown = {m.get("type") for m in modules} - SUPPORTED_LAYERS
    if unknown:
        raise ModelError(f"unsupported layers in the bundle: {sorted(unknown)}")

    from safetensors.numpy import load as load_safetensors
    weights = load_safetensors(archive.read(WEIGHTS_NAME))

    return Model(modules=modules,
                 weights={k: np.asarray(v) for k, v in weights.items()},
                 manifest=manifest)


def load_bundle_file(path: str) -> Model:
    with open(path, "rb") as handle:
        return load_bundle(handle.read())
