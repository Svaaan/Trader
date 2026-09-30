"""Reading a model bundle without torch, and refusing what it cannot run.

The loader in src/trader/model.py rebuilds the network from the manifest and
multiplies the matrices itself. That is only safe if it reads the format the
producer actually writes -- so these build a bundle with the producer's own
code and load it back.

There used to be a second producer behind a network, and this file checked the
format against it. It is gone, and `trainer.pack_bundle` is now the only thing
that writes a bundle. That makes the round trip here tighter rather than looser:
pack and load are the same repository and must agree exactly, including the
softmax conversion in the last layer, which is wrong by a factor of two in the
obvious implementation and does not raise when it is.

A fixture written by hand would encode my belief about the format and would
keep passing after the format changed, so the format checks use the packer and
only the arithmetic checks build bytes directly.
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import model as model_mod          # noqa: E402
from trader import trainer as trainer_mod      # noqa: E402

# --- the real format -------------------------------------------------------

def a_trained_network(input_dim=9, hidden=8, seed=3):
    """Layers in the shape `baseline.fit_mlp` returns them: (in, out), one logit.

    Deliberately not square, so a transpose mistake is a shape error rather
    than silently wrong numbers.
    """
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(0, 0.3, (input_dim, hidden)).astype(np.float32),
         rng.normal(0, 0.1, hidden).astype(np.float32)),
        (rng.normal(0, 0.3, (hidden, 1)).astype(np.float32),
         rng.normal(0, 0.1, 1).astype(np.float32)),
    ]


def test_a_real_bundle_loads_and_predicts():
    names = [f"f{i}" for i in range(9)]
    blob = trainer_mod.pack_bundle(a_trained_network(), names)

    loaded = model_mod.load_bundle(blob)

    assert loaded.input_dim == 9
    assert loaded.class_names == ["down", "up"]
    assert loaded.feature_names == names

    probabilities = loaded.probabilities(np.zeros((4, 9), dtype=np.float32))
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_the_bundle_still_holds_what_the_loader_looks_for():
    """A rename on the producer side should fail here, not in production."""
    blob = trainer_mod.pack_bundle(a_trained_network(),
                                   [f"f{i}" for i in range(9)])

    names = set(zipfile.ZipFile(io.BytesIO(blob)).namelist())
    assert model_mod.WEIGHTS_NAME in names
    assert model_mod.CONFIG_NAME in names


# --- the arithmetic --------------------------------------------------------

def make_bundle(manifest: dict, state: dict) -> bytes:
    """A bundle assembled here, for cases the producer will not make."""
    from safetensors.numpy import save

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(model_mod.WEIGHTS_NAME, save(state))
        archive.writestr(model_mod.CONFIG_NAME, json.dumps(manifest))
    return buffer.getvalue()


def test_the_forward_pass_matches_the_arithmetic_by_hand():
    """One layer, known numbers, no framework to hide behind."""
    state = {
        "net.0.weight": np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32),
        "net.0.bias": np.array([0.5, -0.5], dtype=np.float32),
    }
    manifest = {"architecture": "mlp",
                "modules": [{"type": "Linear", "in_features": 2, "out_features": 2}]}

    loaded = model_mod.load_bundle(make_bundle(manifest, state))
    out = loaded.forward(np.array([[3.0, 4.0]], dtype=np.float32))

    # [3,4] . [1,2] + 0.5 = 11.5 ;  [3,4] . [0,-1] - 0.5 = -4.5
    assert np.allclose(out, [[11.5, -4.5]]), out


def test_relu_is_applied_between_layers():
    state = {
        "net.0.weight": np.array([[1.0], [-1.0]], dtype=np.float32),
        "net.0.bias": np.zeros(2, dtype=np.float32),
        "net.2.weight": np.array([[1.0, 1.0]], dtype=np.float32),
        "net.2.bias": np.zeros(1, dtype=np.float32),
    }
    manifest = {"architecture": "mlp", "modules": [
        {"type": "Linear", "in_features": 1, "out_features": 2},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 2, "out_features": 1},
    ]}

    loaded = model_mod.load_bundle(make_bundle(manifest, state))

    # Input 5 -> [5, -5] -> ReLU -> [5, 0] -> sum = 5. Without the ReLU it is 0,
    # so this distinguishes the two.
    assert np.allclose(loaded.forward(np.array([[5.0]], dtype=np.float32)), [[5.0]])


def test_probabilities_do_not_overflow_on_confident_scores():
    """exp(1000) is inf, and inf/inf is nan -- a signal that is silently absent."""
    state = {
        "net.0.weight": np.array([[1000.0], [-1000.0]], dtype=np.float32),
        "net.0.bias": np.zeros(2, dtype=np.float32),
    }
    manifest = {"architecture": "mlp",
                "modules": [{"type": "Linear", "in_features": 1, "out_features": 2}]}

    loaded = model_mod.load_bundle(make_bundle(manifest, state))
    out = loaded.probabilities(np.array([[5.0]], dtype=np.float32))

    assert np.isfinite(out).all()
    assert np.allclose(out.sum(), 1.0)


# --- refusing rather than guessing -----------------------------------------

def test_a_transformer_bundle_is_refused():
    """It would produce numbers, and they would not mean anything."""
    manifest = {"architecture": "transformer", "modules": [{"type": "Linear",
                "in_features": 2, "out_features": 2}]}
    state = {"net.0.weight": np.zeros((2, 2), dtype=np.float32)}

    with pytest.raises(model_mod.ModelError, match="transformer"):
        model_mod.load_bundle(make_bundle(manifest, state))


def test_an_unknown_layer_is_refused():
    manifest = {"architecture": "mlp", "modules": [
        {"type": "Linear", "in_features": 2, "out_features": 2},
        {"type": "BatchNorm1d"},
    ]}
    state = {"net.0.weight": np.zeros((2, 2), dtype=np.float32)}

    with pytest.raises(model_mod.ModelError, match="BatchNorm1d"):
        model_mod.load_bundle(make_bundle(manifest, state))


def test_the_wrong_number_of_features_is_refused():
    """Columns are positional; the wrong count means the wrong meaning."""
    state = {"net.0.weight": np.zeros((2, 9), dtype=np.float32),
             "net.0.bias": np.zeros(2, dtype=np.float32)}
    manifest = {"architecture": "mlp",
                "modules": [{"type": "Linear", "in_features": 9, "out_features": 2}]}

    loaded = model_mod.load_bundle(make_bundle(manifest, state))

    with pytest.raises(model_mod.ModelError, match="expects 9 features"):
        loaded.forward(np.zeros((1, 5), dtype=np.float32))


def test_something_that_is_not_a_bundle_is_refused():
    with pytest.raises(model_mod.ModelError, match="not a model bundle"):
        model_mod.load_bundle(b"this is not a zip file")
