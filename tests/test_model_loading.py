"""Reading a HelloWorldAi bundle without torch, and refusing what it cannot run.

The loader in src/trader/model.py rebuilds the network from the manifest and
multiplies the matrices itself. That is only safe if it reads the format the
producer actually writes, so these build a bundle with HelloWorldAi's own
packing code and load it back.

Using the real producer is the point. A fixture I wrote by hand would encode my
belief about the format, and would keep passing after the format changed --
which is the failure this is meant to catch.

If HelloWorldAi is not checked out beside this project the format checks skip,
and the arithmetic checks still run.
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

HELLOWORLD_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "HelloWorldAi", "src"))


def helloworld_packers():
    """HelloWorldAi's own manifest and bundle builders, if they are reachable."""
    if not os.path.isdir(HELLOWORLD_SRC):
        pytest.skip(f"HelloWorldAi not found at {HELLOWORLD_SRC}")
    if HELLOWORLD_SRC not in sys.path:
        sys.path.insert(0, HELLOWORLD_SRC)
    try:
        from backend.service.modelBundle import build_bundle
        from backend.service.modelManifest import build_manifest
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"HelloWorldAi's packers are not importable: {exc}")
    return build_manifest, build_bundle


def a_trained_state_dict(input_dim=9, hidden=8, output=2, seed=3):
    """Weights shaped like the ones training produces, in torch's convention.

    Linear weights are (out_features, in_features), which is the transpose of
    what the multiply wants. Getting that backwards produces a shape error on a
    square layer and silently wrong numbers on a rectangular one, so the
    fixture is deliberately not square.
    """
    rng = np.random.default_rng(seed)
    return {
        "net.0.weight": rng.normal(0, 0.3, (hidden, input_dim)).astype(np.float32),
        "net.0.bias": rng.normal(0, 0.1, hidden).astype(np.float32),
        "net.2.weight": rng.normal(0, 0.3, (output, hidden)).astype(np.float32),
        "net.2.bias": rng.normal(0, 0.1, output).astype(np.float32),
    }


# --- the real format -------------------------------------------------------

def test_a_real_bundle_loads_and_predicts():
    build_manifest, build_bundle = helloworld_packers()

    spec = {"architecture": "mlp", "input_dim": 9, "hidden_dim": 8,
            "depth": 1, "output_dim": 2}
    state = a_trained_state_dict()

    manifest = build_manifest(spec, state, model_name="trader-test",
                              class_names=["down", "up"],
                              feature_names=[f"f{i}" for i in range(9)])
    blob = build_bundle(state, manifest, None)

    loaded = model_mod.load_bundle(blob)

    assert loaded.input_dim == 9
    assert loaded.class_names == ["down", "up"]
    assert loaded.feature_names == [f"f{i}" for i in range(9)]

    probabilities = loaded.probabilities(np.zeros((4, 9), dtype=np.float32))
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_the_bundle_still_holds_what_the_loader_looks_for():
    """A rename on the producer side should fail here, not in production."""
    build_manifest, build_bundle = helloworld_packers()

    spec = {"architecture": "mlp", "input_dim": 9, "hidden_dim": 8,
            "depth": 1, "output_dim": 2}
    state = a_trained_state_dict()
    blob = build_bundle(state, build_manifest(spec, state), None)

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
