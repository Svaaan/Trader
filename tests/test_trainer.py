"""Two backends that have to produce the same kind of thing.

The pair only means something if the artifacts are interchangeable. If the
locally trained model went through a different loader, a different forward pass
or a different evaluator than the one HelloWorldAi returns, then a gap between
their scores would be a fact about this code rather than about the round trip,
and the comparison would be worse than not having it.

So most of what is checked here is sameness: that a packed bundle reproduces the
network that made it exactly, that it loads through the real loader, and that
the two-class conversion is the one that agrees with the sigmoid rather than the
one that looks right.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import baseline, model as model_mod, trainer          # noqa: E402


@pytest.fixture
def separable():
    """Data a small network can actually fit, so the weights mean something."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 9))
    y = ((x[:, 0] + x[:, 1]) > 0).astype(int)
    return x, y


@pytest.fixture
def names():
    return [f"f{i}" for i in range(9)]


# --- the artifact ----------------------------------------------------------

def test_a_packed_bundle_loads_through_the_real_loader(separable, names):
    x, y = separable
    network = baseline.fit_mlp(x, y, hidden=16, depth=2, steps=1000, seed=0)

    loaded = model_mod.load_bundle(trainer.pack_bundle(network.layers, names))

    assert loaded.input_dim == 9
    assert loaded.feature_names == names
    assert loaded.class_names == ["down", "up"]


def test_the_bundle_reproduces_the_network_that_made_it(separable, names):
    """Exactly, not approximately.

    softmax([a0, a1])[1] is sigmoid(a1 - a0), so the "down" row has to be zero
    and the "up" row the trained weights. Anything else is a different model
    wearing the same weights.
    """
    x, y = separable
    network = baseline.fit_mlp(x, y, hidden=16, depth=2, steps=1000, seed=0)

    loaded = model_mod.load_bundle(trainer.pack_bundle(network.layers, names))

    direct = network.probabilities(x[:500])
    through_bundle = loaded.probabilities(x[:500])[:, 1]
    # float32 storage is the only thing allowed to differ.
    assert np.allclose(direct, through_bundle, atol=1e-5)


def test_mirroring_the_final_weights_would_have_been_wrong(separable, names):
    """The bug this file exists to prevent, written down so it stays prevented.

    [-w, +w] is the obvious way to turn one logit into two class scores and it
    doubles the logit: softmax([-z, z])[1] is sigmoid(2z). It raises nothing,
    looks like nothing, and sharpens every probability the model reports.
    """
    x, y = separable
    network = baseline.fit_mlp(x, y, hidden=16, depth=2, steps=1000, seed=0)

    weight, bias = network.layers[-1]
    mirrored = list(network.layers[:-1]) + [(
        np.concatenate([-weight, weight], axis=1),
        np.concatenate([-bias, bias]))]

    # Pack the mirrored version by hand, the way pack_bundle would if it were
    # wrong, and confirm it disagrees enough to matter.
    import io, json, zipfile
    from safetensors.numpy import save

    state, modules = {}, []
    for index, (w, b) in enumerate(mirrored):
        state[f"net.{index * 2}.weight"] = np.ascontiguousarray(w.T.astype(np.float32))
        state[f"net.{index * 2}.bias"] = b.astype(np.float32)
        modules.append({"type": "Linear", "in_features": int(w.shape[0]),
                        "out_features": int(w.shape[1])})
        if index < len(mirrored) - 1:
            modules.append({"type": "ReLU"})

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(model_mod.WEIGHTS_NAME, save(state))
        archive.writestr(model_mod.CONFIG_NAME,
                         json.dumps({"architecture": "mlp", "modules": modules}))

    wrong = model_mod.load_bundle(buffer.getvalue()).probabilities(x[:500])[:, 1]
    direct = network.probabilities(x[:500])

    assert np.max(np.abs(direct - wrong)) > 0.01, (
        "the mirrored packing has to be detectably different, or this test is "
        "not guarding anything")


def test_training_locally_refuses_to_return_a_bundle_that_disagrees(
        separable, names, monkeypatch):
    """The self-check inside train_local has to be able to fire.

    A silent mismatch between the network and its bundle looks exactly like a
    model that learned something slightly different, which cannot be spotted by
    reading the output.
    """
    x, y = separable
    # Captured before patching, or `broken` calls itself forever.
    real_pack = trainer.pack_bundle

    def broken(layers, feature_names, **kwargs):
        weight, bias = layers[-1]
        mangled = list(layers[:-1]) + [(weight * 3.0, bias)]
        return real_pack(mangled, feature_names, **kwargs)

    monkeypatch.setattr(trainer, "pack_bundle", broken)

    with pytest.raises(ValueError, match="disagrees with the network"):
        trainer.train_local(x, y, names,
                            trainer.Hyperparameters(hidden=16, steps=500))


def test_a_locally_trained_bundle_is_a_working_model(separable, names):
    x, y = separable
    blob = trainer.train_local(
        x, y, names, trainer.Hyperparameters(hidden=16, depth=2, steps=3000))

    loaded = model_mod.load_bundle(blob)
    accuracy = ((loaded.probabilities(x)[:, 1] > 0.5) == y).mean()
    assert accuracy > 0.85, "the local trainer produced a model that cannot fit"


def test_the_local_trainer_is_deterministic_for_a_seed(separable, names):
    x, y = separable
    hyper = trainer.Hyperparameters(hidden=16, steps=500, seed=3)
    assert trainer.train_local(x, y, names, hyper) == \
        trainer.train_local(x, y, names, hyper)


# --- the comparison --------------------------------------------------------

def test_a_gap_inside_the_seed_spread_reads_as_agreement():
    out = trainer.compare({"accuracy": 0.515}, {"accuracy": 0.518},
                          noise_floor=0.010)
    assert out["multiples_of_noise"] < 1.0
    assert "agree" in out["reading"]


def test_a_remote_model_much_worse_points_at_the_round_trip():
    out = trainer.compare({"accuracy": 0.540}, {"accuracy": 0.505},
                          noise_floor=0.004)
    assert out["gap"] < 0
    assert out["multiples_of_noise"] > 5
    assert "round trip" in out["reading"]


def test_a_remote_model_much_better_is_treated_as_suspicious():
    """Both saw the same training rows, so an advantage has to come from
    somewhere, and the usual somewhere is having seen more than it should."""
    out = trainer.compare({"accuracy": 0.505}, {"accuracy": 0.560},
                          noise_floor=0.004)
    assert out["gap"] > 0
    assert "suspicious" in out["reading"]


def test_no_noise_floor_means_no_verdict_on_the_gap():
    out = trainer.compare({"accuracy": 0.51}, {"accuracy": 0.55})
    assert "multiples_of_noise" not in out
    assert "nothing to judge" in out["reading"]


def test_comparing_against_nothing_returns_nothing():
    assert trainer.compare({}, {"accuracy": 0.5}) == {}
    assert trainer.compare({"accuracy": 0.5}, {}) == {}
