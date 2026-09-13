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


# --- a dead network is a training failure, not a market finding -------------

def dead_network(width=8, inputs=4):
    """A network whose second hidden layer cannot fire, built by hand.

    Constructed rather than trained into, because the point is to test the
    detector deterministically -- whether a given learning rate kills a given
    dataset is a property of both, and not something a unit test should race.
    """
    first = (np.ones((inputs, width)), np.zeros(width))
    # Every weight into the second layer negative, with a negative bias: no
    # non-negative input can make any of these positive.
    second = (-np.ones((width, width)), -np.ones(width))
    out = (np.zeros((width, 1)), np.array([0.07]))
    return baseline.MLP(layers=[first, second, out])


def test_a_fully_dead_layer_is_detected(separable):
    x, _ = separable
    health = baseline.network_health(dead_network(inputs=9), np.abs(x))

    assert health["dead_units"][1] == health["hidden_width"][1]
    assert health["collapsed"] is True
    assert health["probability_std"] == 0.0


def test_a_working_network_is_not_flagged(separable, names):
    x, y = separable
    network = baseline.fit_mlp(x, y, hidden=16, depth=2, steps=2000, seed=0)
    health = baseline.network_health(network, x)

    assert health["collapsed"] is False
    assert health["probability_std"] > 0.01


def test_training_refuses_to_return_a_collapsed_network(separable, names,
                                                        monkeypatch):
    """The regression test for a constant model read as a market result.

    At lr=0.01 every unit of the second hidden layer died on the real panel and
    the model returned a constant 0.516992. The gate then said "it has learned
    nothing", which is a sentence about the market, and it was a sentence about
    the optimiser.
    """
    x, y = separable
    monkeypatch.setattr(baseline, "fit_mlp",
                        lambda *a, **k: dead_network(inputs=9))

    with pytest.raises(ValueError, match="training collapsed"):
        trainer.train_local(np.abs(x), y, names)


def test_the_collapse_message_blames_the_optimiser_not_the_data(separable,
                                                                names,
                                                                monkeypatch):
    x, y = separable
    monkeypatch.setattr(baseline, "fit_mlp",
                        lambda *a, **k: dead_network(inputs=9))

    with pytest.raises(ValueError) as caught:
        trainer.train_local(np.abs(x), y, names)

    message = str(caught.value)
    assert "learning rate" in message
    assert "Nothing about the market can be concluded" in message


def test_a_constant_model_is_refused_even_without_a_dead_layer(separable, names,
                                                               monkeypatch):
    """Zero output variance is a failure however it was arrived at."""
    x, y = separable

    def flat(*args, **kwargs):
        width, inputs = 8, 9
        return baseline.MLP(layers=[
            (np.ones((inputs, width)), np.zeros(width)),
            (np.ones((width, width)), np.ones(width)),
            # Alive hidden units, but the output ignores every one of them.
            (np.zeros((width, 1)), np.array([0.3])),
        ])

    monkeypatch.setattr(baseline, "fit_mlp", flat)
    with pytest.raises(ValueError, match="constant model"):
        trainer.train_local(np.abs(x), y, names)


def test_the_default_learning_rate_is_adams_not_ten_times_it():
    """0.01 is what shipped, and it killed the network on both backends."""
    import inspect

    from trader import helloworld

    assert inspect.signature(baseline.fit_mlp).parameters["lr"].default == 0.001
    assert trainer.Hyperparameters().learning_rate == 0.001
    assert inspect.signature(
        helloworld.Client.submit).parameters["learning_rate"].default == 0.001


# --- it stops itself --------------------------------------------------------

def overfittable():
    """Enough noise features that a network will memorise if allowed to."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4000, 12))
    # A weak real signal buried in noise: fitting it is possible, memorising
    # the rest is easier, which is the situation early stopping exists for.
    logit = 0.35 * x[:, 0]
    y = (rng.random(4000) < 1 / (1 + np.exp(-logit))).astype(int)
    return x, y


def test_the_step_count_is_a_ceiling_not_an_instruction():
    """Three very different budgets have to reach the same place.

    Measured on the wide panel: ceilings of 20,000, 80,268 and 200,000 steps
    all stop at 5,000 and score identically. If the ceiling still decided the
    outcome, early stopping would not be doing anything.
    """
    x, y = overfittable()

    stops = [baseline.fit_mlp(x, y, hidden=32, steps=steps, seed=0,
                              check_every=250, patience=4).stopped_at
             for steps in (4000, 16000, 40000)]

    assert len(set(stops)) == 1, f"the ceiling still decided the outcome: {stops}"
    assert stops[0] < 4000, "it never stopped early at all"


def test_more_training_does_not_beat_stopping_when_it_should(monkeypatch):
    """The answer to "should it loop until it is smarter" is no, measurably."""
    x, y = overfittable()
    cut = int(len(x) * 0.7)
    x_fit, y_fit, x_out, y_out = x[:cut], y[:cut], x[cut:], y[cut:]

    stopped = baseline.fit_mlp(x_fit, y_fit, hidden=32, steps=40000, seed=0,
                               check_every=250, patience=4)
    ground_on = baseline.fit_mlp(x_fit, y_fit, hidden=32, steps=40000, seed=0,
                                 validation=0.0)

    def accuracy(net):
        return float(((net.probabilities(x_out) > 0.5) == y_out).mean())

    assert accuracy(stopped) >= accuracy(ground_on), (
        "training to the ceiling beat stopping early, which means this test "
        "is not exercising overfitting")


def test_the_weights_kept_are_the_best_ones_not_the_last_ones():
    x, y = overfittable()
    network = baseline.fit_mlp(x, y, hidden=32, steps=40000, seed=0,
                               check_every=250, patience=4)

    # The recorded stopping point is where validation loss was lowest, and it
    # is strictly before training gave up.
    assert 0 < network.stopped_at < 40000
    assert network.validation_loss is not None


def test_the_validation_slice_is_the_most_recent_training_rows():
    """Taken at random it would let the model stop using days it memorised.

    `combine` leaves rows in date order, so the tail is the most recent stretch
    of the training period -- the nearest rehearsal of the test set available
    without touching it.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 6))
    # The last fifth is unlearnable noise; a model validated on it cannot
    # improve there, so it should stop almost immediately.
    y = (x[:, 0] > 0).astype(int)
    y[-400:] = (rng.random(400) > 0.5).astype(int)

    network = baseline.fit_mlp(x, y, hidden=16, steps=20000, seed=0,
                               validation=0.2, check_every=200, patience=3)
    assert network.stopped_at < 20000


def test_stopping_can_be_switched_off():
    x, y = overfittable()
    network = baseline.fit_mlp(x, y, hidden=16, steps=1500, seed=0,
                               validation=0.0)
    assert network.validation_loss is None
    assert network.stopped_at == 1500
