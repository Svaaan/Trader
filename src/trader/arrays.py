"""Which array library the arithmetic runs on, and how to stop caring.

The project refuses torch on purpose: the whole argument of `model.py` is that
every operation turning a price into a signal should be visible, and two
gigabytes of framework in the middle of that is the opposite. That argument is
about *readability*, though, not about hardware -- and it does not rule out
running the same readable operations on a GPU.

CuPy is numpy's API reimplemented on CUDA. `x @ w + b`, `np.maximum(z, 0)`,
`rng.permutation(n)` all mean the same thing and run on the card. So the trainer
does not get a GPU branch; it gets an array module, and the same lines execute
in both places. There is one implementation of the arithmetic, which is the
property that made "no torch" worth having in the first place.

**What it is actually worth.** Measured on this machine, one gradient step of
the 7,233-parameter network:

    batch      CPU        GPU floor (~30 kernel launches)
        64     0.53 ms    0.21 ms
     1,024     4.26 ms    0.21 ms
     8,192    39.36 ms    0.21 ms

So even at batch 64 the card is ahead, and by batch 1,024 it is twenty times
ahead. But training one model takes about six seconds of a four-minute run, so
making it a hundred times faster saves six seconds and nothing else.

The reason to have this is *many* models at once -- the seed ensemble, the
walk-forward folds, a configuration search. Those are the same small network
repeated, which is one batched matmul on a card and a queue on a CPU. That is
where the twenty becomes real.

Nothing requires it. With no CuPy installed, or no CUDA, or `TRADER_DEVICE=cpu`,
everything here returns numpy and the project behaves exactly as before.
"""

from __future__ import annotations

import functools
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

CPU = "cpu"
GPU = "gpu"

# cpu | gpu | auto. "auto" uses the card when there is one and says so once.
DEVICE = os.environ.get("TRADER_DEVICE", "auto").strip().lower()


@functools.lru_cache(maxsize=1)
def _cupy():
    """CuPy, if it is installed and a working device is actually attached.

    Importing CuPy succeeds on a machine with no GPU and fails later, in the
    middle of training, with something obscure. So the probe runs a real kernel
    -- a tiny one -- and anything that goes wrong here means CPU.
    """
    try:
        import cupy
    except ImportError:
        return None

    try:
        count = cupy.cuda.runtime.getDeviceCount()
        if not count:
            return None
        # Force a compile and a copy back. A driver or header problem shows up
        # on this line rather than four minutes into a run.
        if int((cupy.arange(4) * 2).sum().get()) != 12:
            return None
        return cupy
    except Exception as exc:                            # noqa: BLE001
        logger.info("CuPy is installed but unusable, staying on the CPU: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """Whether a usable GPU backend exists at all."""
    return _cupy() is not None


@functools.lru_cache(maxsize=1)
def device_name() -> str:
    """Something to print, so a run records where it actually ran."""
    module = _cupy()
    if module is None:
        return "cpu (numpy)"
    try:
        name = module.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return f"gpu ({name})"
    except Exception:                                   # noqa: BLE001
        return "gpu"


def xp(prefer: str | None = None):
    """The array module to use: cupy when asked for and usable, else numpy.

    `prefer` overrides TRADER_DEVICE for one call, which is what lets a test
    force the CPU path and what lets a caller keep a small job off the card.
    """
    want = (prefer or DEVICE or "auto").strip().lower()

    if want == CPU:
        return np
    if want == GPU:
        module = _cupy()
        if module is None:
            raise RuntimeError(
                "TRADER_DEVICE=gpu was asked for but CuPy is not usable here. "
                "Install cupy-cuda13x[ctk], or set TRADER_DEVICE=cpu.")
        return module

    return _cupy() or np


def to_device(array, module):
    """Move an array to whatever `module` operates on."""
    if module is np:
        return to_cpu(array)
    return module.asarray(array)


def to_cpu(array):
    """Bring an array back to numpy, wherever it has been.

    Everything that leaves the trainer -- weights to be packed into a bundle,
    probabilities to be scored -- goes through here. A cupy array that escapes
    into pandas or safetensors fails somewhere far from the cause.
    """
    getter = getattr(array, "get", None)
    if callable(getter):
        return getter()
    return np.asarray(array)


def describe(module) -> str:
    return "gpu" if module is not np else "cpu"
