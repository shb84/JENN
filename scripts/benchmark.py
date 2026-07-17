"""Performance benchmark for JENN's training hot path (issue #39).

Measures two things and records them to ``benchmark_results.json``:

1. **Micro** — wall-clock of the two hottest functions,
   :func:`jenn.core.propagation.gradient_enhancement` and
   :func:`jenn.core.propagation.next_layer_partials`, on synthetic arrays with
   ``n_x >= 2`` and large ``m`` (the regime the ``for j in range(n_x)`` loops
   dominate).
2. **End-to-end** — training the airfoil example
   (``docs/examples/data`` -> ``n_x=16``, ``m~42k``, gradient-enhanced) with a
   fixed seed and fixed iteration budget.

The script always calls the *current* implementation in the installed package,
so run it once before the optimization (``--tag baseline``) and once after
(``--tag vectorized``) to get a before/after comparison.

Usage::

    pixi run -e dev python scripts/benchmark.py --tag baseline
    pixi run -e dev python scripts/benchmark.py --tag vectorized --skip-e2e
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import jenn
from jenn.core.cache import Cache
from jenn.core.data import Dataset
from jenn.core.parameters import Parameters
from jenn.core.propagation import (
    gradient_enhancement,
    model_backward,
    model_partials_forward,
    next_layer_partials,
)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_RESULTS = _REPO / "benchmark_results.json"
_AIRFOIL = _REPO / "docs" / "examples" / "data"


def _time(fn, repeats: int, inner: int = 1) -> float:
    """Return median milliseconds per call over ``repeats`` samples."""
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner * 1e3)
    return statistics.median(samples)


def bench_micro(n_x: int = 16, m: int = 5000, repeats: int = 25) -> dict:
    """Micro-benchmark the two hot propagation functions on one layer."""
    layer_sizes = [n_x, 12, 12, 1]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_x, m))
    Y = rng.standard_normal((1, m))
    J = rng.standard_normal((1, n_x, m))

    parameters = Parameters(layer_sizes, hidden_activation="tanh")
    parameters.initialize(random_state=1)
    data = Dataset(X, Y, J)
    data.set_weights()
    cache = Cache(layer_sizes, m)

    # Populate the cache (forward + backward) so both functions have real inputs.
    model_partials_forward(X, parameters, cache)
    model_backward(data, parameters, cache)

    layer = 2  # a hidden layer (tanh) so second-derivative terms are non-zero
    ge = _time(lambda: gradient_enhancement(layer, parameters, cache, data), repeats)
    nlp = _time(lambda: next_layer_partials(layer, parameters, cache), repeats)
    return {
        "shape": {"n_x": n_x, "m": m, "layer_sizes": layer_sizes},
        "gradient_enhancement_ms": ge,
        "next_layer_partials_ms": nlp,
    }


def _load_airfoil() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.loadtxt(_AIRFOIL / "cd_x_y.csv", delimiter=";", dtype=float)
    X = xy[:, :-1].T
    Y = xy[:, -1].reshape((-1, 1)).T
    J = np.loadtxt(_AIRFOIL / "cd_dy.csv", delimiter=";", dtype=float).T
    n_y, _ = Y.shape
    n_x, m = X.shape
    J = J.reshape((n_y, n_x, m))
    return X, Y, J


def bench_e2e(epochs: int = 25, max_iter: int = 2, seed: int = 42) -> dict:
    """Train the airfoil model end-to-end and time it (fixed seed/budget)."""
    if not (_AIRFOIL / "cd_x_y.csv").exists():
        return {"skipped": "airfoil data not found"}

    X, Y, J = _load_airfoil()
    n_x, m = X.shape
    n_y, _ = Y.shape

    model = jenn.NeuralNet(layer_sizes=[n_x, 12, 12, n_y])
    t0 = time.perf_counter()
    model.fit(
        x=X,
        y=Y,
        dydx=J,
        alpha=1e-2,
        lambd=1e-2,
        gamma=1.0,
        is_backtracking=True,
        is_normalize=True,
        is_verbose=False,
        max_iter=max_iter,
        batch_size=1000,
        epochs=epochs,
        shuffle=True,
        random_state=seed,
    )
    elapsed = time.perf_counter() - t0
    return {
        "shape": {"n_x": n_x, "m": m},
        "config": {"epochs": epochs, "max_iter": max_iter, "seed": seed},
        "train_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run", help="label for this run")
    parser.add_argument("--skip-e2e", action="store_true", help="skip airfoil train")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jenn": jenn.__version__,
        "micro": bench_micro(repeats=args.repeats),
    }
    if not args.skip_e2e:
        record["e2e"] = bench_e2e()

    results = {}
    if _RESULTS.exists():
        results = json.loads(_RESULTS.read_text())
    results.setdefault("runs", {})[args.tag] = record
    _RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    m = record["micro"]
    print(f"[{args.tag}]  numpy {record['numpy']}  {record['platform']}")
    print(f"  gradient_enhancement : {m['gradient_enhancement_ms']:8.3f} ms")
    print(f"  next_layer_partials  : {m['next_layer_partials_ms']:8.3f} ms")
    if "e2e" in record and "train_seconds" in record["e2e"]:
        print(f"  airfoil train        : {record['e2e']['train_seconds']:8.3f} s")
    print(f"  -> {_RESULTS}")


if __name__ == "__main__":
    main()
