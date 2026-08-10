"""MCP server.
==============

FastMCP server exposing JENN surrogate-modeling tools over stdio.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from jenn.core.model import NeuralNet
from jenn.post_processing.metrics import rsquare

from ._convert import prepare_training_data
from ._store import ModelRecord, ModelRegistry

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as err:  # pragma: no cover
    msg = (
        "The JENN MCP server needs the optional 'mcp' dependency. "
        "Install it with: pip install 'jenn[mcp]'"
    )
    raise ModuleNotFoundError(msg) from err


_REGISTRY = ModelRegistry()


INSTRUCTIONS = """\
Train a Jacobian-Enhanced Neural Network (JENN) using jenn.
Providing partials is optional, but improve the fit when available.
GUARD: don't act on one stochastic run; re-run with a new seed and compare.
"""

mcp = FastMCP("jenn", instructions=INSTRUCTIONS)


@mcp.tool()
def ping() -> str:
    """Return 'pong' to confirm the JENN MCP server is running."""
    return "pong"


def main() -> None:
    """Run the server over stdio (blocks until the client disconnects)."""
    mcp.run()


def _fit_metrics(
    model: NeuralNet,
    x: np.ndarray,  # feature-first (n_x, m)
    y: np.ndarray,  # feature-first (n_y, m)
    dydx: np.ndarray | None,
) -> dict[str, Any]:
    """Structured goodness-of-fit metrics for values and (optionally) partials."""
    # Keep the derivative work inside a single `dydx is not None` block: it
    # narrows the Optional for the type checker and reuses the one forward pass.
    if dydx is not None:
        y_pred, dydx_pred = model(x)  # both response and partials, one pass
        r2p = rsquare(dydx_pred, dydx)  # 3-D -> per-partial R², shape (n_y, n_x)
        rmsep = np.sqrt(np.mean((dydx_pred - dydx) ** 2, axis=-1))
        partials = {
            "r2": r2p.tolist(),
            "r2_min": float(r2p.min()),
            "rmse": rmsep.tolist(),
        }
    else:
        y_pred = model.predict(x)
        partials = None

    r2 = rsquare(y_pred, y)  # NOTE arg order: (prediction, truth); shape (n_y,)
    rmse = np.sqrt(np.mean((y_pred - y) ** 2, axis=-1))
    maxe = np.max(np.abs(y_pred - y), axis=-1)
    response = {
        "r2_per_output": r2.tolist(),
        "rmse_per_output": rmse.tolist(),
        "max_abs_error_per_output": maxe.tolist(),
    }

    return {
        "n_samples": int(x.shape[1]),
        "response": response,
        "partials": partials,
    }


GUIDANCE_TRAIN = (
    "These metrics are from a SINGLE stochastic training run on the TRAINING set. "
    "Training-set scores cannot reveal overfitting, and one run is noisy. Before "
    "diagnosing over/under-fitting or changing hyperparameters, re-run `train` with "
    "a different `random_state` and compare, and call `evaluate` on held-out data."
)


@mcp.tool()
def train(
    x: list[list[float]],
    y: list[list[float]],
    dydx: list[list[list[float]]] | None = None,
    hidden_layers: list[int] | None = None,
    hidden_activation: str = "tanh",
    output_activation: str = "linear",
    is_normalize: bool = True,
    gamma: float = 1.0,
    lambd: float = 0.0,
    alpha: float = 0.05,
    max_iter: int = 200,
    epochs: int = 1,
    batch_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Train a JENN surrogate and return a model_id plus training-set metrics.

    Data is row-per-sample: x=(m, n_x), y=(m, n_y), dydx=(m, n_y, n_x).
    The agent owns architecture/hyperparameters (hidden_layers, gamma, lambd);
    this tool is a thin wrapper over jenn.NeuralNet.fit. See the returned
    `guidance` and re-run before acting on a single result.
    """
    inputs, outputs, partials = prepare_training_data(x, y, dydx)
    n_x, n_y = inputs.shape[0], outputs.shape[0]
    hidden = list(hidden_layers) if hidden_layers else [12]
    layer_sizes = [n_x, *hidden, n_y]

    model = NeuralNet(layer_sizes, hidden_activation, output_activation)
    start = time.perf_counter()
    model.fit(
        inputs,
        outputs,
        partials,
        is_normalize=is_normalize,
        gamma=gamma,
        lambd=lambd,
        alpha=alpha,
        max_iter=max_iter,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    training_seconds = time.perf_counter() - start

    hyperparameters = {
        "hidden_activation": hidden_activation,
        "output_activation": output_activation,
        "is_normalize": is_normalize,
        "gamma": gamma,
        "lambd": lambd,
        "alpha": alpha,
        "max_iter": max_iter,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    record = ModelRecord(
        model=model,
        layer_sizes=layer_sizes,
        hyperparameters=hyperparameters,
        random_state=random_state,
        x=inputs,
        y=outputs,
        dydx=partials,
        training_seconds=training_seconds,
    )
    handle = _REGISTRY.add(record)

    return {
        "model_id": handle,
        "layer_sizes": layer_sizes,
        "n_samples": int(inputs.shape[1]),
        "n_inputs": n_x,
        "n_outputs": n_y,
        "hyperparameters": hyperparameters,
        "random_state": random_state,
        "training_seconds": round(training_seconds, 4),
        "training_metrics": _fit_metrics(model, inputs, outputs, partials),
        "guidance": GUIDANCE_TRAIN,
    }


GUIDANCE_EVALUATE = (
    "One run is stochastic. Before acting on a diagnosis (overfitting, "
    "underfitting, weak partials), compare against at least one other run "
    "trained with a different random_state, and prefer holdout over training data."
)


@mcp.tool()
def evaluate(
    model_id: str,
    x: list[list[float]] | None = None,
    y: list[list[float]] | None = None,
    dydx: list[list[list[float]]] | None = None,
) -> dict[str, Any]:
    """Score a trained model on held-out data, or on its training data.

    Pass both x and y (row-per-sample) for held-out metrics; pass
    neither to score the data the model was trained on. dydx is optional
    in both cases.
    """
    record = _REGISTRY.get(model_id)  # raises KeyError on unknown id

    if x is not None and y is not None:
        inputs, outputs, partials = prepare_training_data(x, y, dydx)
        dataset = "holdout"
        if (inputs.shape[0], outputs.shape[0]) != (
            record.x.shape[0],
            record.y.shape[0],
        ):
            msg = (
                "Data shape mismatch: model expects (n_inputs, n_outputs) = "
                f"({record.x.shape[0]}, {record.y.shape[0]}), got "
                f"({inputs.shape[0]}, {outputs.shape[0]})."
            )
            raise ValueError(msg)
    elif x is None and y is None:
        inputs, outputs, partials = record.x, record.y, record.dydx
        dataset = "training"
    else:
        msg = (
            "Provide both x and y for held-out evaluation, "
            "or neither to score the training data."
        )
        raise ValueError(msg)

    metrics = _fit_metrics(record.model, inputs, outputs, partials)
    return {
        "model_id": model_id,
        "dataset": dataset,
        "metrics": metrics,
        "guidance": GUIDANCE_EVALUATE,
    }


@mcp.tool()
def export(model_id: str, path: str | None = None) -> dict[str, Any]:
    """Save a trained model to JENN's native parameters JSON, reloadable via load.

    Returns the absolute file path and the JSON contents. Reload later
    with jenn.NeuralNet.load(path).
    """
    record = _REGISTRY.get(model_id)  # raises KeyError on unknown id
    target = Path(path) if path else Path(f"jenn_{model_id}.json")
    target = target.expanduser().resolve()
    record.model.save(target)  # reuse NeuralNet.save
    contents = json.loads(target.read_text())
    return {
        "model_id": model_id,
        "path": str(target),
        "format": "jenn-parameters-json",
        "note": "Reload with jenn.NeuralNet.load(path).",
        "parameters": contents,
    }


@mcp.tool()
def list_models() -> dict[str, Any]:
    """List the trained models currently held in this server session.

    Returns lightweight metadata only (no training data or weights) so
    the agent can pick and compare runs cheaply; the heavy arrays stay
    server-side, referenced by model_id.
    """
    models = [
        {
            "model_id": handle,
            "layer_sizes": record.layer_sizes,
            "n_samples": int(record.x.shape[1]),
            "random_state": record.random_state,
            "training_seconds": round(record.training_seconds, 4),
            "hyperparameters": record.hyperparameters,
        }
        for handle, record in _REGISTRY.items()
    ]
    return {"count": len(models), "models": models}


WORKFLOW = """\
Build a validated JENN surrogate:
1. Infer a modest architecture from the data (few samples -> small net).
2. Call `train`; read `training_metrics` (value vs. partials R²).
3. GUARD: one run is stochastic. Re-run `train` with a different `random_state`
   and compare, and call `evaluate` on held-out data BEFORE any diagnosis.
4. Diagnose and adjust:
   - train R² >> holdout R²  -> overfitting -> raise `lambd` or shrink the net.
   - value R² good but partials R² low -> raise `gamma` (weights the Jacobian term).
5. When good enough for the intended use, call `export` and hand the file to the user.
"""


@mcp.prompt()
def surrogate_workflow() -> str:
    """Recommended end-to-end workflow for building a JENN surrogate model."""
    return WORKFLOW
