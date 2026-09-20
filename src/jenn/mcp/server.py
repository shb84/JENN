"""MCP server.
==============

MCPServer exposing JENN surrogate-modeling tools over stdio.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import contextlib
import csv
import itertools
import json
import os
import time
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import quote

import numpy as np

from jenn.core.model import NeuralNet
from jenn.post_processing.metrics import rsquare
from jenn.utilities import load_csv, load_csv_inputs, load_npz, load_npz_inputs

from ._convert import (
    prepare_training_data,
    to_feature_first,
    to_row_per_sample,
    to_row_per_sample_partials,
)
from ._store import DatasetRecord, ModelRecord, Registry

try:
    import anyio
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ResourceNotFoundError
    from mcp.shared.path_security import PathEscapeError, safe_join
    from mcp_types import Resource as MCPResource
except ModuleNotFoundError as err:  # pragma: no cover
    msg = (
        "The JENN MCP server needs the optional 'mcp' dependency, version 2 or "
        "later (mcp 1.x named this class FastMCP and is no longer supported). "
        "Install it with: pip install 'jenn[mcp]'"
    )
    raise ModuleNotFoundError(msg) from err


# ----------------------------------------------------------
# --- SERVER SETUP -----------------------------------------
# ----------------------------------------------------------


_MODELS: Registry[ModelRecord] = Registry()
_DATASETS: Registry[DatasetRecord] = Registry()


INSTRUCTIONS = """\
Train a Jacobian-Enhanced Neural Network (JENN) using jenn.
Providing partials is optional, but improve the fit when available.
GUARD: don't act on one stochastic run; re-run with a new seed and compare.
"""


class _JennMCP(MCPServer):
    """An :class:`MCPServer` with one live resource per file in ``$JENN_DIR``.

    Each file is advertised as its own ``jenn://files/<name>`` entry,
    which is what makes it individually pickable in an agent's ``@``
    menu; reads go through the ``jenn://files/{+path}`` template below.
    """

    # Overridden because the SDK only serves statically registered resources,
    # leaving no other hook for a directory whose contents change between calls.
    async def list_resources(self) -> list[MCPResource]:
        """Registered resources, plus one per discoverable file."""
        static = await super().list_resources()
        dynamic = await anyio.to_thread.run_sync(_file_resources)
        return [*static, *dynamic]


mcp = _JennMCP("jenn", instructions=INSTRUCTIONS)


@mcp.tool()
def ping() -> str:
    """Return 'pong' to confirm the JENN MCP server is running."""
    return "pong"


def main() -> None:
    """Run the server over stdio (blocks until the client disconnects)."""
    mcp.run()


# ----------------------------------------------------------
# --- SHARED HELPERS ---------------------------------------
# ----------------------------------------------------------


_DEFAULT_DIR_NAME = ".jenn_dir"


def _jenn_root() -> Path:
    """The working folder for JENN files: ``$JENN_DIR``, else ``./.jenn_dir``.

    Put training data here to make it visible to the agent; exported
    models land here too. Set ``JENN_DIR`` to use a folder of your own
    (created only if it is the default, so a typo is reported rather
    than made).
    """
    env = os.environ.get("JENN_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = (Path.cwd() / _DEFAULT_DIR_NAME).resolve()
    # A read-only working directory means no default root, not a crash:
    # the scan below simply finds nothing.
    with contextlib.suppress(OSError):
        root.mkdir(exist_ok=True)
    return root


def _resolve_path(path: str) -> Path:
    """Resolve a user-supplied file path against ``JENN_DIR``.

    Absolute and ``~`` paths are used as given; anything else is read
    relative to ``JENN_DIR``, so ``"data.csv"`` means the file of that
    name in the working folder.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _jenn_root() / p
    return p.resolve()


def _label(names: list[str] | None, index: int) -> str | int:
    """Column name if known, else the positional index."""
    return names[index] if names else index


def _partition_partials(
    mask: np.ndarray | None,
    input_names: list[str] | None,
    output_names: list[str] | None,
) -> tuple[list[list[str | int]], list[list[str | int]]]:
    """Split partials into available vs missing ``[output, input]`` label pairs."""
    if mask is None:
        return [], []
    n_y, n_x = mask.shape
    available: list[list[str | int]] = []
    missing: list[list[str | int]] = []
    for o in range(n_y):
        for i in range(n_x):
            pair = [_label(output_names, o), _label(input_names, i)]
            (available if mask[o, i] else missing).append(pair)
    return available, missing


# ----------------------------------------------------------
# --- TRAINING BOUNDS (EXTRAPOLATION CHECK) ----------------
# ----------------------------------------------------------

BOUNDS_UNAVAILABLE = "unavailable (this model carries no stored training bounds)"

BOUNDS_NOTE = (
    "`overshoot` is a fraction of the trained span, so it reads the same "
    "regardless of units: ~0.02 is rounding error, >0.25 means the model is "
    "guessing. Clip the query to `trained`, or gather training data out to "
    "`query`. NOTE: this is a bounding-box test -- being inside it is necessary "
    "but not sufficient for interpolation, since a point can sit within every "
    "per-input range and still be far from any training sample."
)


def _bounds_summary(record: ModelRecord) -> list[dict[str, Any]] | None:
    """Per-input training range for metadata responses, or None if unknown.

    :return:``[{"input": <label>, "min": ..., "max": ...}, ...]``
    """
    if record.x_min is None or record.x_max is None:
        return None
    return [
        {
            "input": _label(record.input_names, i),
            "min": float(record.x_min[i]),
            "max": float(record.x_max[i]),
        }
        for i in range(record.x_min.size)
    ]


def _read_bounds(
    data: dict[str, Any],
    n_x: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Recover the bounds `export` wrote from a parsed model file.

    Bounds are optional -- a model saved through ``NeuralNet.save`` has
    none -- and anything unusable is reported as unknown rather than
    raised, so bad bounds never stop good weights from loading.
    """
    try:
        x_min = np.asarray(data["x_min"], dtype=float).ravel()
        x_max = np.asarray(data["x_max"], dtype=float).ravel()
    except (KeyError, TypeError, ValueError):
        return None, None, None
    if x_min.size != n_x or x_max.size != n_x:
        return None, None, None  # a short array would broadcast silently
    # JSON `null` arrives as NaN (orjson accepts it, numpy coerces it), and NaN
    # compares False against everything, so the check would never fire again.
    if not (np.all(np.isfinite(x_min)) and np.all(np.isfinite(x_max))):
        return None, None, None
    if np.any(x_min > x_max):
        return None, None, None  # inverted: every point would read as outside
    names = data.get("input_names")
    if not (
        isinstance(names, list)
        and len(names) == n_x
        and all(isinstance(name, str) for name in names)
    ):
        names = None
    return x_min, x_max, names


def _overshoot(excess: float, lo: float, hi: float) -> float:
    """Scale ``excess`` by the trained span ``[lo, hi]`` so it is unit-free.

    A constant input (zero span) is scaled by its magnitude instead.
    """
    span = hi - lo
    scale = span if span > 0 else max(abs(lo), 1.0)  # never divide by zero
    return excess / scale


def _bounds_report(
    record: ModelRecord,
    inputs_ff: np.ndarray,  # feature-first (n_x, m)
) -> dict[str, Any] | str | None:
    """Report how far ``inputs_ff`` falls outside the training bounding box.

    :param inputs_ff: feature-first inputs of shape ``(n_x, m)``
    :return:``None`` if every sample is inside the box,
        :data:`BOUNDS_UNAVAILABLE` if the model has no stored bounds,
        else a report listing only the inputs that were exceeded.
    """
    if record.x_min is None or record.x_max is None:
        return BOUNDS_UNAVAILABLE

    x_min = record.x_min.reshape(-1, 1)
    x_max = record.x_max.reshape(-1, 1)
    below = x_min - inputs_ff  # > 0 where a sample undershoots
    above = inputs_ff - x_max  # > 0 where a sample overshoots
    outside = (below > 0.0) | (above > 0.0)  # (n_x, m)
    if not outside.any():
        return None

    inputs: list[dict[str, Any]] = []
    for i in np.flatnonzero(outside.any(axis=1)):
        lo, hi = float(record.x_min[i]), float(record.x_max[i])
        excess = max(float(below[i].max()), float(above[i].max()))
        inputs.append({
            "input": _label(record.input_names, int(i)),
            "trained": [lo, hi],
            "query": [float(inputs_ff[i].min()), float(inputs_ff[i].max())],
            "overshoot": round(_overshoot(excess, lo, hi), 4),
            "n_outside": int(outside[i].sum()),
        })

    return {
        "n_samples_outside": int(outside.any(axis=0).sum()),
        "n_samples": int(inputs_ff.shape[1]),
        "worst_overshoot": max(entry["overshoot"] for entry in inputs),
        "inputs": inputs,
        "note": BOUNDS_NOTE,
    }


def _fit_metrics(
    model: NeuralNet,
    x: np.ndarray,  # feature-first (n_x, m)
    y: np.ndarray,  # feature-first (n_y, m)
    dydx: np.ndarray | None,
    mask: np.ndarray | None = None,  # (n_y, n_x) availability; None => all present
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> dict[str, Any]:
    """Structured goodness-of-fit metrics for values and (optionally) partials.

    Partials masked out by ``mask == 0`` hold placeholder values, so
    they are listed under ``ignored`` rather than scored.
    """
    # Keep the derivative work inside a single `dydx is not None` block: it
    # narrows the Optional for the type checker and reuses the one forward pass.
    if dydx is not None:
        y_pred, dydx_pred = model(x)  # both response and partials, one pass
        r2p = rsquare(dydx_pred, dydx)  # 3-D -> per-partial R², shape (n_y, n_x)
        rmsep = np.sqrt(np.mean((dydx_pred - dydx) ** 2, axis=-1))
        n_y, n_x = r2p.shape
        available: list[dict[str, Any]] = []
        ignored: list[dict[str, Any]] = []
        for o in range(n_y):
            for i in range(n_x):
                pair = {
                    "output": _label(output_names, o),
                    "input": _label(input_names, i),
                }
                if mask is not None and not mask[o, i]:
                    ignored.append(pair)
                else:
                    available.append(
                        {**pair, "r2": float(r2p[o, i]), "rmse": float(rmsep[o, i])},
                    )
        partials = {
            "available": available,
            "ignored": ignored,
            "r2_min": min((p["r2"] for p in available), default=None),
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


def _resolve_dataset(
    dataset_id: str | None,
    x: list[list[float]] | None,
    y: list[list[float]] | None,
    dydx: list[list[list[float]]] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list[str] | None,
    list[str] | None,
]:
    """Resolve either a ``dataset_id`` or inline arrays to feature-first data.

    :return:``(x, y, dydx, mask, input_names, output_names)``. ``mask``
        and the names are ``None`` for inline (positional) data.
    """
    if dataset_id is not None:
        if x is not None or y is not None or dydx is not None:
            msg = "Provide either dataset_id or inline x/y/dydx, not both."
            raise ValueError(msg)
        rec = _DATASETS.get(dataset_id)  # raises KeyError on unknown id
        return (
            rec.x,
            rec.y,
            rec.dydx,
            rec.partial_mask,
            rec.input_names,
            rec.output_names,
        )
    if x is None or y is None:
        msg = "Provide a dataset_id, or both x and y."
        raise ValueError(msg)
    inputs, outputs, partials = prepare_training_data(x, y, dydx)
    return inputs, outputs, partials, None, None, None


def _effective_gamma(
    gamma: float | list[dict[str, float | str]],
    mask: np.ndarray | None,
    input_names: list[str] | None,
    output_names: list[str] | None,
    n_y: int,
    n_x: int,
) -> np.ndarray | float:
    """Combine the availability mask with the requested per-partial scale.

    :return: a scalar when there is no ``mask``, else ``mask * scale``
        shaped ``(n_y, n_x, 1)`` to broadcast over samples. Availability
        wins: no scale can resurrect an absent partial.
    """
    if isinstance(gamma, list):
        if mask is None or input_names is None or output_names is None:
            msg = (
                "Per-partial gamma overrides require a dataset ingested with "
                "column names; inline/positional data has none."
            )
            raise ValueError(msg)
        scale = np.ones((n_y, n_x))
        for override in gamma:
            missing_keys = {"output", "input", "weight"} - override.keys()
            if missing_keys:
                msg = (
                    f"gamma override {override} is missing key(s) "
                    f"{sorted(missing_keys)}; each needs 'output', 'input', 'weight'."
                )
                raise ValueError(msg)
            out, inp = str(override["output"]), str(override["input"])
            if out not in output_names:
                msg = f"gamma override output '{out}' is not in {output_names}."
                raise ValueError(msg)
            if inp not in input_names:
                msg = f"gamma override input '{inp}' is not in {input_names}."
                raise ValueError(msg)
            scale[output_names.index(out), input_names.index(inp)] = float(
                override["weight"],
            )
        return (mask * scale)[:, :, None]
    if mask is None:
        return float(gamma)  # inline dense data -> unchanged scalar behavior
    return (mask * float(gamma))[:, :, None]


# ----------------------------------------------------------
# --- TRAINING AND EVALUATION ------------------------------
# ----------------------------------------------------------


GUIDANCE_TRAIN = (
    "These metrics are from a SINGLE stochastic training run on the TRAINING set. "
    "Training-set scores cannot reveal overfitting, and one run is noisy. Before "
    "diagnosing over/under-fitting or changing hyperparameters, re-run `train` with "
    "a different `random_state` and compare, and call `evaluate` on held-out data."
)


@mcp.tool()
def train(
    x: list[list[float]] | None = None,
    y: list[list[float]] | None = None,
    dydx: list[list[list[float]]] | None = None,
    dataset_id: str | None = None,
    hidden_layers: list[int] | None = None,
    hidden_activation: str = "tanh",
    output_activation: str = "linear",
    is_normalize: bool = True,
    gamma: float | list[dict[str, float | str]] = 1.0,
    lambd: float = 0.0,
    alpha: float = 0.05,
    max_iter: int = 200,
    epochs: int = 1,
    batch_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Train a JENN surrogate and return a model_id plus training-set metrics.

    Supply data either inline (row-per-sample: x=(m, n_x), y=(m, n_y),
    dydx=(m, n_y, n_x)) or by `dataset_id` from a prior `ingest`
    (mutually exclusive). With an ingested dataset, `gamma` may be a
    scalar OR a list of per-partial overrides, e.g.
    `[{"output":"Cd","input":"alpha","weight":3.0}]`; partials absent
    from the data are always weighted 0 and cannot be resurrected by an
    override. Metrics are on the training set from one stochastic run:
    re-run with a different `random_state` before acting on them.
    """
    inputs, outputs, partials, mask, input_names, output_names = _resolve_dataset(
        dataset_id,
        x,
        y,
        dydx,
    )
    n_x, n_y = inputs.shape[0], outputs.shape[0]
    hidden = list(hidden_layers) if hidden_layers else [12]
    layer_sizes = [n_x, *hidden, n_y]
    gamma_eff = _effective_gamma(gamma, mask, input_names, output_names, n_y, n_x)

    model = NeuralNet(layer_sizes, hidden_activation, output_activation)
    start = time.perf_counter()
    model.fit(
        inputs,
        outputs,
        partials,
        is_normalize=is_normalize,
        gamma=gamma_eff,
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
        "gamma": gamma,  # as requested (scalar or overrides), not the expanded array
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
        partial_mask=mask,
        x_min=inputs.min(axis=1),  # inputs is (n_x, m): axis 1 is the sample axis
        x_max=inputs.max(axis=1),
        input_names=input_names,
        output_names=output_names,
    )
    handle = _MODELS.add(record)

    return {
        "model_id": handle,
        "dataset_id": dataset_id,
        "layer_sizes": layer_sizes,
        "n_samples": int(inputs.shape[1]),
        "n_inputs": n_x,
        "n_outputs": n_y,
        "hyperparameters": hyperparameters,
        "random_state": random_state,
        "training_seconds": round(training_seconds, 4),
        "training_metrics": _fit_metrics(
            model,
            inputs,
            outputs,
            partials,
            mask,
            input_names,
            output_names,
        ),
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
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Score a trained model on held-out data, or on its training data.

    Pass a `dataset_id` (from `ingest`) or both x and y (row-per-sample)
    for held-out metrics; pass none of them to score the data the model
    was trained on. dydx is optional in every case.
    """
    record = _MODELS.get(model_id)  # raises KeyError on unknown id
    # Reuse the model's own availability mask/names for labelling by default;
    # a held-out dataset_id overrides them with its own.
    mask, input_names, output_names = (
        record.partial_mask,
        record.input_names,
        record.output_names,
    )

    if dataset_id is not None:
        if x is not None or y is not None or dydx is not None:
            msg = "Provide either dataset_id or inline x/y, not both."
            raise ValueError(msg)
        rec = _DATASETS.get(dataset_id)  # raises KeyError on unknown id
        inputs, outputs, partials = rec.x, rec.y, rec.dydx
        mask, input_names, output_names = (
            rec.partial_mask,
            rec.input_names,
            rec.output_names,
        )
        dataset = f"dataset:{dataset_id}"
    elif x is not None and y is not None:
        inputs, outputs, partials = prepare_training_data(x, y, dydx)
        mask = None  # inline partials are all present as given
        dataset = "holdout"
        if (inputs.shape[0], outputs.shape[0]) != (
            record.layer_sizes[0],
            record.layer_sizes[-1],
        ):
            msg = (
                "Data shape mismatch: model expects (n_inputs, n_outputs) = "
                f"({record.layer_sizes[0]}, {record.layer_sizes[-1]}), got "
                f"({inputs.shape[0]}, {outputs.shape[0]})."
            )
            raise ValueError(msg)
    elif x is None and y is None:
        if record.x is None or record.y is None:
            msg = (
                "This model was loaded from disk and carries no training data to "
                "score. Provide holdout x and y, or a dataset_id."
            )
            raise ValueError(msg)
        inputs, outputs, partials = record.x, record.y, record.dydx
        dataset = "training"
    else:
        msg = (
            "Provide both x and y for held-out evaluation, "
            "or neither to score the training data."
        )
        raise ValueError(msg)

    metrics = _fit_metrics(
        record.model,
        inputs,
        outputs,
        partials,
        mask,
        input_names,
        output_names,
    )
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
    with `load_model`, or with jenn.NeuralNet.load(path) from the API. A
    relative `path` (or the default name) resolves under `$JENN_DIR`
    (`./.jenn_dir` if unset); an absolute path is used as-is.

    The file also carries the training-input bounding box (`x_min`,
    `x_max`, and `input_names` when known) so a later session can flag
    extrapolation; the API loader ignores these keys.
    """
    record = _MODELS.get(model_id)  # raises KeyError on unknown id
    target = _resolve_path(path or f"jenn_{model_id}.json")
    record.model.save(target)  # reuse NeuralNet.save
    contents = json.loads(target.read_text())
    if record.x_min is not None and record.x_max is not None:
        contents["x_min"] = record.x_min.tolist()
        contents["x_max"] = record.x_max.tolist()
        if record.input_names is not None:
            contents["input_names"] = list(record.input_names)
        target.write_text(json.dumps(contents), encoding="utf-8")
    return {
        "model_id": model_id,
        "path": str(target),
        "format": "jenn-parameters-json",
        "note": "Reload with jenn.NeuralNet.load(path).",
        "training_bounds": _bounds_summary(record),
        "parameters": contents,
    }


@mcp.tool()
def list_models() -> dict[str, Any]:
    """List the trained models currently held in this server session.

    Returns metadata only -- architecture, hyperparameters, training
    bounds -- for comparing runs; the weights and data stay server-side,
    referenced by model_id.
    """
    models = [
        {
            "model_id": handle,
            "layer_sizes": record.layer_sizes,
            "n_samples": None if record.x is None else int(record.x.shape[1]),
            "random_state": record.random_state,
            "training_seconds": round(record.training_seconds, 4),
            "hyperparameters": record.hyperparameters,
            "training_bounds": _bounds_summary(record),
            "source": record.source,
        }
        for handle, record in _MODELS.items()
    ]
    return {"count": len(models), "models": models}


# ----------------------------------------------------------
# --- DATA INGESTION ---------------------------------------
# ----------------------------------------------------------


@mcp.tool()
def ingest(
    path: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    derivatives: list[dict[str, str]] | None = None,
    fmt: str | None = None,
    delimiter: str = ",",
) -> dict[str, Any]:
    """Load training data from a file and register it as a dataset.

    Formats: `csv` (wide, one row per sample) and `npz` (arrays `x`, `y`, and
    optionally `dydx`, already feature-first); inferred from the suffix unless
    `fmt` is given. For CSV, `inputs`/`outputs` name the value columns and
    `derivatives` explicitly pairs each present partial with its column::

        {"output": <name in outputs>, "input": <name in inputs>, "column": <col>}

    No naming convention is assumed. Partials that are not listed are
    reported as absent and weighted 0 at train time. Returns a
    `dataset_id` to pass to `train`/`evaluate`, which keeps the data
    server-side instead of re-sending it per call.

    A relative `path` resolves under `$JENN_DIR` (`./.jenn_dir` if
    unset); an absolute path is used as-is.
    """
    target = _resolve_path(path)
    if not target.is_file():
        msg = f"No such file: {target}"
        raise ValueError(msg)
    kind = (fmt or target.suffix.lstrip(".")).lower()

    if kind == "csv":
        if not inputs or not outputs:
            msg = "CSV ingest requires `inputs` and `outputs` column lists."
            raise ValueError(msg)
        arrays = load_csv(target, inputs, outputs, derivatives, delimiter)
        input_names: list[str] | None = list(inputs)
        output_names: list[str] | None = list(outputs)
        source = str(target)
    elif kind == "npz":
        arrays = load_npz(target)
        input_names = output_names = None
        source = f"npz:{target}"
    else:
        msg = f"Unsupported format '{kind}'; expected 'csv' or 'npz'."
        raise ValueError(msg)

    data_x, data_y, data_dydx, mask = arrays
    record = DatasetRecord(
        x=data_x,
        y=data_y,
        dydx=data_dydx,
        partial_mask=mask,
        input_names=input_names,
        output_names=output_names,
        source=source,
    )
    handle = _DATASETS.add(record)

    available, missing = _partition_partials(mask, input_names, output_names)
    if data_dydx is None:
        note = "No derivatives provided; this dataset trains on values only."
    elif missing:
        note = (
            f"{len(missing)} of {len(available) + len(missing)} partials are "
            "absent and will be gamma-masked to 0 at train time."
        )
    else:
        note = "All partials present."
    return {
        "dataset_id": handle,
        "source": source,
        "n_samples": int(data_x.shape[1]),
        "n_inputs": int(data_x.shape[0]),
        "n_outputs": int(data_y.shape[0]),
        "input_names": input_names,
        "output_names": output_names,
        "partials_available": available,
        "partials_missing": missing,
        "note": note,
    }


@mcp.tool()
def list_datasets() -> dict[str, Any]:
    """List the ingested datasets currently held in this server session.

    Returns metadata only -- shapes, column names, and which partials
    are available vs. missing; the arrays stay server-side, referenced
    by dataset_id.
    """
    datasets = [
        {
            "dataset_id": handle,
            "source": record.source,
            "n_samples": int(record.x.shape[1]),
            "n_inputs": int(record.x.shape[0]),
            "n_outputs": int(record.y.shape[0]),
            "input_names": record.input_names,
            "output_names": record.output_names,
            "has_partials": record.dydx is not None,
            "partials_available": _partition_partials(
                record.partial_mask,
                record.input_names,
                record.output_names,
            )[0],
            "partials_missing": _partition_partials(
                record.partial_mask,
                record.input_names,
                record.output_names,
            )[1],
        }
        for handle, record in _DATASETS.items()
    ]
    return {"count": len(datasets), "datasets": datasets}


# ----------------------------------------------------------
# --- MODEL REUSE (LOAD & PREDICT) -------------------------
# ----------------------------------------------------------


@mcp.tool()
def load_model(path: str) -> dict[str, Any]:
    """Load a saved JENN model from disk and register it for reuse.

    Use this to pick up a model `export`ed in an earlier session and run
    it with `predict`. Returns the `model_id` plus `source`,
    `layer_sizes`, `n_inputs`, `n_outputs`, and `training_bounds` -- the
    per-input range the model was trained over, or `null` if the file
    carries none. A saved model holds only weights and normalization, so
    `evaluate` on it needs holdout data or a `dataset_id`.

    A relative `path` resolves under `$JENN_DIR` (`./.jenn_dir` if
    unset); an absolute path is used as-is.
    """
    target = _resolve_path(path)
    if not target.is_file():
        msg = f"No such file: {target}"
        raise ValueError(msg)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not (isinstance(data, dict) and {"layer_sizes", "W", "b"} <= data.keys()):
        msg = f"`{target}` is not a valid JENN model file."
        raise ValueError(msg)
    model = NeuralNet.load(target)
    layer_sizes = model.parameters.layer_sizes
    x_min, x_max, input_names = _read_bounds(data, layer_sizes[0])
    record = ModelRecord(
        model=model,
        layer_sizes=layer_sizes,
        training_seconds=0.0,
        x_min=x_min,
        x_max=x_max,
        input_names=input_names,
        source=str(target),
    )
    handle = _MODELS.add(record)
    return {
        "model_id": handle,
        "source": str(target),
        "layer_sizes": layer_sizes,
        "n_inputs": layer_sizes[0],
        "n_outputs": layer_sizes[-1],
        "training_bounds": _bounds_summary(record),
        "note": (
            "Data-less model (weights + normalization only). Run it with "
            "`predict`; to `evaluate` it, supply holdout x/y or a dataset_id."
        ),
    }


def _resolve_predict_inputs(
    x: list[list[float]] | None,
    path: str | None,
    inputs: list[str] | None,
    delimiter: str,
) -> np.ndarray:
    """Resolve inline ``x`` or a CSV/NPZ ``path`` to feature-first inputs.

    ``x`` and ``path`` are mutually exclusive. A ``.csv`` path needs an
    ``inputs`` column list; a ``.npz`` path reads its feature-first
    ``x`` array.

    :return: feature-first inputs of shape ``(n_x, m)``
    """
    if x is not None and path is not None:
        msg = "Provide either inline x or a file path, not both."
        raise ValueError(msg)
    if x is not None:
        return to_feature_first(x, "x")
    if path is None:
        msg = "Provide inline x or a file path."
        raise ValueError(msg)
    target = _resolve_path(path)
    if not target.is_file():
        msg = f"No such file: {target}"
        raise ValueError(msg)
    kind = target.suffix.lstrip(".").lower()
    if kind == "csv":
        if not inputs:
            msg = "CSV input requires an `inputs` column list."
            raise ValueError(msg)
        return load_csv_inputs(target, inputs, delimiter)
    if kind == "npz":
        return load_npz_inputs(target)
    msg = f"Unsupported input format '{kind}'; expected 'csv' or 'npz'."
    raise ValueError(msg)


def _write_predictions(
    output_path: str,
    inputs_ff: np.ndarray,  # feature-first (n_x, m)
    y_ff: np.ndarray,  # feature-first (n_y, m)
    dydx_ff: np.ndarray | None,  # feature-first (n_y, n_x, m) or None
    delimiter: str,
) -> dict[str, Any]:
    """Write predictions to a file, feature-first (matching the loaders).

    ``.npz`` holds ``x``/``y`` (plus ``dydx`` when partials were
    requested); ``.csv`` holds the response columns only (partials
    require ``.npz``).

    :return:``{"path": ..., "n_samples": ...}``
    """
    out = _resolve_path(output_path)
    kind = out.suffix.lstrip(".").lower()
    if kind == "npz":
        arrays = {"x": inputs_ff, "y": y_ff}
        if dydx_ff is not None:
            arrays["dydx"] = dydx_ff
        np.savez(out, **arrays)
    elif kind == "csv":
        if dydx_ff is not None:
            msg = "CSV output holds response columns only; use .npz to write partials."
            raise ValueError(msg)
        rows = to_row_per_sample(y_ff)  # (m, n_y)
        header = [f"y{i}" for i in range(y_ff.shape[0])]
        with out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
    else:
        msg = f"Unsupported output format '{kind}'; expected 'csv' or 'npz'."
        raise ValueError(msg)
    return {"path": str(out), "n_samples": int(inputs_ff.shape[1])}


@mcp.tool()
def predict(
    model_id: str,
    x: list[list[float]] | None = None,
    path: str | None = None,
    inputs: list[str] | None = None,
    delimiter: str = ",",
    with_partials: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run a trained model on new inputs, optionally returning the Jacobian.

    Supply inputs either inline (row-per-sample `x`=(m, n_x)) OR by file
    `path` (mutually exclusive): a `.csv` (a row-per-sample table, with
    an `inputs` column list) or a `.npz` holding a feature-first array
    `x` of shape `(n_x, m)` -- the core axis order, the transpose of the
    inline arrays. The input width must match the model's `n_inputs`.
    With `with_partials`, the response also includes `dydx` (row-per-
    sample (m, n_y, n_x)). By default results are returned inline; give
    `output_path` (`.csv` or `.npz`) to write them to a file instead
    (for large runs) -- the inline arrays are then omitted. A `.csv`
    output is a row-per-sample table of the response columns only; a
    `.npz` is feature-first (`x`/`y`/`dydx`) and persists partials.
    Relative `path`/`output_path` values resolve under `$JENN_DIR`
    (`./.jenn_dir` if unset); absolute paths are used as-is.

    An `extrapolation` key appears only when a sample falls outside the
    box the model was trained over, naming the inputs that were exceeded
    and by what fraction of their trained span. Being inside the box is
    necessary but not sufficient for interpolation: a point can sit
    within every per-input range and still be far from any training
    sample.
    """
    record = _MODELS.get(model_id)  # raises KeyError on unknown id
    inputs_ff = _resolve_predict_inputs(x, path, inputs, delimiter)

    n_x = record.layer_sizes[0]
    if inputs_ff.shape[0] != n_x:
        msg = f"Input width {inputs_ff.shape[0]} does not match model n_inputs {n_x}."
        raise ValueError(msg)

    if with_partials:
        y_ff, dydx_ff = record.model(inputs_ff)  # (n_y, m), (n_y, n_x, m)
    else:
        y_ff = record.model.predict(inputs_ff)  # (n_y, m)
        dydx_ff = None

    # Reported for a file-bound run too: writing thousands of predictions is more
    # reason to know they were extrapolated, not less.
    extrapolation = _bounds_report(record, inputs_ff)

    result: dict[str, Any]
    if output_path is not None:
        written = _write_predictions(output_path, inputs_ff, y_ff, dydx_ff, delimiter)
        result = {"model_id": model_id, **written}
    else:
        result = {"model_id": model_id, "y": to_row_per_sample(y_ff)}
        if dydx_ff is not None:
            result["dydx"] = to_row_per_sample_partials(dydx_ff)
    if extrapolation is not None:  # omitted entirely when every sample is inside
        result["extrapolation"] = extrapolation
    return result


# ----------------------------------------------------------
# --- RESOURCES --------------------------------------------
# ----------------------------------------------------------


def _jenn_model_info(path: Path) -> dict[str, Any] | None:
    """Return ``{'layer_sizes': ...}`` if ``path`` is a JENN model JSON, else None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if isinstance(data, dict) and {"layer_sizes", "W", "b"} <= data.keys():
        return {"layer_sizes": data.get("layer_sizes")}
    return None


def _csv_header(path: Path) -> tuple[list[str], str]:
    """Read a CSV's header names and guess its delimiter (for discovery only)."""
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        first = file.readline()
    try:
        delimiter = csv.Sniffer().sniff(first, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    header = next(csv.reader([first], delimiter=delimiter), [])
    return [name.strip() for name in header], delimiter


def _file_details(path: Path, suffix: str) -> dict[str, Any] | None:
    """Format-specific metadata for a file, or None if it is not JENN-relevant."""
    if suffix == ".csv":
        columns, delimiter = _csv_header(path)
        return {
            "kind": "data",
            "format": "csv",
            "columns": columns,
            "delimiter": delimiter,
        }
    if suffix == ".npz":
        with np.load(path) as archive:  # lazy: reads the zip index, not the arrays
            arrays = list(archive.files)
        return {"kind": "data", "format": "npz", "arrays": arrays}
    if suffix == ".json":
        info = _jenn_model_info(path)  # None for non-JENN JSON -> skipped
        return (
            None if info is None else {"kind": "model", "format": "jenn-model", **info}
        )
    return None


def _file_entry(path: Path, root: Path) -> dict[str, Any] | None:
    """Describe one discoverable file, or None if it is not JENN-relevant."""
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".npz", ".json"}:
        return None
    entry: dict[str, Any] = {
        "path": str(path),
        "name": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
    }
    try:
        details = _file_details(path, suffix)
    except (ValueError, OSError) as err:  # one bad file must not sink the listing
        return {**entry, "format": suffix.lstrip("."), "error": str(err)}
    if details is None:
        return None
    return {**entry, **details}


def _scan_files(root: Path) -> dict[str, Any]:
    """List JENN-relevant files (CSV/NPZ data + exported model JSON) under root."""
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        entry = _file_entry(path, root)
        if entry is not None:
            files.append(entry)
    return {"root": str(root), "count": len(files), "files": files}


@mcp.resource(
    "jenn://files",
    name="jenn-files",
    description=(
        "Local JENN files under $JENN_DIR (or ./.jenn_dir): CSV/NPZ data files "
        "to ingest, and exported model JSONs to load."
    ),
    mime_type="application/json",
)
def files() -> dict[str, Any]:
    """List local JENN data and model files to reference by path."""
    return _scan_files(_jenn_root())


# --- one resource per file (so an agent's `@` menu can browse them) ---

_MAX_FILE_RESOURCES = 200  # a big JENN_DIR must not flood the picker
_PREVIEW_LINES = 6
_PREVIEW_CHARS = 200


def _entry_summary(entry: dict[str, Any]) -> str:
    """One-line description of a file, for the ``@`` menu row."""
    fmt = entry.get("format", "")
    if "error" in entry:
        return f"{fmt} file (unreadable: {entry['error']})"
    if entry.get("kind") == "model":
        return f"JENN model - layers {entry.get('layer_sizes')}"
    if fmt == "csv":
        return "CSV data - columns: " + ", ".join(entry.get("columns", []))
    if fmt == "npz":
        return "NPZ data - arrays: " + ", ".join(entry.get("arrays", []))
    return f"{fmt} file"  # pragma: no cover -- _file_details covers every suffix


def _file_resources() -> list[MCPResource]:
    """One :class:`MCPResource` per discoverable file under ``JENN_DIR``.

    Blocking (it stats and sniffs files), so callers run it in a thread.
    """
    # Never raise: that would blank the entire resource list, taking the
    # static `jenn://files` listing down with it.
    try:
        listing = _scan_files(_jenn_root())
    except OSError:  # pragma: no cover -- unreadable root
        return []
    resources: list[MCPResource] = []
    for entry in listing["files"][:_MAX_FILE_RESOURCES]:
        # A URI is posix-shaped whatever the platform, and safe="/" keeps a
        # nested name's separators as path structure -- which is what the
        # `{+path}` template matches on.
        name = PurePath(entry["name"]).as_posix()
        resources.append(
            MCPResource(
                uri=f"jenn://files/{quote(name, safe='/')}",
                name=name,
                title=name,
                description=_entry_summary(entry),
                mime_type="application/json",
            ),
        )
    return resources


def _csv_preview(path: Path) -> list[str]:
    """The first few physical lines of a CSV, each length-capped."""
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        return [
            line.rstrip("\r\n")[:_PREVIEW_CHARS]
            for line in itertools.islice(file, _PREVIEW_LINES)
        ]


@mcp.resource(
    "jenn://files/{+path}",
    name="jenn-file",
    description=(
        "One JENN file under $JENN_DIR: its format, columns or arrays (or a "
        "model's architecture), and the path to hand to `ingest`/`load_model`."
    ),
    mime_type="application/json",
)
def file(path: str) -> dict[str, Any]:
    """Describe one local JENN file, so it can be picked instead of typed.

    Returns what is needed to *choose* a file -- columns, arrays, model
    architecture, path, and a short CSV preview -- not its rows, which
    `ingest` keeps server-side.
    """
    root = _jenn_root()
    unknown = f"Unknown resource: jenn://files/{path}"
    # `path` is already percent-decoded and screened for traversal, absolute
    # paths, and null bytes by the SDK; `safe_join` re-checks against the
    # resolved root, which also catches symlinks pointing out of the tree.
    try:
        target = safe_join(root, path)
    except (PathEscapeError, ValueError) as err:
        raise ResourceNotFoundError(unknown) from err
    entry = _file_entry(target, root) if target.is_file() else None
    if entry is None:  # missing, or present but not a JENN data/model file
        raise ResourceNotFoundError(unknown)
    if entry.get("format") == "csv" and "error" not in entry:
        entry["preview"] = _csv_preview(target)
    entry["hint"] = (
        "Pass this path to `load_model`."
        if entry.get("kind") == "model"
        else "Pass this path to `ingest` (map inputs/outputs/derivatives to columns)."
    )
    return entry


# ----------------------------------------------------------
# --- PROMPTS ----------------------------------------------
# ----------------------------------------------------------


WORKFLOW = """\
Build a validated JENN surrogate:
0. Discover local files with the `jenn://files` resource (CSV/NPZ data + exported
   models). If the data is in a file, call `ingest` (map input/output columns and
   any available derivative columns) and train from the returned `dataset_id`;
   note which partials it reports as missing (they are gamma-masked to 0).
1. Infer a modest architecture from the data (few samples -> small net).
2. Call `train`; read `training_metrics` (value vs. per-partial R²).
3. GUARD: one run is stochastic. Re-run `train` with a different `random_state`
   and compare, and call `evaluate` on held-out data BEFORE any diagnosis.
4. Diagnose and adjust:
   - train R² >> holdout R²  -> overfitting -> raise `lambd` or shrink the net.
   - value R² good but a partial's R² low -> raise `gamma` (weights the Jacobian
     term); with a dataset you can boost just that partial via a per-partial
     `gamma` override.
5. When good enough for the intended use, call `export` and hand the file to the user.
   The export records the training input bounds, so a later session can reload it
   with `load_model` and still know the valid domain.
6. On every `predict`, check for an `extrapolation` key: it appears only when the
   query leaves the trained box. Treat a large `worst_overshoot` as a reason to
   clip the query to `trained`, re-scope the study, or gather training data out to
   `query` -- not as a number to report onward unqualified.
"""


@mcp.prompt()
def surrogate_workflow() -> str:
    """Recommended end-to-end workflow for building a JENN surrogate model."""
    return WORKFLOW
