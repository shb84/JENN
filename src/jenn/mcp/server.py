"""MCP server.
==============

FastMCP server exposing JENN surrogate-modeling tools over stdio.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from jenn.core.model import NeuralNet
from jenn.post_processing.metrics import rsquare
from jenn.utilities import load_csv, load_npz

from ._convert import prepare_training_data
from ._store import DatasetRecord, ModelRecord, Registry

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as err:  # pragma: no cover
    msg = (
        "The JENN MCP server needs the optional 'mcp' dependency. "
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

mcp = FastMCP("jenn", instructions=INSTRUCTIONS)


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

    Masked-out partials (``mask == 0``) are reported under ``ignored`` rather
    than scored: their ``dydx`` values are placeholders, so an R² against them
    would be meaningless.
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

    :return: ``(x, y, dydx, mask, input_names, output_names)``. ``mask`` and the
        names are ``None`` for inline (positional) data.
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

    Scalar ``gamma`` on inline data passes straight through (backward
    compatible). With an availability ``mask``, the result is
    ``(mask * scale)`` shaped ``(n_y, n_x, 1)`` so it broadcasts over samples;
    availability always wins (a scale can never resurrect an absent partial).
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
    dydx=(m, n_y, n_x)) or by `dataset_id` from a prior `ingest` (mutually
    exclusive). With an ingested dataset, `gamma` may be a scalar OR a list of
    per-partial overrides, e.g. `[{"output":"Cd","input":"alpha","weight":3.0}]`;
    partials absent from the data are always weighted 0 (they cannot be
    resurrected by an override). The agent owns architecture/hyperparameters;
    this tool is a thin wrapper over jenn.NeuralNet.fit. See the returned
    `guidance` and re-run before acting on a single result.
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

    Pass a `dataset_id` (from `ingest`) or both x and y (row-per-sample) for
    held-out metrics; pass none of them to score the data the model was trained
    on. dydx is optional in every case.
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
    with jenn.NeuralNet.load(path).
    """
    record = _MODELS.get(model_id)  # raises KeyError on unknown id
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

    No naming convention is assumed. The gamma availability mask is built
    automatically from which partials are listed; absent partials are reported
    and will be weighted 0 at train time. Returns a `dataset_id` to pass to
    `train`/`evaluate`, keeping the data server-side (not re-sent per call).
    """
    target = Path(path).expanduser().resolve()
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

    Returns lightweight metadata only (shapes, column names, and which partials
    are available vs. missing); the arrays stay server-side, referenced by
    dataset_id.
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
# --- RESOURCES --------------------------------------------
# ----------------------------------------------------------


def _jenn_root() -> Path:
    """Directory scanned for JENN files: ``$JENN_DIR``, else the CWD."""
    return Path(os.environ.get("JENN_DIR", ".")).expanduser().resolve()


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
        "Local JENN files under $JENN_DIR (or the server CWD): CSV/NPZ data files "
        "to ingest, and exported model JSONs to load."
    ),
    mime_type="application/json",
)
def files() -> dict[str, Any]:
    """List local JENN data and model files to reference by path."""
    return _scan_files(_jenn_root())


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
"""


@mcp.prompt()
def surrogate_workflow() -> str:
    """Recommended end-to-end workflow for building a JENN surrogate model."""
    return WORKFLOW
