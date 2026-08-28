# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JENN (Jacobian-Enhanced Neural Networks) — NumPy-based MLPs trained to minimize a loss that also fits the partial derivatives (Jacobians), giving better accuracy from fewer training points. No deep-learning framework; pure NumPy. Published as arXiv 2406.09132.

## Toolchain (pixi)

Environments and tasks are managed by `pixi`; the development env is `dev`.

- Full unit tests: `pixi run -e dev test-unit` · notebook tests: `pixi run -e dev test-nb`
- A single test (runs the `pytest` binary directly): `pixi run -e dev pytest tests/test_mcp.py::test_train_evaluate_export_roundtrip -q`
- Lint/type/docs (the release gates): `pixi run -e dev ruff`, `pixi run -e dev mypy`, `pixi run -e dev docs` · everything: `pixi run -e dev all`
- Editable install (also regenerates console scripts after `pyproject.toml` changes): `pixi run -e dev pip-e`

Gotchas:
- `pixi run -e dev ruff` / `mypy` run the *tasks* (e.g. ruff = `format --check && check` over the whole repo). For ad-hoc, path-scoped linting/formatting use `pixi run -e dev python -m ruff format|check <paths>`.
- Type-check via the `mypy` task, **not** `mypy src/` directly — the latter trips a mypy "Duplicate module named jenn" quirk.

## Conventions (ruff runs with `select = ["ALL"]`, preview on)

- Every source file needs the two-line copyright header (`# Copyright (C) 2018 Steven H. Berguin` / MIT) — enforced by CPY001.
- Module docstrings use an RST title + `====` underline (see any `core/` module).
- Suppress a rule with `# ruff:ignore[rule-name]` (the name, not the code) — `# noqa` is rejected in preview mode.
- The ignore list in `pyproject.toml` already permits boolean params, many-arg functions, and raised string messages — match the surrounding style rather than fighting it.

## Architecture

### Core (`src/jenn/core/`)
The neural-net math. `model.py` = the `NeuralNet` class (`fit` / `predict` / `predict_partials` / `__call__` / `save` / `load`); `propagation.py` (forward/backward + partials), `cost.py` (squared loss + a gradient-enhancement term weighted by `gamma`), `training.py`, `optimization.py` (ADAM + backtracking line search), `parameters.py` (weights/biases + JSON serde), `data.py`, `activation.py`.

**Data is feature-first throughout core:** `x` is `(n_x, m)`, `y` is `(n_y, m)`, `dydx` is `(n_y, n_x, m)` — features/outputs on the first axis, the `m` samples on the **last**. This is the single easiest thing to get wrong.

Notable `fit` hyperparameters: `gamma` (gradient-enhancement weight — may be an **array** for per-partial weighting, e.g. 0 to ignore a missing partial), `lambd` (L2), `is_normalize`. `jenn.metrics.rsquare(y_pred, y_true)` reduces over the last axis, so it works for both 2-D `(n_y, m)` and 3-D `(n_y, n_x, m)`; argument order is (prediction, truth).

### File-based data loading (`src/jenn/utilities/_load.py`)
`load_csv` / `load_npz` (exported from `jenn.utilities`) read training data from a file into core's **feature-first** arrays plus a Jacobian *availability mask* `(n_y, n_x)` — `1` where a partial is present, `0` where absent. CSV uses an **explicit** column-role mapping (`inputs`/`outputs` lists + `derivatives` as `(output, input, column)` triples — no naming convention). Absent partials are filled with `0.0` (a **finite** placeholder — never `NaN`, since the cost does `J_error *= sqrt(J_weights)` and `NaN * 0` would poison the loss); the mask, used as a `gamma` weight of 0, is what nullifies them. Framework-free (no MCP import) so any API user can use it. NPZ reads all-`NaN` layers as absent partials.

### MCP server (`src/jenn/mcp/`, optional `jenn[mcp]` extra, Python ≥ 3.10)
An `MCPServer` (mcp ≥ 2 — 1.x's `FastMCP` is gone) stdio server that lets an agent build and validate a surrogate. `server.py` holds the tools (`train` / `evaluate` / `export` / `list_models` / `ingest` / `list_datasets` + a `surrogate_workflow` prompt) and shared helpers (`_fit_metrics`, `_resolve_dataset`, `_effective_gamma`). `_convert.py` bridges the **row-per-sample** MCP boundary (`x` = `(m, n_x)`, etc.) to core's feature-first layout by transposing (inline arrays only; the file loaders already return feature-first). `_store.py` holds two session-scoped in-memory registries: `model_id → ModelRecord` and `dataset_id → DatasetRecord` (both die with the process; `export` is how a model persists). `ingest` wraps `jenn.utilities.load_csv/load_npz` into a dataset; `train`/`evaluate` accept a `dataset_id` (keeps large arrays off the agent's context) and `train`'s `gamma` becomes `float | list[dict]` — a scalar scale, or per-partial overrides on a **named** (ingested) dataset — multiplied by the availability mask into a `(n_y, n_x, 1)` array before `fit`. Masked-out partials are reported under `partials.ignored` in metrics, not scored. Resources: `jenn://files` is the whole-folder listing, and `_JennMCP.list_resources` (a subclass override — the SDK only serves statically registered resources) adds one live `jenn://files/<name>` entry per file so an agent's `@` menu can browse them; reads go through the `jenn://files/{+path}` template and return a metadata card, not the file's rows. Tests guard with `importorskip("mcp")`. Launch with `jenn-mcp` or `python -m jenn.mcp`.

## Release
The version lives in `src/jenn/__init__.py`; pushing a tag drives the release pipeline. See `CONTRIBUTING.md`.
