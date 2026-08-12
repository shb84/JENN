r"""Load training data from files.
=================================

Read training data (inputs, outputs, and optionally partial derivatives)
from a file into the feature-first arrays used by :mod:`jenn`, so that
callers never have to hand-assemble numpy arrays.

Two entry points are provided:

- :func:`load_csv` -- a wide, one-row-per-sample table with an explicit
  column-role mapping. Derivative columns are paired to a ``(output, input)``
  partial explicitly, so real column names need not follow any convention.
- :func:`load_npz` -- a native escape hatch: a ``.npz`` holding arrays named
  ``x``, ``y`` and (optionally) ``dydx``, already feature-first.

Both return ``(x, y, dydx, mask)`` where ``mask`` is the Jacobian
*availability* mask: a ``(n_y, n_x)`` array holding ``1`` where the partial
:math:`\partial y_i / \partial x_j` is present in the data and ``0`` where
it is absent. Absent partials are filled with ``0`` in ``dydx`` (a finite
placeholder); the mask is what tells a trainer to ignore them (e.g. by using
it as a ``gamma`` weight of ``0``).
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations  # needed if python is 3.9

import csv
from pathlib import Path

import numpy as np


def _read_columns(path: Path, delimiter: str) -> dict[str, np.ndarray]:
    """Read a numeric CSV into a mapping of column name -> ``(m,)`` array.

    :param path: path to the CSV file
    :param delimiter: field separator (e.g. ``","`` or ``";"``)
    :return: dict mapping each header name to its column as a float array
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=delimiter)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            msg = f"`{path}` is empty; expected a header row."
            raise ValueError(msg) from None
        n_cols = len(header)
        rows: list[list[str]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:  # tolerate blank lines
                continue
            if len(row) != n_cols:
                msg = (
                    f"`{path}` line {line_no} has {len(row)} fields, expected {n_cols}."
                )
                raise ValueError(msg)
            rows.append(row)
    if not rows:
        msg = f"`{path}` has a header but no data rows."
        raise ValueError(msg)
    try:  # one conversion for the whole table, so the error path stays out of the loop
        data = np.asarray(rows, dtype=float)  # (m, n_cols)
    except ValueError:
        msg = f"`{path}` contains a value that cannot be parsed as a number."
        raise ValueError(msg) from None
    return {name: data[:, j] for j, name in enumerate(header)}


def _stack(
    columns: dict[str, np.ndarray],
    names: list[str],
    role: str,
    path: Path,
) -> np.ndarray:
    """Stack named columns into a feature-first ``(len(names), m)`` array."""
    missing = [name for name in names if name not in columns]
    if missing:
        msg = (
            f"{role} columns {missing} not found in `{path}`. "
            f"Available columns: {list(columns)}."
        )
        raise ValueError(msg)
    return np.vstack([columns[name] for name in names])


def load_csv(
    path: str | Path,
    inputs: list[str],
    outputs: list[str],
    derivatives: list[dict[str, str]] | None = None,
    delimiter: str = ",",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    r"""Load training data from a wide, one-row-per-sample CSV file.

    Column roles are set by explicit mapping, never by a naming convention.
    ``inputs`` and ``outputs`` name the value columns (in the desired order).
    ``derivatives`` pairs each *available* partial with the column holding it,
    one entry per partial::

        {"output": <name in outputs>, "input": <name in inputs>,
         "column": <column holding d(output)/d(input)>}

    Any ``(output, input)`` pair not listed is treated as unavailable: its
    ``dydx`` entry is filled with ``0`` and its ``mask`` entry is ``0``.

    :param path: path to the CSV file
    :param inputs: column names mapped to ``x``, in order
    :param outputs: column names mapped to ``y``, in order
    :param derivatives: explicit ``(output, input, column)`` mapping for the
        partials that are present; ``None`` (or empty) means no Jacobian
    :param delimiter: field separator (e.g. ``","`` or ``";"``)
    :return: tuple ``(x, y, dydx, mask)`` with feature-first shapes ``(n_x, m)``,
        ``(n_y, m)``, ``(n_y, n_x, m)`` and ``(n_y, n_x)``. ``dydx`` and ``mask``
        are ``None`` when no derivatives are given.
    """
    path = Path(path).expanduser()
    columns = _read_columns(path, delimiter)
    x = _stack(columns, inputs, "input", path)  # (n_x, m)
    y = _stack(columns, outputs, "output", path)  # (n_y, m)

    if not derivatives:
        return x, y, None, None

    n_x, m = x.shape
    n_y = y.shape[0]
    dydx = np.zeros((n_y, n_x, m))
    mask = np.zeros((n_y, n_x))
    for spec in derivatives:
        missing_keys = {"output", "input", "column"} - spec.keys()
        if missing_keys:
            msg = (
                f"derivative entry {spec} is missing key(s) "
                f"{sorted(missing_keys)}; each entry needs 'output', "
                f"'input' and 'column'."
            )
            raise ValueError(msg)
        out, inp, col = spec["output"], spec["input"], spec["column"]
        if out not in outputs:
            msg = f"derivative output '{out}' is not one of outputs {outputs}."
            raise ValueError(msg)
        if inp not in inputs:
            msg = f"derivative input '{inp}' is not one of inputs {inputs}."
            raise ValueError(msg)
        if col not in columns:
            msg = f"derivative column '{col}' not found in `{path}`."
            raise ValueError(msg)
        oi, ii = outputs.index(out), inputs.index(inp)
        dydx[oi, ii, :] = columns[col]
        mask[oi, ii] = 1.0
    return x, y, dydx, mask


def load_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    r"""Load feature-first training data from a ``.npz`` file.

    The archive must hold arrays named ``x`` ``(n_x, m)`` and ``y`` ``(n_y, m)``,
    and may hold ``dydx`` ``(n_y, n_x, m)``. This is the native escape hatch: the
    arrays are consumed as-is, with no column mapping.

    Partial *availability* is read from ``NaN`` markers: a ``(output, input)``
    layer that is entirely ``NaN`` is treated as absent (``mask`` ``0``, values
    replaced with ``0``). A partial must be either fully present or fully absent;
    a layer with only some ``NaN`` samples raises (per-sample availability is not
    supported).

    :param path: path to the ``.npz`` file
    :return: tuple ``(x, y, dydx, mask)`` with feature-first shapes ``(n_x, m)``,
        ``(n_y, m)``, ``(n_y, n_x, m)`` and ``(n_y, n_x)``. ``dydx`` and ``mask``
        are ``None`` when the archive has no ``dydx``.
    """
    path = Path(path).expanduser()
    with np.load(path) as archive:
        for key in ("x", "y"):
            if key not in archive:
                msg = f"`{path}` has no array named '{key}'."
                raise ValueError(msg)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        dydx = np.asarray(archive["dydx"], dtype=float) if "dydx" in archive else None

    if x.ndim != 2 or y.ndim != 2:
        msg = (
            f"`x` and `y` in `{path}` must be 2-D feature-first arrays; "
            f"got x.ndim={x.ndim}, y.ndim={y.ndim}."
        )
        raise ValueError(msg)
    if x.shape[1] != y.shape[1]:
        msg = (
            f"`x` and `y` in `{path}` disagree on sample count: "
            f"x has {x.shape[1]}, y has {y.shape[1]}."
        )
        raise ValueError(msg)

    if dydx is None:
        return x, y, None, None

    n_x, m = x.shape
    n_y = y.shape[0]
    if dydx.shape != (n_y, n_x, m):
        msg = (
            f"`dydx` in `{path}` has shape {dydx.shape}; expected "
            f"(n_y, n_x, m) = {(n_y, n_x, m)}."
        )
        raise ValueError(msg)

    is_nan = np.isnan(dydx)
    missing = is_nan.all(axis=2)  # (n_y, n_x) fully-absent partials
    partial = is_nan.any(axis=2) & ~missing
    if partial.any():
        msg = (
            f"`dydx` in `{path}` has partials that are NaN in only some samples; "
            "a partial must be fully present or fully absent (all-NaN)."
        )
        raise ValueError(msg)
    mask = (~missing).astype(float)
    dydx = np.where(is_nan, 0.0, dydx)  # finite placeholder for absent partials
    return x, y, dydx, mask
