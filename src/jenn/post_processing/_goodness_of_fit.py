# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations  # needed if python is 3.9

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

import matplotlib.pyplot as plt

from ._actual_by_predicted import plot_actual_by_predicted
from ._histogram import plot_histogram
from ._residual_by_predicted import plot_residual_by_predicted


def plot_goodness_of_fit(
    y_pred: NDArray,
    y_true: NDArray,
    title: str = "",
    percent: bool = False,
) -> Figure:
    """Make goodness of fit summary plots."""
    fig, ax = plt.subplots(1, 3, figsize=(9.75, 3), layout="tight")
    fig.suptitle(title)
    plot_actual_by_predicted(y_pred, y_true, ax=ax[0])
    plot_histogram(y_pred, y_true, ax=ax[1], percent=percent)
    plot_residual_by_predicted(y_pred, y_true, ax=ax[2], percent=percent)
    return fig
