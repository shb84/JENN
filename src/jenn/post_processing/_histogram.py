# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure


def plot_histogram(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = 7,
    legend_label: str = "data",
    alpha: float = 0.75,
    percent: bool = False,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure | None:
    """Plot prediction error distribution.

    .. note::
        This method uses ravel(). A NumPy array with shape (n_y, m) becomes (n_y * m,).

    :param y_pred: predicted values for each dataset, list of arrays of shape (m,)
    :param y_true: true values for each dataset, list of arrays of shape (m,)
    :param figsize: figure size
    :param fontsize: text size to use for axis labels
    :param fontsize: text size to use for legend labels
    :param alpha: transparency of dots (between 0 and 1)
    :param percent: show residuals as percentages
    :param ax: the matplotlib axes on which to plot the data
    :return: matplotlib Figure instance
    """
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize, layout="tight")

    # Sanity check inputs
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Compute residuals
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    residuals = (y_pred - y_true) / y_true * 100 if percent else y_pred - y_true

    # Compute statistics
    avg = residuals.mean()
    std = residuals.std()

    # Make histogram
    ax.hist(
        residuals.ravel(),
        alpha=alpha,
        label=legend_label,
        color="gray",
        density=True,
        range=[avg - 6 * std, avg + 6 * std],
        bins=30,
    )

    # Add statistics
    avg = residuals.mean()
    std = residuals.std()
    ax.axvline(x=avg, color="r", linestyle="-", linewidth=1, label=f"avg = {avg:.3f}")
    ax.axvline(
        x=avg + std, color="r", linestyle=":", linewidth=1, label=f"std = {std:.3f}"
    )
    ax.axvline(x=avg - std, color="r", linestyle=":", linewidth=1)

    # Finish annotating axes
    ax.set_xlabel("Residuals (%)" if percent else "Residuals", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)

    plt.close(fig)
    return fig
