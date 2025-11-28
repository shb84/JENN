# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure


def plot_residual_by_predicted(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = 7,
    legend_label: str = "data",
    alpha: float = 0.5,
    percent: bool = False,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure | None:
    """Plot prediction error vs. predicted value.

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
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same length")

    # Loop over datasets to overlay them in one plot (e.g. train, test)
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    residuals = (y_pred - y_true) / y_true * 100 if percent else y_pred - y_true
    ax.scatter(
        y_pred,
        residuals,
        alpha=alpha,
        color="gray",
        label=legend_label,
        edgecolors="black",
    )

    # Add statistics
    avg = residuals.mean()
    std = residuals.std()
    ax.axhline(y=avg, color="r", linestyle="-", linewidth=1, label=f"avg = {avg:.3f}")
    ax.axhline(
        y=avg + std, color="r", linestyle=":", linewidth=1, label=f"std = {std:.3f}"
    )
    ax.axhline(y=avg - std, color="r", linestyle=":", linewidth=1)

    # Finish annotating axes
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("Residuals (%)" if percent else "Residuals", fontsize=fontsize)
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)

    plt.close(fig)
    return fig
