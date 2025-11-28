# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from jenn.post_processing._metrics import rsquare


def plot_actual_by_predicted(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = 7,
    legend_label: str = "data",
    alpha: float = 0.5,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure | None:
    """Plot predicted vs. actual value.

    .. note::
        This method uses ravel(). A NumPy array with shape (n_y, m) becomes (n_y * m,).

    :param y_pred: predicted values for each dataset, list of arrays of shape (m,)
    :param y_true: true values for each dataset, list of arrays of shape (m,)
    :param figsize: figure size
    :param fontsize: text size to use for axis labels
    :param fontsize: text size to use for legend labels
    :param alpha: transparency of dots (between 0 and 1)
    :param ax: the matplotlib axes on which to plot the data
    :return: matplotlib Figure instance
    """
    if not legend_fontsize:
        legend_fontsize = fontsize
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
    r2 = rsquare(y_pred, y_true).squeeze()
    label = f"{legend_label} (" + r"$R^2$" + f"={r2:.2f})"
    ax.scatter(
        y_true, y_pred, alpha=alpha, color="gray", label=label, edgecolors="black"
    )

    # Add a perfect fit line to show deviations
    ymin = min(y_pred.min(), y_true.min())
    ymax = max(y_pred.max(), y_true.max())
    line = [ymin, ymax]
    ax.plot(line, line, color="r", linestyle=":", linewidth=1, label="perfect fit line")

    # Finish annotating axes
    ax.set_xlabel("Actual", fontsize=fontsize)
    ax.set_ylabel("Predicted", fontsize=fontsize)
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)
    ax.set_aspect("equal")

    plt.close(fig)
    return fig
