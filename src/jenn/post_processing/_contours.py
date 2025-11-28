# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure


def plot_contours(  # noqa: C901
    func: Callable,
    x_min: np.ndarray,
    x_max: np.ndarray,
    x0: np.ndarray | None = None,
    x1_index: int = 0,
    x2_index: int = 1,
    y_index: int = 0,
    x_train: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 0.5,
    title: str = "",
    x1_label: str | None = None,
    x2_label: str | None = None,
    y_label: str | None = None,
    levels: int = 20,
    resolution: int = 100,
    show_colorbar: bool = False,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure | None:
    """Plot contours of a scalar function of two variables, y = f(x1, x2).

    .. note::
        This method takes in a function of signature form y=f(x)
        and maps it onto a function of signature form y=f(x1, x2)
        such that the contours can be plotted.

    :param func: the function to be evaluate, y = f(x)
    :param lb: lower bounds on x
    :param ub: upper bounds on x
    :param x1_index: index of x to use for factor #1
    :param x2_index: index of x to use for factor #2
    :param y_index: index of y to be plotted
    :param x_train: option to overlay training data if provided
    :param x_test: option to overlay test data if provided
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param x1_label: factor #1 label
    :param x2_label: factor #1 label
    :param y_label: response label
    :param levels: number of contour levels
    :param resolution: line resolution
    :param show_colorbar: show the colorbar
    :param ax: the matplotlib axes on which to plot the data
    :return: matplotlib figure instance
    """
    if x0 is None:
        x0 = 0.5 * (x_min + x_max).reshape((-1, 1))
    if x1_label is None:
        x1_label = f"x{x1_index}"
    if x2_label is None:
        x2_label = f"x{x2_index}"
    if y_label is None:
        y_label = f"y{y_index}"

    # Domain
    m = resolution
    x1 = np.linspace(x_min[x1_index], x_max[x1_index], m)
    x2 = np.linspace(x_min[x2_index], x_max[x2_index], m)
    x1, x2 = np.meshgrid(x1, x2)

    # Response
    y = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            x = x0.copy()
            x[x1_index] = x1[i, j]
            x[x2_index] = x2[i, j]
            y[i, j] = func(x).ravel()[y_index]

    # Plot
    if ax:
        fig = ax.get_figure()
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    cs = ax.contour(x1, x2, y, levels, cmap="RdGy", alpha=alpha)
    if show_colorbar:
        cbar = plt.colorbar(cs, shrink=1, location="right")
        cbar.set_label(y_label)  # Label for the colorbar
    legend = []
    if x_train is not None:
        ax.scatter(x_train[0], x_train[1], marker=".", c="k", alpha=1)
        legend.append("train")
    if x_test is not None:
        ax.scatter(x_test[0], x_test[1], marker="+", c="r", alpha=1)
        legend.append("test")
    if legend:
        ax.legend(legend, loc=1)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x1_label, fontsize=fontsize)
    ax.set_ylabel(x2_label, fontsize=fontsize)
    plt.close(fig)
    return fig
