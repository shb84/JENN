from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure
from typing import Tuple


def plot_contours(
    func: Callable,
    lb: np.typing.ArrayLike,
    ub: np.typing.ArrayLike,
    x0: np.typing.ArrayLike | None = None,
    x_index: Tuple[int, int] = (0, 1),
    y_index: int = 0,  
    x_train: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    figsize: Tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 0.5,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    levels: int = 20,
    resolution: int = 100,
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
    :param x_index: indices of x to be plotted, e.g. (3, 9) implies x1=x[3] and x2=x[9]
    :param y_index: index of y to be plotted (if y is not scalar)
    :param x_train: option to overlay training data if provided 
    :param x_test: option to overlay test data if provided 
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param xlabel: factor #1 label
    :param ylabel: factor #2 label
    :param levels: number of contour levels
    :param resolution: line resolution
    :param ax: the matplotlib axes on which to plot the data
    :return: matplotlib figure instance
    """
    lb = np.array(lb)
    ub = np.array(ub)
    if x0 is None: 
        x0 = 0.5 * (lb + ub).reshape((-1, 1))

    # Domain
    m = resolution
    x1 = np.linspace(lb[x_index[0]], ub[x_index[0]], m)
    x2 = np.linspace(lb[x_index[1]], ub[x_index[1]], m)
    x1, x2 = np.meshgrid(x1, x2)

    # Response
    y = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            x = x0.copy()
            x[x_index[0]] = x1[i, j]
            x[x_index[1]] = x2[i, j]
            y[i, j] = func(x).ravel()[y_index]

    # Plot
    if ax:
        fig = ax.get_figure()
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.contour(x1, x2, y, levels, cmap="RdGy", alpha=alpha)
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
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.close(fig)
    return fig