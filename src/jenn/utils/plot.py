"""Plotting.
============

This module provides optional but helpful utilities to 
assess goodness of fit and visualize trends. 

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn 

    # Assuming the following are available: 
    x_train, y_train, dydx_train = _ # user provided
    x_test, y_test, dydx_test = _    # user provided
    nn = _                           # previously trained NeuralNet

    # Show goodness of fit of the partials 
    i = 0  # index of the response to plot
    jenn.utils.plot.goodness_of_fit(
        y_true=dydx_test[i], 
        y_pred=nn.predict_partials(x_test)[i], 
        title="Partial Derivative: dy/dx (NN)"
    )

    # Example: visualize local trends
    jenn.utils.plot.sensitivity_profiles(
        f=[nn.predict], 
        x_min=x_train.min(), 
        x_max=x_train.max(), 
        x_true=x_train, 
        y_true=y_train, 
        resolution=100, 
        legend=['nn'], 
        xlabels=['x'], 
        ylabels=['y'],
    )
"""  # noqa: W291

from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .metrics import r_square

if find_spec("matplotlib"):
    MATPLOTLIB_INSTALLED = True
    import matplotlib.pyplot as plt
else:
    MATPLOTLIB_INSTALLED = False


LINE_STYLES = {
    "solid": "solid",  # Same as (0, ()) or '-'
    "dotted": "dotted",  # Same as (0, (1, 1)) or ':'
    "dashdot": "dashdot",  # Same as '-.'
    "dashed": "dashed",  # Same as '--'
    #
    "loosely dotted": (0, (1, 10)),
    # "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashed": (0, (5, 10)),
    # "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    #
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    #
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def requires_matplotlib(func: Callable) -> Callable:
    """Return error if matplotlib not installed."""

    @wraps(func)
    def wrapper(*args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        if MATPLOTLIB_INSTALLED:
            return func(*args, **kwargs)
        raise ValueError("Matplotlib is not installed.")

    return wrapper


@requires_matplotlib
def actual_by_predicted(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ax: Union[plt.Axes, None] = None,  # noqa: ANN401
    figsize: Tuple[float, float] = (3.25, 3),
    title: str = "",
    fontsize: int = 9,
    alpha: float = 1.0,
) -> plt.Figure:  # noqa: ANN401
    """Create actual by predicted plot for a single response.

    :param y_pred: predicted values, array of shape (m,)
    :param y_true: true values, array of shape (m,)
    :param ax: the matplotlib axes on which to plot the data
    :param figsize: figure size
    :param title: title of figure
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :return: matplotlib figure instance
    """
    if y_pred.ndim > 1:
        if y_pred.ndim == 2 and 1 in y_pred.shape:
            pass  # just needs to be unraveled
        else:
            raise ValueError(
                f"Expected one dimensional array, "
                f"but y_pred has {y_pred.ndim} dimensions."
            )
    if y_true.ndim > 1 and 1 not in y_true.shape:
        if y_true.ndim == 2 and 1 in y_true.shape:
            pass  # just needs to be unraveled
        else:
            raise ValueError(
                f"Expected one dimensional array, "
                f"but y_true has {y_true.ndim} dimensions."
            )
    fig = plt.figure(figsize=figsize, layout="tight")
    if not ax:
        spec = fig.add_gridspec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0, 0])
    actual = y_true.ravel()
    predicted = y_pred.ravel()
    ax.scatter(actual, predicted, color="k", alpha=alpha)
    line = [actual.min(), actual.max()]
    ax.plot(line, line, color="r", linestyle=":")
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("Actual", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)
    ax.legend(["predictions", "perfect fit line"], fontsize=fontsize)
    plt.close(fig)
    return fig


def contours(
    func: Callable,
    lb: Tuple[float, float],
    ub: Tuple[float, float],
    x_train: Union[np.ndarray, None] = None,
    x_test: Union[np.ndarray, None] = None,
    figsize: Tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 0.5,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    levels: int = 20,
    resolution: int = 100,
    ax: Union[plt.Axes, None] = None,  # noqa: ANN401
) -> Union[None, plt.Figure]:  # noqa: ANN401
    """Plot contours of a scalar function of two variables.

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
    # Domain
    m = resolution
    x1 = np.linspace(lb[0], ub[0], m)
    x2 = np.linspace(lb[1], ub[1], m)
    x1, x2 = np.meshgrid(x1, x2)

    # Response
    y = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            y[i, j] = func(np.array([[x1[i, j]], [x2[i, j]]])).ravel()[0]

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


@requires_matplotlib
def convergence(
    histories: List[Dict[str, Dict[str, List[float]]]],
    figsize: Tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
    legend: Union[List[str], None] = None,
) -> Union[plt.Figure, None]:  # noqa: ANN401
    """Plot training history.

    :param histories: training history for each model
    :param figsize: subfigure size of each subplot
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param legend: label for each model
    :return: matplotlib figure instance
    """
    if not histories:
        return None

    fig = plt.figure(figsize=figsize, layout="tight")
    fig.suptitle(title)

    linestyles = iter(LINE_STYLES.values())
    for history in histories:
        linestyle = next(linestyles)
        epochs = list(history.keys())
        if len(epochs) > 1:
            avg_costs = []
            for epoch in epochs:
                batches = history[epoch].keys()
                avg_batch_costs = []
                for batch in batches:
                    avg_batch_cost = np.mean(history[epoch][batch])
                    avg_batch_costs.append(avg_batch_cost)
                avg_costs.append(sum(avg_batch_costs) / len(batches))
            plt.plot(
                range(len(epochs)),
                np.array(avg_costs),
                alpha=alpha,
                color="k",
                linewidth=2,
                linestyle=linestyle,
            )
            plt.xlabel("epoch", fontsize=fontsize)
            plt.ylabel("avg cost", fontsize=fontsize)
        elif len(history["epoch_0"]) > 1:
            avg_cost = []
            batches = history["epoch_0"].keys()
            for batch in batches:
                avg_cost.append(np.mean(history["epoch_0"][batch]))
            plt.plot(
                range(len(batches)),
                avg_cost,
                alpha=alpha,
                color="k",
                linewidth=2,
                linestyle=linestyle,
            )
            plt.xlabel("batch", fontsize=fontsize)
            plt.ylabel("avg cost", fontsize=fontsize)
        else:
            cost = history["epoch_0"]["batch_0"]
            plt.plot(
                range(len(cost)),
                cost,
                alpha=alpha,
                color="k",
                linewidth=2,
                linestyle=linestyle,
            )
            plt.xlabel("iteration", fontsize=fontsize)
            plt.ylabel("cost", fontsize=fontsize)
    ax = plt.gca()
    if legend:
        ax.legend(legend)
    ax.set_yscale("log")
    plt.close(fig)
    return fig


@requires_matplotlib
def residuals_by_predicted(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    percent_residuals: bool = False,
    ax: Union[plt.Axes, None] = None,  # noqa: ANN401
    figsize: Tuple[float, float] = (3.25, 3),
    title: str = "",
    fontsize: int = 9,
    alpha: float = 1.0,
) -> plt.Figure:  # noqa: ANN401
    """Create residual by predicted plot for a single response.

    :param y_pred: predicted values, array of shape (m,)
    :param y_true: true values, array of shape (m,)
    :param percent_residuals: shows residuals as percentages if True
    :param ax: the matplotlib axes on which to plot the data
    :param figsize: figure size
    :param title: title of figure
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :return: matplotlib figure instance
    """
    if y_pred.ndim > 1:
        if y_pred.ndim == 2 and 1 in y_pred.shape:
            pass  # just needs to be unraveled
        else:
            raise ValueError(
                f"Expected one dimensional array, "
                f"but y_pred has {y_pred.ndim} dimensions."
            )
    if y_true.ndim > 1 and 1 not in y_true.shape:
        if y_true.ndim == 2 and 1 in y_true.shape:
            pass  # just needs to be unraveled
        else:
            raise ValueError(
                f"Expected one dimensional array, "
                f"but y_true has {y_true.ndim} dimensions."
            )
    fig = plt.figure(figsize=figsize, layout="tight")
    if not ax:
        spec = fig.add_gridspec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0, 0])
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if percent_residuals:
        residuals = 100 * ((y_pred - y_true) / (y_true + 1e-12)).ravel()
    else:
        residuals = y_pred - y_true
    avg_error = residuals.mean()
    std_error = residuals.std()
    ax.axhline(y=avg_error, color="k", linestyle="-", linewidth=2)
    ax.axhline(y=avg_error + std_error, color="k", linestyle=":", linewidth=2)
    ax.axhline(y=avg_error - std_error, color="k", linestyle=":", linewidth=2)
    ax.scatter(y_pred, residuals, color="k", alpha=alpha)
    ax.axhline(y=0, color="r", linestyle=":")
    ax.set_title(title, fontsize=fontsize)
    if percent_residuals:
        ax.set_ylabel("Residuals (%)", fontsize=fontsize)
    else:
        ax.set_ylabel("Residuals", fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.grid(True)
    ax.legend([f"avg = {avg_error:.3f}", f"std = {std_error:.3f}"], fontsize=fontsize)
    plt.close(fig)
    return fig


@requires_matplotlib
def goodness_of_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percent_residuals: bool = False,
    figsize: Tuple[float, float] = (6.5, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
) -> plt.Figure:  # noqa: ANN401
    """Create 'residual by predicted' and 'actual by predicted' plots.

    :param y_true: true values, array of shape (m,)
    :param y_pred: predicted values, array of shape (m,)
    :param percent_residuals: shows residuals as percentages if True
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :return: matplotlib figure instance
    """
    if title is None:
        title = ""
    r2 = r_square(y_pred, y_true).squeeze()
    fig = plt.figure(figsize=figsize, layout="tight")
    fig.suptitle(title + f" (R-Squared = {r2:.3f})")
    spec = fig.add_gridspec(ncols=2, nrows=1)
    ax0 = fig.add_subplot(spec[0, 0])
    actual_by_predicted(
        ax=ax0,
        y_pred=y_pred,
        y_true=y_true,
        fontsize=fontsize,
        alpha=alpha,
    )
    ax1 = fig.add_subplot(spec[0, 1])
    residuals_by_predicted(
        ax=ax1,
        y_pred=y_pred,
        y_true=y_true,
        percent_residuals=percent_residuals,
        fontsize=fontsize,
        alpha=alpha,
    )
    plt.close(fig)
    return fig


@requires_matplotlib
def sensitivity_profile(
    ax: plt.Axes,  # noqa: ANN401
    x0: np.ndarray,
    y0: np.ndarray,
    x_pred: np.ndarray,
    y_pred: Union[np.ndarray, List[np.ndarray]],
    x_true: Union[np.ndarray, None] = None,
    y_true: Union[np.ndarray, None] = None,
    alpha: float = 1.0,
    xlabel: str = "x",
    ylabel: str = "y",
    legend: Union[List[str], None] = None,
    figsize: Tuple[float, float] = (6.5, 3),
    fontsize: int = 9,
    show_cursor: bool = True,
) -> plt.Figure:  # noqa: ANN401
    """Plot sensitivity profile for a single input, single output.

    :param ax: the matplotlib axes on which to plot the data
    :param x0: point at which the profile is centered, array of shape
        (1,)
    :param y0: model evaluated as x0, list of arrays of shape (1,)
    :param x_pred: input values for prediction, an array of shape (m,)
    :param y_pred: predicted output values for each model, list of
        arrays of shape (m,)
    :param x_true: inputs value of actual data, array of shape (m, n_x)
    :param y_true: output values of actual data. An array of shape (m,)
    :param alpha: transparency of dots (between 0 and 1)
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param legend: legend name of each model
    :param figsize: figure size
    :param fontsize: text size
    :param show_cursor: show x0 as a red dot (or not)
    :return: matplotlib figure instance
    """
    fig = plt.figure(figsize=figsize, layout="tight")
    if not ax:
        spec = fig.add_gridspec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0, 0])
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if legend is None:
        legend = []
    x0 = x0.ravel()
    y0 = y0.ravel()
    x_pred = x_pred.ravel()
    linestyles = iter(LINE_STYLES.values())
    for array in y_pred:
        linestyle = next(linestyles)
        ax.plot(x_pred, array.ravel(), color="k", linestyle=linestyle, linewidth=2)
    if x_true is not None and y_true is not None:
        x_true = x_true.ravel()
        y_true = y_true.ravel()
        ax.scatter(x_true, y_true, color="k", alpha=alpha)
        legend.append("data")
    ax.legend(legend, fontsize=fontsize)
    if show_cursor:
        for n in range(y0.size):
            ax.scatter(x0, y0[n], color="r")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True)
    plt.close(fig)
    return fig


@requires_matplotlib
def sensitivity_profiles(
    f: Union[Callable, List[Callable]],
    x_min: np.ndarray,
    x_max: np.ndarray,
    x0: Union[np.ndarray, None] = None,
    x_true: Union[np.ndarray, None] = None,
    y_true: Union[np.ndarray, None] = None,
    figsize: Tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
    xlabels: Union[List[str], None] = None,
    ylabels: Union[List[str], None] = None,
    legend: Union[List[str], None] = None,
    resolution: int = 100,
    show_cursor: bool = True,
) -> plt.Figure:  # noqa: ANN401
    """Plot grid of all outputs vs. all inputs evaluated at x0.

    :param f: callable function(s) for evaluating y_pred = f_pred(x)
    :param x0: point at which to evaluate profiles, array of shape (n_x, 1)
    :param x_true: inputs at which y_true is evaluated, array of shape (n_x, m)
    :param y_true: true values, array of shape (n_y, m)
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param xlabels: x-axis labels
    :param ylabels: y-axis labels
    resolution: line resolution
    :param legend: legend labels for each model
    :param show_cursor: show x0 as a red dot (or not)
    """
    funcs = f
    if not isinstance(funcs, list):
        funcs = [funcs]
    x_min = x_min.ravel()
    x_max = x_max.ravel()
    if x0 is None:
        x0 = 0.5 * (x_min + x_max).reshape((-1, 1))
    y0 = np.concatenate([func(x0) for func in funcs], axis=1)
    n_x = x0.shape[0]
    n_y = y0.shape[0]
    x_indices = range(n_x)
    y_indices = range(n_y)
    xlabels = xlabels if xlabels else [f"x_{i}" for i in x_indices]
    ylabels = ylabels if ylabels else [f"y_{i}" for i in y_indices]
    width, height = figsize
    fig = plt.figure(figsize=(n_x * width, height), layout="tight")
    fig.suptitle(title)
    spec = fig.add_gridspec(ncols=n_x, nrows=n_y)
    for i in x_indices:
        x_pred = np.tile(x0, (1, resolution))
        x_pred[i] = np.linspace(x_min[i], x_max[i], resolution)
        y_preds = []
        for func in funcs:
            y_pred = func(x_pred)
            y_preds.append(y_pred)
        for j in y_indices:
            sensitivity_profile(
                ax=fig.add_subplot(spec[j, i]),
                x0=x0[i],
                y0=y0[j],
                x_pred=x_pred[i],
                y_pred=[y_pred[j] for y_pred in y_preds],
                x_true=x_true[i] if x_true is not None else None,
                y_true=y_true[i] if y_true is not None else None,
                fontsize=fontsize,
                alpha=alpha,
                xlabel=xlabels[i],
                ylabel=ylabels[j],
                legend=legend,
                show_cursor=show_cursor,
            )
        plt.close(fig)
    return fig
