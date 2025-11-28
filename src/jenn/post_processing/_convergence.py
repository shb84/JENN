# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from ._styling import LINE_STYLES

History = dict[str, dict[str, list[float]]]


def plot_convergence(  # noqa: PLR0912, C901
    histories: History | list[History],
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
    legend: list[str] | None = None,
    is_xlog: bool = False,
    is_ylog: bool = True,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure | None:
    """Plot training history.

    :param histories: training history for each model
    :param figsize: subfigure size of each subplot
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param legend: label for each model
    :param is_xlog: use log scale for x-axis
    :param is_ylog: use log scale for y-axis
    :param ax: the matplotlib axes on which to plot the data
    :return: matplotlib figure instance
    """
    if not isinstance(histories, list):
        histories = [histories]
    if not histories:
        return None

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize, layout="tight")
    fig.suptitle(title)  # type: ignore [union-attr]

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
            batches = history["epoch_0"].keys()
            avg_cost = [np.mean(history["epoch_0"][batch]) for batch in batches]
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
    if is_xlog:
        ax.set_xscale("log")
    if is_ylog:
        ax.set_yscale("log")
    plt.grid(True)
    plt.close(fig)
    return fig
