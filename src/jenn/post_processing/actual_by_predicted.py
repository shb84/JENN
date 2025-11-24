import matplotlib.pyplot as plt
import numpy as np
from typing import List
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from jenn.post_processing.metrics import r_square
        

def plot_actual_by_predicted(
    y_pred: np.ndarray | List[np.ndarray],
    y_true: np.ndarray | List[np.ndarray],
    response: str = "Response(s)",
    datasets: List[str] | None = None,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = None,
    alpha: float = 1.0,
) -> Figure:
    """Create actual by predicted plot for a single response.

    .. note::
        If shape of input NumPy arrays is two dimensional or more, it will 
        be called with ravel(). For example, y.shape = (n_y, m) will become 
        y.shape = (n_y * m,) which implies all data will be melted together. 
        Individual responses will not be treated separately. This is helpful 
        to get an aggregate understand of the goodness of fit across the board. 
        If goodness of fit is desired for each response, then this method must 
        be called separately for each one, e.g. plot_actual_by_predicted(y_pred[i], y_true[i])

    :param y_pred: predicted values for each dataset, list of arrays of shape (m,)
    :param y_true: true values for each dataset, list of arrays of shape (m,)
    :param datasets: names of datasets (e.g. ["train", "val", "test"])
    :param figsize: figure size
    :param response: name of response
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param melt: plot all responses in one plot
    :return: matplotlib figure instance
    """
    if not legend_fontsize: 
        legend_fontsize = fontsize 

    # Sanity check inputs 
    if isinstance(y_pred, np.ndarray): 
        y_pred = [y_pred]
    if isinstance(y_true, np.ndarray): 
        y_true = [y_true]
    if not isinstance(y_pred, list): 
        raise ValueError("ypred must be a list of arrays, e.g. [y_train_pred, y_test_pred]")
    if not isinstance(y_true, list): 
        raise ValueError("ytrue must be a list of arrays, e.g. [y_train_true, y_test_true]")
    if len(y_true) != len(y_pred): 
        raise ValueError("y_true and y_pred must have same length")
    if datasets is None: 
        datasets = ["data"] * len(y_true)
    if len(y_true) != len(datasets): 
        raise ValueError("y_true and y_pred must have same length as datasets")
    
    # Instantiate lists 
    legend = [] 
    lower = []
    upper = []

    # Loop over datasets to overlay them in one plot (e.g. train, test)
    fig, ax = plt.subplots(figsize=figsize)
    for i, dataset in enumerate(datasets): 
        pred = y_pred[i].ravel()
        true = y_true[i].ravel()
        ax.scatter(true, pred, alpha=alpha)
        r2 = r_square(pred, true).squeeze() 
        legend.append(f"{dataset} (" + r'$R^2$' + f"={r2:.2f})")
        lower.append(true.min())
        upper.append(true.max())
    
    # Add a perfect fit line to show deviations 
    legend.append("perfect fit line")
    line = [min(lower), max(upper)]
    ax.plot(line, line, color="k", linestyle=":")

    # Finish annotating axes 
    ax.set_xlabel(f"Actual {response}", fontsize=fontsize)
    ax.set_ylabel(f"Predicted {response}", fontsize=fontsize)
    ax.grid(True)
    ax.legend(legend, fontsize=legend_fontsize)

    plt.axis("equal")
    plt.close(fig)
    return fig 


# def actual_by_predicted(
#     y_true: np.ndarray | List[np.ndarray],
#     y_pred: np.ndarray | List[np.ndarray],
#     figsize: tuple[float, float] = (3.25, 3),
#     fontsize: int = 9,
#     alpha: float = 1.0,
#     datasets: str | List[str] = "data", 
#     responses: str | List[str] = "response", 
# ) -> Figure:
#     """Create an actual by predicted plot for each response.

#     :param y_true: true values, single array or list of arrays of shape (n_y, m)
#     :param y_pred: predicted values, single array or list of arrays of shape (n_y, m)
#     :param percent_residuals: shows residuals as percentages if True
#     :param figsize: figure size
#     :param fontsize: text size
#     :param alpha: transparency of dots (between 0 and 1)
#     :param title: title of figure
#     :return: matplotlib figure instance
#     """
#     # Sanity check inputs 
#     if isinstance(responses, str): 
#         responses = [responses]
#     if isinstance(datasets, str): 
#         datasets = [datasets]
#     if not isinstance(y_pred, list): 
#         raise ValueError("y_pred must be a list of arrays, e.g. [y_train_pred, y_test_pred]")
#     if not isinstance(y_true, list): 
#         raise ValueError("y_true must be a list of arrays, e.g. [y_train_true, y_test_true]")
#     if len(y_true) != len(y_pred): 
#         raise ValueError("y_true and y_pred must have same length")
#     if len(y_true) != len(datasets): 
#         raise ValueError("y_true and y_pred must have same length as datasets")

#     # Ensure arrays of shape (n_y, m)
#     y_true = [_ensure_2D_array(y_true[i], name=f"{dataset} (true)") for i, dataset in enumerate(datasets)]
#     y_pred = [_ensure_2D_array(y_pred[i], name=f"{dataset} (pred)") for i, dataset in enumerate(datasets)]
        
#     # Make figure 
#     fig, ax = plt.subplots(nrows=len(responses), ncols=1, figsize=figsize)
#     for i, response in enumerate(responses): 
#         _actual_by_predicted(
#             y_true=[y_true[k][i] for k, _ in enumerate(datasets)],
#             y_pred=[y_pred[k][i] for k, _ in enumerate(datasets)],
#             labels=datasets, 
#             ax=ax[i], 
#             title=response, 
#             fontsize=fontsize,
#             alpha=alpha,
#         )
#     plt.close(fig)
#     return fig