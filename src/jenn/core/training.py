"""Train neural network."""
import functools
from collections import defaultdict

import numpy as np

from .cache import Cache
from .cost import Cost
from .data import Dataset
from .optimization import ADAM, Backtracking, Optimizer
from .parameters import Parameters
from .propagation import model_backward, model_partials_forward


def objective_function(
    X: np.ndarray,
    cost: Cost,
    parameters: Parameters,
    cache: Cache,
    stacked_params: np.ndarray,
) -> np.float64:
    """Evaluate cost function for training.

    Parameters
    ----------
    X: np.ndarray
        Training data inputs. An array of shape (n_x, m)
        where n_x = number of inputs
                m = number of examples

    cost: Cost
        Cost function to be evaluated.

    parameters: Parameters
        Neural net parameters. Object that stores
        neural net parameters for each layer.

    cache: Cache
        Neural net cache. Object that stores
        neural net quantities for each layer,
        during forward prop, so they can be
        accessed during backprop.

    stacked_params: np.ndarray
        Neural network parameters, represented as single
        array of stacked parameters for all layers.
        e.g. np.array([
                    [W1],
                    [b1],
                    [W2],
                    [b2],
                    [W3],
                    [b3],
                    ...
                ])
    """
    parameters.unstack(stacked_params)
    Y_pred, J_pred = model_partials_forward(X, parameters, cache)
    return cost.evaluate(Y_pred, J_pred)


def objective_gradient(
    data: Dataset,
    parameters: Parameters,
    cache: Cache,
    lambd: float,
    gamma: float,
    stacked_params: np.ndarray,
) -> np.ndarray:  # noqa: PLR0913
    """Evaluate cost function gradient for backprop.

    Parameters
    ----------
    data: Dataset
        Object containing training and associated metadata.

    parameters: Parameters
        Neural net parameters. Object that stores
        neural net parameters for each layer.

    cache: Cache
        Neural net cache. Object that stores
        neural net quantities for each layer,
        during forward prop, so they can be
        accessed during backprop.

    lambd: int, optional
        Coefficient that multiplies regularization term in cost function.
        Default is 0.0

    gamma: int, optional
        Coefficient that multiplies gradient-enhancement term in cost function.
        Default is 0.0

    stacked_params: np.ndarray
        Neural network parameters, represented as single
        array of stacked parameters for all layers.
        e.g. np.array([
                    [W1],
                    [b1],
                    [W2],
                    [b2],
                    [W3],
                    [b3],
                    ...
                ])
    """
    parameters.unstack(stacked_params)
    model_backward(data, parameters, cache, lambd, gamma)
    return parameters.stack_partials()


def train_model(
    data: Dataset,
    parameters: Parameters,
    alpha: float = 0.050,
    lambd: float = 0.000,
    gamma: float = 0.000,
    beta1: float = 0.900,
    beta2: float = 0.999,
    epochs: int = 1,
    max_iter: int = 200,
    batch_size: int = None,
    shuffle: bool = True,
    random_state: int = None,
    is_backtracking: bool = False,
    is_verbose: bool = False,
) -> dict:  # noqa: PLR0913
    """Train neural net.

    Note:
        Model parameters are updated in place.

    Parameters
    ----------
    data: Dataset
        Object containing training and associated metadata.

    parameters: Parameters
        Neural net parameters. Object that stores
        neural net parameters for each layer.

    alpha: float, optional
        Learning rate (controls optimizer step size for line search)
        Default is 0.05

    lambd: float, optional
        Regularization coefficient (controls how much to penalize
        objective function for over-fitting)
        Default is 0.0

    gamma: float, optional
        Gradient-enhancement coefficient (controls important of 1st-order
        accuracy errors in objective function.)
        Default is 1.0 (full gradient-enhancement)
        Note: only active when dydx is provided (ignored otherwise)

    beta1: float, optional
        Hyperparameter that controls momentum for
        [ADAM](https://arxiv.org/abs/1412.6980) optimizer.
        Default is 0.9

    beta2: float, optional
        Hyperparameter that controls momentum for
        [ADAM](https://arxiv.org/abs/1412.6980) optimizer.
        Default is 0.999

    epochs: int, optional
        Number of epochs (passes through data). Default is 1.
        Note: total number of objective function calls =
                number of epochs
                    x number of batches
                        x number of iterations (search directions)
                            x number of evaluations along each line search

    batch_size: int, optional
        Size of each batch for minibatch, which is a routine that randomly
        splits the data into discrete batches to train faster (in cases
        where data is very large).
        Note: total number of objective function calls =
                number of epochs
                    x number of batches
                        x number of iterations (search directions)
                            x number of evaluations along each line search

    max_iter: int, optional
        Maximum number of optimizer iterations. Default 200.
        Note: total number of objective function calls =
                number of epochs
                    x number of batches
                        x number of iterations (search directions)
                            x number of evaluations along each line search

    shuffle: bool, optional
        Shuffle data for minibatch. Default is True.

    random_state: int, optional
        Random seed for minibatch repeatability. Default is None.

    is_backtracking: bool, optional
        Use backtracking line search, where step size is progressively
        reduced (multiple steps) until cost function no longer improves
        along search direction. Default is False (single step).
        Note: total number of objective function calls =
                number of epochs
                    x number of batches
                        x number of iterations (search directions)
                            x number of evaluations along each line search

    is_verbose: bool, optional
        Print out progress for each iteration, each batch, each epoch.
        Default is False.

    Returns
    -------
    history: dict[int, dict[int, list[np.float64]]]
        Cost function history for each epoch, batch, iteration. The cost history
        is useful for plotting convergence. For example, to access the cost
        history of epoch 2, batch 5, iteration 9:

            cost = history['epoch_2']['batch_5'][9]
    """
    history = defaultdict(dict)

    update = ADAM(beta1, beta2)
    line_search = Backtracking(update, max_count=is_backtracking * 1_000)
    optimizer = Optimizer(line_search)

    stacked_params = parameters.stack()

    for e in range(epochs):
        batches = data.mini_batches(batch_size, shuffle, random_state)
        for b, batch in enumerate(batches):
            cache = Cache(parameters.layer_sizes, batch.m)
            cost = Cost(batch, parameters, lambd, gamma)
            func = functools.partial(
                objective_function,
                batch.X,
                cost,
                parameters,
                cache,
            )
            grad = functools.partial(
                objective_gradient,
                batch,
                parameters,
                cache,
                lambd,
                gamma,
            )
            optimizer.minimize(
                x=stacked_params,
                f=func,
                dfdx=grad,
                alpha=alpha,
                max_iter=max_iter,
                verbose=is_verbose,
                epoch=e,
                batch=b,
            )
            history[f"epoch_{e}"][f"batch_{b}"] = optimizer.cost_history
    return history
