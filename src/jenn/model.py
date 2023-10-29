"""Neural network model.

This module contains the main class to train a neural net and make predictions.
It is in charge of setting up and calling the right support functions to
accomplish these various tasks.
"""

import numpy as np
from time import time
from functools import wraps

from .core.parameters import Parameters
from .core.training import train_model
from .core.cache import Cache
from .core.data import Dataset, normalize, denormalize, denormalize_partials
from .core.propagation import partials_forward, model_forward, model_partials_forward


__all__ = ["NeuralNet"]


def timeit(func):
    """Return elapsed time to run a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time()
        results = func(*args, **kwargs)
        toc = time()
        print(f'elapsed time: {toc-tic:.3f} s')
        return results
    return wrapper


class NeuralNet:
    """Neural network model.

    Parameters
    ----------
    layer_sizes: list[int]
        The number of nodes in each layer of the neural
        network, including input and output layers.

    hidden_activation: str, optional
        The activation function to use for hidden layers.
        Default is "tanh". Options are: "relu", "linear", "tanh"

    output_activation: str, optional
        The activation function to use for the output layer.
        Default is "linear". Options are: "relu", "linear", "tanh"
    """

    def __init__(
            self,
            layer_sizes: list[int],
            hidden_activation: str = 'tanh',
            output_activation: str = 'linear',
    ):
        self.parameters = Parameters(
            layer_sizes,
            hidden_activation,
            output_activation,
        )

    @timeit
    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dydx: np.ndarray = None,
            is_normalize: bool = False,
            alpha: float = 0.050,
            lambd: float = 0.000,
            gamma: float = 1.000,
            beta1: float = 0.900,
            beta2: float = 0.999,
            epochs: int = 1,
            batch_size: int = None,
            max_iter: int = 200,
            shuffle: bool = True,
            random_state: int = None,
            is_backtracking=False,
            is_verbose=False,
    ) -> None:
        """
        Train neural network.

        Parameters
        ----------
        x: numpy.ndarray
            Training data inputs. Array of shape (n_x, m)
            where n_x = number of inputs
                    m = number of examples

        y: numpy.ndarray
            Training data outputs. Array of shape (n_y, m)
            where n_y = number of outputs
                    m = number of examples

        dydx: numpy.ndarray
            Training data gradients. Array of shape (n_y,n_x, m)
            where n_x = number of inputs
                  n_y = number of outputs
                    m = number of examples

        is_normalize: bool, optional
            Normalize data by mean and variance. Default is False.

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
        """
        hyperparams = dict(
            alpha=alpha,
            lambd=lambd,
            gamma=gamma,
            beta1=beta1,
            beta2=beta2,
            epochs=epochs,
            max_iter=max_iter,
            batch_size=batch_size,
            shuffle=shuffle,
            random_state=random_state,
            is_backtracking=is_backtracking,
            is_verbose=is_verbose,
        )
        data = Dataset(x, y, dydx)
        params = self.parameters
        params.mu_x[:] = 0.0
        params.mu_y[:] = 0.0
        params.sigma_x[:] = 1.0
        params.sigma_y[:] = 1.0
        if is_normalize:
            params.mu_x[:] = data.avg_x
            params.mu_y[:] = data.avg_y
            params.sigma_x[:] = data.std_x
            params.sigma_y[:] = data.std_y
            data = data.normalize()
        train_model(data, params, **hyperparams)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y = f(x)

        Note:
            Consider using 'evaluate(x)' instead of 'predict(x) if both
            partials and function evaluations are needed at x (its more
            efficient than running predict(x) followed by predict_partials(x)
            which would end up running model_forward(x) twice under the hood).

        Parameters
        ----------
        x: numpy.ndarray
            Input array of shape (n_x, m)
            where n_x = number of inputs
                    m = number of examples

        Returns
        -------
        y: numpy.ndarray
            Outputs array of shape (n_y, m)
            where n_y = number of outputs
                    m = number of examples
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm = model_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        return y

    def predict_partials(self, x: np.ndarray) -> np.ndarray:
        """Predict y = f'(x)

        Note:
            This function needs to call model_forward before
            calling partials_forward because partials require
            information computed during model_forward. Consider
            using 'evaluate(x)' instead if both partials and function
            evaluations are needed at the same x (its more
            efficient than running predict(x) followed by predict_partials(x)
            which would end up running model_forward(x) twice under the hood)

        Parameters
        ----------
        x: numpy.ndarray
            Input array of shape (n_x, m)
            where n_x = number of inputs
                    m = number of examples

        Returns
        -------
        dydx: numpy.ndarray
            Output gradient as array of shape (n_y, n_x, m)
            where n_x = number of inputs
                  n_y = number of outputs
                    m = number of examples
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        dydx_norm = partials_forward(x_norm, params, cache)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return dydx

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict y = f(x) and dy/dx = f'(x)

        Note:
            Use this function if both y, and dy/dx are
            needed are same x. It is more efficient than
            separately calling predict(x) followed by predict_partials(x)
            which would end up running model_forward(x) twice under the hood.

        Parameters
        ----------
        x: numpy.ndarray
            Input array of shape (n_x, m)
            where n_x = number of inputs
                    m = number of examples

        Returns
        -------
        y, dydx: tuple[numpy.ndarray, numpy.ndarray]
            y = output array of shape (n_y, m)
            dydx = output gradient as array of shape (n_y, n_x, m)
            where n_x = number of inputs
                  n_y = number of outputs
                    m = number of examples
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm, dydx_norm = model_partials_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return y, dydx
