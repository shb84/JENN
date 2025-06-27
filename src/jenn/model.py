"""Model.
=========

This module contains the main class to train a neural net and make
predictions. It acts as an interface between the user and the core 
functions doing computations under-the-hood. 

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn 

    # Fit model
    nn = jenn.model.NeuralNet(
        layer_sizes=[
            x_train.shape[0],  # input layer 
            7, 7,              # hidden layer(s) -- user defined
            y_train.shape[0]   # output layer 
         ],  
        ).fit(
            x_train, y_train, dydx_train, **kwargs # note: user must provide this
        )

    # Predict response only 
    y_pred = nn.predict(x_test)

    # Predict partials only 
    dydx_pred = nn.predict_partials(x_train)

    # Predict response and partials in one step (preferred)
    y_pred, dydx_pred = nn.evaluate(x_test) 

.. Note::
    The method `evaluate()` is preferred over separately 
    calling `predict()` followed by `predict_partials()` 
    whenever both the response and its partials are needed at the same point. 
    This saves computations since, in the latter approach, forward propagation 
    is unecessarily performed twice. Similarly, to avoid unecessary partial
    deerivative calculations, the `predict()` method should be preferred whenever
    only response values are needed. The method `predict_partials()` is provided 
    for those situations where it is necessary to separate out Jacobian predictions, 
    due to how some target optimization software architected for example.  
"""  # noqa: W291

from pathlib import Path
from typing import Any, List, Tuple, Union, Optional

import numpy as np

from .core.cache import Cache
from .core.data import Dataset, denormalize, denormalize_partials, normalize, avg, std
from .core.parameters import Parameters
from .core.propagation import model_forward, model_partials_forward, partials_forward
from .core.training import train_model

__all__ = ["NeuralNet"]


class NeuralNet:
    """Neural network model.

    :param layer_sizes: number of nodes in each layer (including
        input/output layers)
    :param hidden_activation: activation function used in hidden layers
    :param output_activation: activation function used in output layer
    """

    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "tanh",
        output_activation: str = "linear",
    ):  # noqa D107
        self.history: Union[dict[Any, Any], None] = None
        self.parameters = Parameters(
            layer_sizes,
            hidden_activation,
            output_activation,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dydx: Union[np.ndarray, None] = None,
        is_normalize: bool = False,
        alpha: float = 0.05,
        beta: Union[np.ndarray, float] = 1.0,
        gamma: Union[np.ndarray, float] = 1.0,
        lambd: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 0.5,
        tol: float = 1e-12,
        max_count: int = 10,
        epsilon_absolute: float = 1e-12,
        epsilon_relative: float = 1e-12,
        epochs: int = 1,
        batch_size: Union[int, None] = None,
        max_iter: int = 1000,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        is_backtracking: bool = False,
        is_warmstart: bool = False,
        is_verbose: bool = False,
        custom_loss: Optional[type] = None,
        N1_max: int = 100,
        N2_max: int = 100,
    ) -> "NeuralNet":  # noqa: PLR0913
        r"""Train neural network.

        :param x: training data inputs, array of shape (n_x, m)
        :param y: training data outputs, array of shape (n_y, m)
        :param dydx: training data Jacobian, array of shape (n_y, n_x, m)
        :param is_normalize: normalize training by mean and variance
        :param alpha: optimizer learning rate for line search
        :param beta: LSE coefficients [defaulted to one] (optional)
        :param gamma: jacobian-enhancement regularization coefficient [defaulted to zero] (optional)
        :param lambd: regularization coefficient to avoid overfitting [defaulted to zero] (optional)
        :param beta1: `ADAM <https://arxiv.org/abs/1412.6980>`_ optimizer hyperparameter to control momentum
        :param beta2: ADAM optimizer hyperparameter to control momentum
        :param tau: amount by which to reduce :math:`\alpha := \tau \times
            \alpha` on each iteration
        :param tol: stop when cost function doesn't improve more than
            specified tolerance
        :param max_count: stop when line search iterations exceed maximum
            count specified
        :param epsilon_absolute: absolute error stopping criterion
        :param epsilon_relative: relative error stopping criterion
        :param epochs: number of passes through data
        :param batch_size: size of each batch for minibatch
        :param max_iter: max number of optimizer iterations
        :param shuffle: shuffle minibatches or not
        :param random_state: control repeatability
        :param is_backtracking: use backtracking line search or not
        :param is_warmstart: do not initialize parameters
        :param is_verbose: print out progress for each (iteration, batch, epoch)
        :param N1_max: number of iterations for which absolute criterion must hold true before stop
        :param N2_max: number of iterations for which relative criterion must hold true before stop
        :return: NeuralNet instance (self)

        .. warning::
                Normalization usually helps, except when the training
                data is made up of very small numbers. In that case,
                normalizing by the variance has the undesirable effect
                of dividing by a very small number and should not be used.
        """
        params = self.parameters
        if is_warmstart:
            data = Dataset(
                    x, y, dydx, 
                    x_ref=params.mu_x, 
                    y_ref=params.mu_y, 
                    x_scale=params.sigma_x, 
                    y_scale=params.sigma_y
                ).normalize()
        else:
            params.initialize(random_state)
            params.mu_x[:] = 0.0
            params.mu_y[:] = 0.0
            params.sigma_x[:] = 1.0
            params.sigma_y[:] = 1.0
            data = Dataset(x, y, dydx)
            if is_normalize:
                params.mu_x[:] = x_ref = avg(data.X)
                params.mu_y[:] = y_ref = avg(data.Y)
                x_scale = std(data.X)
                y_scale = std(data.Y)
                params.sigma_x[:] = x_scale = np.ones(x_scale.shape) if np.allclose(0, x_scale, atol=1e-6) else x_scale  # do NOT scale if it's results in division by small number
                params.sigma_y[:] = y_scale = np.ones(y_scale.shape) if np.allclose(0, y_scale, atol=1e-6) else y_scale
                data = Dataset(
                    x, y, dydx, 
                    x_ref=x_ref, 
                    y_ref=y_ref, 
                    x_scale=x_scale, 
                    y_scale=y_scale
                ).normalize()
        self.history = train_model(
            data,
            params,
            # hyperparameters
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambd=lambd,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            tol=tol,
            max_count=max_count,
            epsilon_absolute=epsilon_absolute,
            epsilon_relative=epsilon_relative,
            # options
            epochs=epochs,
            max_iter=max_iter,
            batch_size=batch_size,
            shuffle=shuffle,
            random_state=random_state,
            is_backtracking=is_backtracking,
            is_verbose=is_verbose,
            custom_loss=custom_loss,
            N1_max=N1_max,
            N2_max=N2_max,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""Predict responses.

        :param x: vectorized inputs, array of shape (n_x, m)
        :return: predicted response(s), array of shape (n_y, m)
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm = model_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        return y

    def predict_partials(self, x: np.ndarray) -> np.ndarray:
        r"""Predict partials.

        :param x: vectorized inputs, array of shape (n_x, m)
        :return: predicted partial(s), array of shape (n_y, n_x, m)
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        dydx_norm = partials_forward(x_norm, params, cache)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return dydx

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Predict responses and their partials.

        :param x: vectorized inputs, array of shape (n_x, m)
        :return: predicted response(s), array of shape (n_y, m)
        :return: predicted partial(s), array of shape (n_y, n_x, m)
        """
        params = self.parameters
        cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm, dydx_norm = model_partials_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return y, dydx

    def save(self, file: Union[str, Path] = "parameters.json") -> None:
        """Serialize parameters and save to JSON file."""
        self.parameters.save(file)

    def load(self, file: Union[str, Path] = "parameters.json") -> "NeuralNet":
        """Load previously saved parameters from json file."""
        self.parameters.load(file)
        return self
