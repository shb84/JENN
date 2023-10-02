from __future__ import annotations

import numpy as np

from typing import List, Tuple

from .parameters import Parameters
from .training import train_model
from .cache import Cache
from .data import Dataset
from .normalization import normalize, denormalize, denormalize_partials
from .propagation import partials_forward, model_forward, model_partials_forward


class NeuralNet:

    def __init__(
            self,
            layer_sizes: List[int],
            hidden_activation: str = 'tanh',
            output_activation: str = 'linear',
    ):
        self.parameters = Parameters(
            layer_sizes,
            hidden_activation,
            output_activation,
        )

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            J: np.ndarray = None,
            is_normalize: bool = False,
            **kwargs,
    ):
        data = Dataset(X, Y, J)
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
        train_model(data, params, **kwargs)

    def predict(self, x: np.ndarray, cache: Cache = None) -> np.ndarray:
        """Return y = f(x)"""
        params = self.parameters
        if cache is None:
            cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm = model_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        return y

    def predict_partials(
            self, x: np.ndarray, cache: Cache = None) -> np.ndarray:
        """Return dy/dx = f'(x)"""
        params = self.parameters
        if cache is None:
            cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        dydx_norm = partials_forward(x_norm, params, cache)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return dydx

    def evaluate(
            self, x: np.ndarray, cache: Cache = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return y = f(x) and dy/dx = f'(x)"""
        params = self.parameters
        if cache is None:
            cache = Cache(params.layer_sizes, m=x.shape[1])
        x_norm = normalize(x, params.mu_x, params.sigma_x)
        y_norm, dydx_norm = model_partials_forward(x_norm, params, cache)
        y = denormalize(y_norm, params.mu_y, params.sigma_y)
        dydx = denormalize_partials(dydx_norm, params.sigma_x, params.sigma_y)
        return y, dydx
