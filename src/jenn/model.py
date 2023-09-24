from __future__ import annotations

import numpy as np

from .parameters import Parameters
from .training import train_model
from .data import Dataset
from .normalization import (
    normalize,
    denormalize,
    normalize_partials,
    denormalize_partials,
)
from .propagation import model_forward, model_partials


class NeuralNet:

    def __init__(
            self,
            layer_sizes,
            hidden_activation='tanh',
            output_activation='linear',
    ):
        self.parameters = Parameters(
            layer_sizes,
            hidden_activation,
            output_activation,
        )

    def _normalize(self, data: Dataset):
        X = normalize(data.X, data.avg_x, data.std_x)
        Y = normalize(data.Y, data.avg_y, data.std_y)
        J = normalize_partials(data.J, data.avg_x, data.std_y)
        self.parameters.mu_x = data.std_x
        self.parameters.mu_y = data.std_y
        self.parameters.sigma_x = data.std_x
        self.parameters.sigma_y = data.std_y
        return Dataset(X, Y, J)

    def fit(self, X, Y, J=None, is_normalize=False, **kwargs):
        data = Dataset(X, Y, J)
        if is_normalize:
            self.parameters.mu_x[:] = data.avg_x
            self.parameters.mu_y[:] = data.avg_y
            self.parameters.sigma_x[:] = data.std_x
            self.parameters.sigma_y[:] = data.std_y
            X_norm = normalize(data.X, data.avg_x, data.std_x)
            Y_norm = normalize(data.Y, data.avg_y, data.std_y)
            J_norm = normalize_partials(data.J, data.std_x, data.std_y)
            data = Dataset(X_norm, Y_norm, J_norm)
        else:
            self.parameters.mu_x[:] = 0.0
            self.parameters.mu_y[:] = 0.0
            self.parameters.sigma_x[:] = 1.0
            self.parameters.sigma_y[:] = 1.0
        train_model(data, self.parameters, **kwargs)

    def predict(self, X):
        X = normalize(X, self.parameters.mu_x, self.parameters.sigma_x)
        Y = model_forward(X, self.parameters)
        return denormalize(Y, self.parameters.mu_y, self.parameters.sigma_y)

    def predict_partials(self, X):
        X = normalize(X, self.parameters.mu_x, self.parameters.sigma_x)
        J = model_partials(X, self.parameters)
        return denormalize_partials(
            J, self.parameters.sigma_x, self.parameters.sigma_y)
