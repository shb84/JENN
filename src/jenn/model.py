from __future__ import annotations
from .parameters import Parameters
from .training import train_model
from .data import Dataset
from .normalization import normalize_partials, normalize, denormalize
from .propagation import model_forward


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

    def fit(self, data: Dataset, is_normalize=False, **kwargs):
        if is_normalize:
            data = self._normalize(data)
        else:
            self.parameters.mu_x = 0.0
            self.parameters.mu_y = 0.0
            self.parameters.sigma_x = 1.0
            self.parameters.sigma_y = 1.0
        train_model(data, self.parameters, **kwargs)

    def predict(self, X):
        X = normalize(X, self.parameters.mu_x, self.parameters.sigma_x)
        Y = model_forward(X, self.parameters)
        return denormalize(Y, self.parameters.mu_y, self.parameters.sigma_y)

    def predict_partials(self, X):
        pass
