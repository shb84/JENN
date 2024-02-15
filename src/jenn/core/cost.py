"""Cost Function.
================="""

import numpy as np

from .data import Dataset
from .parameters import Parameters


class SquaredLoss:
    """Least Squares Estimator.

    Parameters
    ----------
    Y_true: np.ndarray
        Training data outputs. An array of shape (n_y, m)
    """

    def __init__(self, Y_true: np.ndarray):  # noqa: D107
        self.Y_true = Y_true
        self.Y_error = np.zeros(Y_true.shape)  # preallocate to save resources

    def evaluate(self, Y_pred: np.ndarray) -> np.float64:
        """Compute least squares estimator of the states in place.

        Parameters
        ----------
        Y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).
        """
        self.Y_error = Y_pred - self.Y_true
        n_y = self.Y_error.shape[0]
        cost = 0
        for j in range(0, n_y):
            cost += np.dot(self.Y_error[j], self.Y_error[j].T)
        return np.float64(cost)


class GradientEnhancement:
    """Least Squares Estimator for partials.

    Parameters
    ----------
    dY_true: np.ndarray
        Training data gradients. An array of shape (n_y, n_x, m)
        Y' = d(Y)/dX where n_y = number outputs
                           n_x = number inputs
                             m = number examples
    """

    def __init__(self, dY_true: np.ndarray):  # noqa: D107
        self.dY_true = dY_true
        self.dY_error = np.zeros(dY_true.shape)

    def evaluate(self, dY_pred: np.ndarray) -> np.float64:
        """Compute least squares estimator for the partials.

        Parameters
        ----------
        dy_pred: np ndarray of shape (n_y, n_x, m)
            Predicted partials: AL' = d(AL)/dX where n_y = number outputs
                                                     n_x = number inputs
                                                     m = number examples

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).
        """
        dY_true = self.dY_true
        dY_error = self.dY_error
        n_y, n_x, m = dY_true.shape
        cost = 0.0
        for k in range(0, n_y):
            for j in range(0, n_x):
                dY_error[k, j] = dY_pred[k, j] - dY_true[k, j]
                dot_product = np.dot(dY_error[k, j], dY_error[k, j].T)
                cost += np.squeeze(dot_product)
        return np.float64(cost)


class Regularization:
    """Compute regularization penalty.

    Parameters
    ----------
    weights: List[np.ndarray]
        Parameters W associated with each layer of neural network
        i.e. a = g(z) where z = w * a_prev + b
    """

    def __init__(self, weights: np.ndarray):  # noqa: D107
        # Preallocate for speed
        self.weights = weights
        self._squared_weights = [np.zeros(W.shape) for W in weights]

    def evaluate(self, lambd: float) -> np.float64:
        """Compute L2 norm penalty.

        Parameters
        ----------
        lambd: float
            Regularization coefficient
        """
        penalty = 0.0
        if lambd > 0:
            for i, weight in enumerate(self.weights):
                squared_weights = np.square(weight, out=self._squared_weights[i])
                penalty += np.squeeze(np.sum(squared_weights))
        return lambd * np.float64(penalty)


class Cost:
    """Neural Network cost function.

    Parameters
    ----------
    data: Dataset
        Object containing training and associated metadata.

    parameters: Parameters
        Neural net parameters. Object that stores
        neural net parameters for each layer.

    lambd: int, optional
        Coefficient that multiplies regularization term in cost function.
        Default is 0.0

    gamma: int, optional
        Coefficient that multiplies gradient-enhancement term in cost function.
        Default is 0.0
    """

    def __init__(
        self,
        data: Dataset,
        parameters: Parameters,
        lambd: float = 0.0,
        gamma: float = 0.0,
    ):  # noqa: D107
        self.data = data
        self.parameters = parameters
        self.lambd = lambd
        self.gamma = gamma
        self.squared_loss = SquaredLoss(data.Y)
        self.regularization = Regularization(parameters.W)
        if data.J is not None and gamma > 0.0:  # noqa: PLR2004
            self.gradient_enhancement = GradientEnhancement(data.J)

    def evaluate(self, Y_pred: np.ndarray, J_pred: np.ndarray = None) -> np.float64:
        """Evaluate cost function.

        Parameters
        ----------
        Y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        J_pred: np ndarray of shape (n_y, n_x, m), optional
            Predicted partials: AL' = d(AL)/dX where n_y = number outputs
                                                     n_x = number inputs
                                                     m = number examples
            Default is None.
        """
        cost = self.squared_loss.evaluate(Y_pred)
        if J_pred is not None and hasattr(self, "gradient_enhancement"):
            cost += self.gradient_enhancement.evaluate(J_pred) * self.gamma
        cost += self.regularization.evaluate(self.lambd)
        cost *= 0.5 / self.data.m
        return cost
