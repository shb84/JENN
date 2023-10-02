"""Cost Function."""
import numpy as np

from .data import Dataset
from .parameters import Parameters


class SquaredLoss:
    """ Least Squares Estimator """

    def __init__(self, y_true):
        """
        Parameters
        ----------
        y_true: np.ndarray
            Training data outputs. An array of shape (n_y, m)
        """
        self.y_true = y_true
        self.y_error = np.zeros(y_true.shape)  # preallocate to save resources

    def __call__(self, y_pred):
        """
        Compute least squares estimator of the states in place

        Parameters
        ----------
        y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).
        """
        self.y_error = y_pred - self.y_true
        n_y = self.y_error.shape[0]
        cost = 0
        for j in range(0, n_y):
            cost += np.dot(self.y_error[j], self.y_error[j].T)
        return np.float64(cost)


class GradientEnhancement:
    """ Least Squares Estimator for partials """

    def __init__(self, dy_true):
        """
        Parameters
        ----------
        dy_true: np.ndarray
            Training data gradients. An array of shape (n_y, n_x, m)
            Y' = d(Y)/dX where n_y = number outputs
                               n_x = number inputs
                               m = number examples
        """
        self.dy_true = dy_true
        self.dy_error = np.zeros(dy_true.shape)

    def __call__(self, dy_pred):
        """
        Compute least squares estimator for the partials

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
        dy_true = self.dy_true
        dy_error = self.dy_error
        n_y, n_x, m = dy_true.shape
        cost = 0.0
        for k in range(0, n_y):
            for j in range(0, n_x):
                dy_error[k, j] = dy_pred[k, j] - dy_true[k, j]
                dot_product = np.dot(dy_error[k, j], dy_error[k, j].T)
                cost += np.squeeze(dot_product)
        return np.float64(cost)


class Regularization:
    """ Compute regularization penalty """

    def __init__(self, weights):
        """
        Parameters
        ----------
        weights: List[np.ndarray]
            Parameters w associated with each layer of neural network
            i.e. a = g(z) where z = w * a_prev + b
        """
        # Preallocate for speed
        self.weights = weights
        self._squared_weights = [np.zeros(w.shape) for w in weights]

    def __call__(self, lambd: float):
        """Compute L2 norm penalty.

        Parameters
        ----------
        lambd: float
            Regularization coefficient
        """
        penalty = 0.0
        if lambd > 0:
            for i, weight in enumerate(self.weights):
                squared_weights = np.square(
                    weight, out=self._squared_weights[i])
                penalty += np.squeeze(np.sum(squared_weights))
        return lambd * np.float64(penalty)


class Cost:
    """ Neural Network cost function """

    def __init__(
            self,
            data: Dataset,
            parameters: Parameters,
            lambd: float = 0.0,
            gamma: float = 0.0,
    ):
        """
        Parameters
        ----------
        Y_true: np.ndarray
            Training data outputs. An array of shape (n_y, m)

        J_true: np.ndarray
            Training data gradients. An array of shape (n_y, n_x, m)
            Y' = d(Y)/dX where n_y = number outputs
                               n_x = number inputs
                               m = number examples

        weights: List[np.ndarray]
            Initial parameters w associated with each layer of neural network
            i.e. a = g(z) where z = w * a_prev + b
        """
        self.data = data
        self.parameters = parameters
        self.lambd = lambd
        self.gamma = gamma
        self.squared_loss = SquaredLoss(data.Y)
        self.regularization = Regularization(parameters.W)
        if data.J is not None and gamma > 0.0:
            self.gradient_enhancement = GradientEnhancement(data.J)

    def evaluate(self, Y_pred, J_pred=None):
        """
        Parameters
        ----------
        Y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        J_pred: np ndarray of shape (n_y, n_x, m)
            Predicted partials: AL' = d(AL)/dX where n_y = number outputs
                                                     n_x = number inputs
                                                     m = number examples

        weights: List[np.ndarray]
            Parameters w associated with each layer of neural network
            i.e. a = g(z) where z = w * a_prev + b

        alpha: float, optional
            Regularization coefficient. Default is 0

        gamma: float, optional
            Gradient coefficient. Default is 0

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).

        Returns
        -------
        c: np.float64
            Cost function value
        """
        c = self.squared_loss(Y_pred)
        if J_pred is not None and hasattr(self, 'gradient_enhancement'):
            c += self.gradient_enhancement(J_pred) * self.gamma
        c += self.regularization(self.lambd)
        c *= 0.5 / self.data.m
        return c
