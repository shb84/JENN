import numpy as np
from typing import List
from ._fwd_prop import L_model_forward, L_grads_forward
from ._parameters import Parameters
from ._cache import Cache
from ._loss import SquaredLoss, Regularization, GradientEnhancement
from ._data import Dataset


class Cost:
    """ Neural Network cost function """

    def __init__(self, training_data: Dataset, parameters: Parameters):
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
        self.training_data = training_data
        self.parameters = parameters
        self.squared_loss = SquaredLoss(training_data.Y)
        self.regularization = Regularization(parameters.W)
        self.gradient_enhancement = GradientEnhancement(training_data.J)

    def evaluate(self, Y_pred, J_pred=None, alpha=0, gamma=0, indices=None):
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
        c = self.squared_loss(Y_pred, indices)
        if gamma > 0:
            if J_pred is not None:
                c += self.gradient_enhancement(J_pred, indices) * gamma
        if alpha > 0:
            c += self.regularization(alpha)
        c *= 0.5 / self.training_data.m
        return c
