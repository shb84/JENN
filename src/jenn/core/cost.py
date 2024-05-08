"""Cost Function.
=================

This module contains class and methods to efficiently 
compute the neural net cost function used for training. 
It is a modified version of the Least Squared Estimator (LSE), 
augmented with a penalty function for regularization and another 
term which accounts for Jacobian prediction error. See
`paper`_ for details and notation. 
"""  # noqa W291

from typing import Union

import numpy as np

from .data import Dataset
from .parameters import Parameters


class SquaredLoss:
    r"""Least Squares Estimator.

    :param Y_true: training data outputs :math:`Y \in \mathbb{R}^{n_y
        \times m}`
    :param Y_weights: weights by which to prioritize data points (optional)
    """

    def __init__(
        self, Y_true: np.ndarray, Y_weights: Union[np.ndarray, float] = 1.0
    ) -> None:
        self.Y_true = Y_true
        self.Y_error = np.zeros(Y_true.shape)  # preallocate to save resources
        self.Y_weights = np.ones(Y_true.shape) * Y_weights
        self.n_y, self.m = Y_true.shape

    def evaluate(self, Y_pred: np.ndarray) -> np.float64:
        r"""Compute least squares estimator of the states in place.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_y \times m}`
        """
        self.Y_error[:, :] = Y_pred - self.Y_true
        self.Y_error *= np.sqrt(self.Y_weights)
        cost = 0
        for j in range(0, self.n_y):
            cost += np.dot(self.Y_error[j], self.Y_error[j].T)
        return np.float64(cost)


class GradientEnhancement:
    r"""Least Squares Estimator for partials.

    :param J_true: training data jacobian :math:`Y^{\prime} \in
        \mathbb{R}^{n_y \times m}`
        :param J_weights: weights by which to prioritize partials (optional)
    """

    def __init__(
        self, J_true: np.ndarray, J_weights: Union[np.ndarray, float] = 1.0
    ) -> None:
        self.J_true = J_true
        self.J_error = np.zeros(J_true.shape)
        self.J_weights = np.ones(J_true.shape) * J_weights
        self.n_y, self.n_x, self.m = J_true.shape

    def evaluate(self, J_pred: np.ndarray) -> np.float64:
        r"""Compute least squares estimator for the partials.

        :param J_pred: predicted Jacobian :math:`A^{\prime[L]} \in
            \mathbb{R}^{n_y \times n_x \times m}`
        """
        self.J_error[:, :, :] = self.J_weights * (J_pred - self.J_true)
        self.J_error *= np.sqrt(self.J_weights)
        cost = 0.0
        for k in range(0, self.n_y):
            for j in range(0, self.n_x):
                dot_product = np.dot(self.J_error[k, j], self.J_error[k, j].T)
                cost += np.squeeze(dot_product)
        return np.float64(cost)


class Regularization:
    """Compute regularization penalty."""

    def __init__(self, weights: list[np.ndarray], lambd: float = 0.0) -> None:
        r"""Compute L2 norm penalty.

        :param weights: neural parameters :math:`W^{[l]} \in
        \mathbb{R}^{n^{[l]} \times n^{[l-1]}}` associated with each
        layer
        """
        self.weights = weights
        self.lambd = lambd

    def evaluate(
        self,
    ) -> float:
        r"""Compute L2 norm penalty.

        :param weights: neural parameters :math:`W^{[l]} \in
        \mathbb{R}^{n^{[l]} \times n^{[l-1]}}` associated with each
        layer
        :param lambd: regularization coefficient :math:`\lambda \in
            \mathbb{R}` (hyperparameter to be tuned)
        """
        penalty = 0.0
        if self.lambd > 0.0:
            for W in self.weights:
                penalty += np.sum(np.square(W)).squeeze()
            return self.lambd * penalty
        return 0.0


class Cost:
    r"""Neural Network cost function.

    :param data: Dataset object containing training data (and associated
        metadata)
    :param parameters: object containing neural net parameters (and
        associated metadata) for each layer
    :param lambd: regularization coefficient to avoid overfitting
    """

    def __init__(
        self,
        data: Dataset,
        parameters: Parameters,
        lambd: float = 0.0,
    ) -> None:
        self.data = data
        self.parameters = parameters
        self.squared_loss = SquaredLoss(data.Y, data.Y_weights)
        self.regularization = Regularization(parameters.W, lambd)
        if data.J is not None:  # noqa: PLR2004
            self.gradient_enhancement = GradientEnhancement(data.J, data.J_weights)

    def evaluate(
        self, Y_pred: np.ndarray, J_pred: Union[np.ndarray, None] = None
    ) -> np.float64:
        r"""Evaluate cost function.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_x \times m}`
        :param J_pred: predicted Jacobian :math:`A^{\prime[L]} \in
            \mathbb{R}^{n_y \times n_x \times m}`
        """
        cost = self.squared_loss.evaluate(Y_pred)
        if J_pred is not None and hasattr(self, "gradient_enhancement"):
            cost += self.gradient_enhancement.evaluate(J_pred)
        cost += self.regularization.evaluate()
        cost *= 0.5 / self.data.m
        return cost
