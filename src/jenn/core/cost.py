"""Cost Function.
=================

This module contains class and methods to efficiently 
compute the neural net cost function used for training. 
It is a modified version of the Least Squared Estimator (LSE), 
augmented with a penalty function for regularization and another 
term which accounts for Jacobian prediction error. See
`paper`_ for details and notation. 
"""  # noqa W291

import numpy as np

from .data import Dataset
from .parameters import Parameters


class SquaredLoss:
    r"""Least Squares Estimator.

    :param Y_true: training data outputs :math:`Y \in \mathbb{R}^{n_y
        \times m}`
    """

    def __init__(self, Y_true: np.ndarray):  # noqa: D107
        self.Y_true = Y_true
        self.Y_error = np.zeros(Y_true.shape)  # preallocate to save resources

    def evaluate(self, Y_pred: np.ndarray) -> np.float64:
        r"""Compute least squares estimator of the states in place.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_y \times m}`
        :param indices: only evaluate specified training data indices
            (used for minibatch)
        """
        self.Y_error = Y_pred - self.Y_true
        n_y = self.Y_error.shape[0]
        cost = 0
        for j in range(0, n_y):
            cost += np.dot(self.Y_error[j], self.Y_error[j].T)
        return np.float64(cost)


class GradientEnhancement:
    r"""Least Squares Estimator for partials.

    :param J_true: training data jacobian :math:`Y^{\prime} \in
        \mathbb{R}^{n_y \times m}`
    """

    def __init__(self, J_true: np.ndarray):  # noqa: D107
        self.J_true = J_true
        self.J_error = np.zeros(J_true.shape)

    def evaluate(self, J_pred: np.ndarray) -> np.float64:
        r"""Compute least squares estimator for the partials.

        :param J_pred: predicted Jacobian :math:`A^{\prime[L]} \in
            \mathbb{R}^{n_y \times n_x \times m}`
        :param indices: only evaluate specified training data indices
            (used for minibatch)
        """
        dY_true = self.J_true
        dY_error = self.J_error
        n_y, n_x, m = dY_true.shape
        cost = 0.0
        for k in range(0, n_y):
            for j in range(0, n_x):
                dY_error[k, j] = J_pred[k, j] - dY_true[k, j]
                dot_product = np.dot(dY_error[k, j], dY_error[k, j].T)
                cost += np.squeeze(dot_product)
        return np.float64(cost)


class Regularization:
    r"""Compute regularization penalty.

    :param weights: neural parameters :math:`W^{[l]} \in
        \mathbb{R}^{n^{[l]} \times n^{[l-1]}}` associated with each
        layer
    """

    def __init__(self, weights: np.ndarray):  # noqa: D107
        # Preallocate for speed
        self.weights = weights
        self._squared_weights = [np.zeros(W.shape) for W in weights]

    def evaluate(self, lambd: float) -> np.float64:
        r"""Compute L2 norm penalty.

        :param lambd: regularization coefficient :math:`\lambda \in
            \mathbb{R}` (hyperparameter to be tuned)
        """
        penalty = 0.0
        if lambd > 0:
            for i, weight in enumerate(self.weights):
                squared_weights = np.square(weight, out=self._squared_weights[i])
                penalty += np.squeeze(np.sum(squared_weights))
        return lambd * np.float64(penalty)


class Cost:
    r"""Neural Network cost function.

    :param data: Dataset object containing training data (and associated
        metadata)
    :param parameters: object containing neural net parameters (and
        associated metadata) for each layer
    :param lambd: regularization coefficient :math:`\lambda \in
        \mathbb{R}` (hyperparameter to be tuned)
    :param gamma: jacobian-enhancement coefficient :math:`\gamma \in
        \mathbb{R}` (hyperparameter to be tuned)
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
        r"""Evaluate cost function.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_x \times m}`
        :param J_pred: predicted Jacobian :math:`A^{\prime[L]} \in
            \mathbb{R}^{n_y \times n_x \times m}`
        """
        cost = self.squared_loss.evaluate(Y_pred)
        if J_pred is not None and hasattr(self, "gradient_enhancement"):
            cost += self.gradient_enhancement.evaluate(J_pred) * self.gamma
        cost += self.regularization.evaluate(self.lambd)
        cost *= 0.5 / self.data.m
        return cost
