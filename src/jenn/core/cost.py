"""Cost Function.
=================

This module contains class and methods to efficiently 
compute the neural net cost function used for training. 
It is a modified version of the Least Squared Estimator (LSE), 
augmented with a penalty function for regularization and another 
term which accounts for Jacobian prediction error. See
`paper`_ for details and notation. 
"""  # noqa W291

from typing import Optional, Union

import numpy as np
import abc 

from .data import Dataset
from .parameters import Parameters


class Loss: 
    """Base class for loss function."""

    def __init__(self, data: Dataset) -> None:
        self.data = data 

        self.X = data.X
        self.Y_true = data.Y
        self.Y_error = np.zeros(data.Y.shape)  # preallocate to save resources
        self.Y_weights = np.ones(data.Y.shape) *  data.Y_weights
        self.n_y, self.m =  data.Y.shape

        if data.J is not None:  # noqa: PLR2004
            self.J_true = data.J
            self.J_error = np.zeros(data.J.shape)
            self.J_weights = np.ones(data.J.shape) * data.J_weights
            self.n_y, self.n_x, self.m = data.J.shape

    @abc.abstractmethod
    def evaluate(self, Y_pred: np.ndarray) -> np.float64:
        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def evaluate_partials(self, Y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError("To be implemented in subclass.")


class SquaredLoss(Loss):
    r"""Least Squares Estimator.

    :param Y_true: training data outputs :math:`Y \in \mathbb{R}^{n_y
        \times m}`
    :param Y_weights: weights by which to prioritize data points
        (optional)
    """

    def evaluate(self, Y_pred: np.ndarray) -> np.ndarray:
        r"""Compute least squares estimator of the states in place.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_y \times m}`
        """
        self.Y_error[:, :] = 0.5 * self.Y_weights * (Y_pred - self.Y_true) ** 2
        return self.Y_error
    
    def evaluate_partials(self, Y_pred: np.ndarray) -> np.ndarray:
        r"""Compute partials of least squares estimator.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_y \times m}`
        """
        return self.Y_weights * (Y_pred - self.Y_true)


class GradientEnhancement(Loss):
    r"""Least Squares Estimator for partials.

    :param J_true: training data jacobian :math:`Y^{\prime} \in
        \mathbb{R}^{n_y \times m}`
    :param J_weights: weights by which to prioritize partials (optional)
    """

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
    
    def evaluate_partials(self, J_pred: np.ndarray) -> np.ndarray:
        r"""Compute partials of least squares estimator.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_y \times m}`
        """
        return self.J_weights * (J_pred - self.J_true)


class Regularization:
    """Compute regularization penalty."""

    def __init__(self, parameters: Parameters, lambd: float = 0.0) -> None:
        r"""Compute L2 norm penalty.

        :param weights: neural parameters :math:`W^{[l]} \in
            \mathbb{R}^{n^{[l]} \times n^{[l-1]}}` associated with each
            layer
        """
        self.weights = parameters.W
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
        loss_class: Optional[Loss] = None
    ) -> None:
        self.data = data
        self.parameters = parameters
        self.loss = SquaredLoss(data) if loss_class is None else loss_class(data)
        self.regularization = Regularization(parameters, lambd)
        self.gradient_enhancement = GradientEnhancement(data) if data.J is not None else None

    def evaluate(
        self, Y_pred: np.ndarray, J_pred: Union[np.ndarray, None] = None
    ) -> np.float64:
        r"""Evaluate cost function.

        :param Y_pred: predicted outputs :math:`A^{[L]} \in
            \mathbb{R}^{n_x \times m}`
        :param J_pred: predicted Jacobian :math:`A^{\prime[L]} \in
            \mathbb{R}^{n_y \times n_x \times m}`
        """
        loss = self.loss.evaluate(Y_pred)
        cost = np.sum(loss)
        if (J_pred is not None) and (self.gradient_enhancement is not None):
            cost += self.gradient_enhancement.evaluate(J_pred)
        cost += self.regularization.evaluate()
        cost /= self.data.m
        return cost
