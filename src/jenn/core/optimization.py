"""Optimization.
================

.. ADAM: https://doi.org/10.48550/arXiv.1412.6980

This module implements gradient-based optimization using `ADAM`_. 
"""  # noqa W291

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import numpy as np


class Update(ABC):
    r"""Base class for line search.

    Update parameters :math:`\boldsymbol{x}` by taking a step along
    the search direction :math:`\boldsymbol{s}` according to
    :math:`\boldsymbol{x} := \boldsymbol{x} + \alpha
    \boldsymbol{s}`

    .. automethod:: __call__
    """  # noqa W291

    @abstractmethod
    def _update(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        raise NotImplementedError("To be implemented in subclass.")

    def __call__(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        r"""Take a single step along search direction.

        :param params: parameters :math:`x` to be updated
        :param grads: gradient :math:`\nabla_x f` of
            objective function :math:`f` w.r.t. each parameter
            :math:`x`
        :param alpha: learning rate :math:`\alpha`
        """
        return self._update(params, grads, alpha)


class GD(Update):
    r"""Take single step along the search direction using gradient descent.

    GD simply follows the steepest path according to
    :math:`\boldsymbol{x} := \boldsymbol{x} + \alpha \boldsymbol{s}`
    where :math:`\boldsymbol{s} = \nabla_x f`
    """

    def _update(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        return (params - alpha * grads).reshape(params.shape)


class ADAM(Update):
    r"""Take single step along the search direction as determined by `ADAM`_.

    Parameters :math:`\boldsymbol{x}` are updated according to
    :math:`\boldsymbol{x} := \boldsymbol{x} + \alpha \boldsymbol{s}`
    where :math:`\boldsymbol{s}` is determined by ADAM in such a way to
    improve efficiency. This is accomplished making use of previous
    information (see paper).

    :param beta_1: exponential decay rate of 1st moment vector
        :math:`\beta_1\in[0, 1)`
    :param beta_2: exponential decay rate of 2nd moment vector
        :math:`\beta_2\in[0, 1)`
    """

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ):  # noqa D107
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._v: Union[np.ndarray, None] = None
        self._s: Union[np.ndarray, None] = None
        self._t = 0
        self._grads: Union[np.ndarray, None] = None

    def _update(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        beta_1 = self.beta_1
        beta_2 = self.beta_2

        v = self._v
        s = self._s
        t = self._t

        if v is None:
            v = np.zeros(params.shape)
        if s is None:
            s = np.zeros(params.shape)

        a: list[int] = [id(x) for x in grads]
        b: list[int] = []
        if self._grads is not None:
            b = [id(x) for x in self._grads]

        if a != b:
            self._grads = grads
            t += 1  # only update for new search directions

        epsilon = np.finfo(float).eps  # small number to avoid division by zero
        v = beta_1 * v + (1.0 - beta_1) * grads
        s = beta_2 * s + (1.0 - beta_2) * np.square(grads)
        v_corrected = v / (1.0 - beta_1**t)
        s_corrected = s / (1.0 - beta_2**t) + epsilon

        x = params - alpha * v_corrected / np.sqrt(s_corrected)

        self._v = v
        self._s = s
        self._t = t

        return x.reshape(params.shape)


class LineSearch(ABC):
    r"""Take multiple steps of varying size by progressively varying
    :math:`\alpha` along the search direction.

    :param update: object that implements Update base class to update
        parameters according to :math:`\boldsymbol{x} :=
        \boldsymbol{x} + \alpha \boldsymbol{s}`

    .. automethod:: __call__
    """

    def __init__(
        self,
        update: Update,
    ):  # noqa D107
        self.update = update

    @abstractmethod
    def __call__(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        cost: Callable,
        learning_rate: float,
    ) -> np.ndarray:
        r"""Take multiple steps along the search direction.

        :param params: parameters to be updated, array of shape (n,)
        :param grads: cost function gradient w.r.t. parameters, array of
            shape (n,)
        :param cost: cost function, array of shape (1,)
        :param learning_rate: initial step size :math:`\alpha`
        :return: new_params: updated parameters, array of shape (n,)
        """
        raise NotImplementedError


class Backtracking(LineSearch):
    r"""Search for optimum along a search direction.

    :param update: object that updates parameters according to :math:`\boldsymbol{x} := \boldsymbol{x} + \alpha \boldsymbol{s}`
    :param tau: amount by which to reduce :math:`\alpha := \tau \times \alpha` on each iteration
    :param tol: stop when cost function doesn't improve more than specified tolerance
    :param max_count: stop when line search iterations exceed maximum count specified

    .. automethod:: __call__
    """

    def __init__(
        self,
        update: Update,
        tau: float = 0.5,
        tol: float = 1e-6,
        max_count: int = 1_000,
    ):  # noqa D107
        super().__init__(update)
        self.tau = tau
        self.tol = tol
        self.max_count = max_count

    def __call__(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        cost: Callable,
        learning_rate: float = 0.05,
    ) -> np.ndarray:
        r"""Take multiple "update" steps along search direction.

        :param params: parameters :math:`x` to be updated, array of
            shape (n,)
        :param grads: gradient :math:`\nabla_x f` of
            objective function :math:`f` w.r.t. each
            parameter, array of shape (n,)
        :param cost: objective function :math:`f`
        :param learning_rate: maximum allowed step size :math:`\alpha
            \le \alpha_{max}`
        :return: updated parameters :math:`x`, array of shape (n,)
        """
        tau = self.tau
        tol = self.tol
        x0 = self.update(params, grads, alpha=0)
        f0 = cost(x0)
        tau = max(0.0, min(1.0, tau))
        alpha = learning_rate
        x = self.update(params, grads, alpha)
        max_count = max(1, self.max_count)
        for _ in range(max_count):
            if cost(x) < f0:
                return x
            elif alpha < tol:
                return x
            else:
                alpha = learning_rate * tau
                x = self.update(params, grads, alpha)
                tau *= tau
        return x


class Optimizer:
    r"""Find optimum using gradient-based optimization.

    :param line_search: object that implements algorithm to compute
        search direction :math:`\boldsymbol{s}` given the gradient
        :math:`\nabla_x f` at the current parameter values
        :math:`\boldsymbol{x}` and take multiple steps along it to
        update them according to :math:`\boldsymbol{x} := \boldsymbol{x}
        + \alpha \boldsymbol{s}`
    """

    def __init__(
        self,
        line_search: LineSearch,
    ):  # noqa D107
        self.line_search = line_search
        self.vars_history: Union[list[np.ndarray], None] = None
        self.cost_history: Union[list[np.ndarray], None] = None

    def minimize(
        self,
        x: np.ndarray,
        f: Callable,
        dfdx: Callable,
        alpha: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
        epoch: Union[int, None] = None,
        batch: Union[int, None] = None,
        epsilon_absolute: float = 1e-12,
        epsilon_relative: float = 1e-12,
    ) -> np.ndarray:
        r"""Minimize single objective function.

        :param x: parameters to be updated, array of shape (n,)
        :param f: cost function :math:`y = f(\boldsymbol{x})`
        :param alpha: learning rate :math:`\boldsymbol{x} :=
            \boldsymbol{x} + \alpha \boldsymbol{s}`
        :param max_iter: maximum number of optimizer iterations allowed
        :param verbose: whether or not to send progress output to
            standard out
        :param epoch: the epoch in which this optimization is being run
            (for printing)
        :param batch: the batch in which this optimization is being run
            (for printing)
        :param epsilon_absolute: absolute error stopping criterion
        :param epsilon_relative: relative error stopping criterion
        """
        # Stopping criteria (Vanderplaats, "Multidiscipline Design Optimization," ch. 3, p. 121)
        converged = False
        N1 = 0
        N1_max = 100
        N2 = 0
        N2_max = 100

        cost_history: list[np.ndarray] = []
        vars_history: list[np.ndarray] = []

        # Iterative update
        for i in range(0, max_iter):
            y = f(x)

            cost_history.append(y)
            vars_history.append(x)

            x = self.line_search(params=x, cost=f, grads=dfdx(x), learning_rate=alpha)

            if verbose:
                if epoch is not None and batch is not None:
                    e = epoch
                    b = batch
                    print(
                        f"epoch = {e:d}, batch = {b:d}, iter = {i:d}, cost = {y:6.3f}"
                    )
                elif epoch is not None:
                    e = epoch
                    print(f"epoch = {e:d}, iter = {i:d}, cost = {y:6.3f}")
                elif batch is not None:
                    b = batch
                    print(f"batch = {b:d}, iter = {i:d}, cost = {y:6.3f}")
                else:
                    print(f"iter = {i:d}, cost = {y:6.3f}")

            # Absolute convergence criterion
            if i > 1:
                dF1 = abs(cost_history[-1] - cost_history[-2])
                if dF1 < epsilon_absolute * cost_history[0]:
                    N1 += 1
                else:
                    N1 = 0
                if N1 > N1_max:
                    converged = True
                    if verbose:
                        print("Absolute stopping criterion satisfied")

                # Relative convergence criterion
                numerator = abs(cost_history[-1] - cost_history[-2])
                denominator = max(abs(float(cost_history[-1])), 1e-6)
                dF2 = numerator / denominator

                if dF2 < epsilon_relative:
                    N2 += 1
                else:
                    N2 = 0
                if N2 > N2_max:
                    converged = True
                    if verbose:
                        print("Relative stopping criterion satisfied")

                if converged:
                    break

            # Maximum iteration convergence criterion
            if i == max_iter:
                if verbose:
                    print("Maximum optimizer iterations reached")

        self.cost_history = cost_history
        self.vars_history = vars_history

        return x


class GDOptimizer(Optimizer):
    r"""Search for optimum using gradient descent.

    .. warning::
        This optimizer is very inefficient. It was intended
        as a baseline during development. It is not recommended. Use ADAM instead.

    :param tau: amount by which to reduce :math:`\alpha := \tau \times \alpha` on each iteration
    :param tol: stop when cost function doesn't improve more than specified tolerance
    :param max_count: stop when line search iterations exceed maximum count specified
    """

    def __init__(
        self,
        tau: float = 0.5,
        tol: float = 1e-6,
        max_count: int = 1_000,
    ):  # noqa D107
        line_search = Backtracking(
            update=GD(),
            tau=tau,
            tol=tol,
            max_count=max_count,
        )
        super().__init__(line_search)


class ADAMOptimizer(Optimizer):
    r"""Search for optimum using ADAM algorithm.

    :param beta_1: exponential decay rate of 1st moment vector
        :math:`\beta_1\in[0, 1)`
    :param beta_2: exponential decay rate of 2nd moment vector
        :math:`\beta_2\in[0, 1)`
    :param tau: amount by which to reduce :math:`\alpha := \tau \times
        \alpha` on each iteration
    :param tol: stop when cost function doesn't improve more than
        specified tolerance
    :param max_count: stop when line search iterations exceed maximum
        count specified
    """

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 0.5,
        tol: float = 1e-12,
        max_count: int = 1_000,
    ):  # noqa D107
        line_search = Backtracking(
            update=ADAM(beta_1, beta_2),
            tau=tau,
            tol=tol,
            max_count=max_count,
        )
        super().__init__(line_search)
