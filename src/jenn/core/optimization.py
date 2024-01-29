"""Gradient-Based Optimization."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Update(ABC):
    """Take a single step along the search direction.

    Base class. The search direction is determined by the "update"
    method implemented in the base class.
    """

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
        """Take a single step along the search direction.

            x_new = x + alpha * search_direction

        Parameters
        ----------
        params: np.ndarray
            Parameters to be updated

        grads: np.ndarray
            Gradient of objective function w.r.t. each parameter

        alpha: float
            Learning rate
        """
        return self._update(params, grads, alpha)


class GD(Update):
    """Take a single step along the search direction.

    The search direction is determined using gradient descent.
    """

    def _update(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        return (params - alpha * grads).reshape(params.shape)


class ADAM(Update):
    """Take a single step along search direction.

    The search direction is determined using ADAM [REF].

    [REF] Kingma, D. P. and Ba, J., “Adam: A
          Method for Stochastic Optimization,” 12 2014.

    Parameters
    ----------
    beta_1: float
        Exponential decay rate for estimates of first moment vector
        in adam, should be in [0, 1). Only used when solver="adam"
        Default is 0.9

    beta_2: float
        Exponential decay rate for estimates of second moment vector
        in adam, should be in [0, 1). Only used when solver="adam"
        Default is 0.99
    """

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ):  # noqa D107
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._v = None
        self._s = None
        self._t = 0
        self._grads = None

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
    """Take multiple steps along the search direction.

    Parameters
    ----------
    update: Update
        Take a single step along the search direction.
        Note: the update algorithm determines the search
        direction based on the current value of the gradient.
        x_new = x + alpha * search_direction
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
        """Take multiple steps along the search direction.

        Parameters
        ----------
        params: np.ndarray
            Parameters to be updated

        grads: np.ndarray
            Gradient of objective function w.r.t. each parameter

        cost: Callable
            Cost function: cost = f(params)

        learning_rate: float
            The initial step size.

        Returns
        -------
        new_params: np.ndarray
            Updated parameters
        """
        raise NotImplementedError


class Backtracking(LineSearch):
    """Search for optimum along a search direction.

    Parameters
    ----------
    update: Update
        Take a single step along the search direction.
        Note: the update algorithm determines the search
        direction based on the current value of the gradient.
        x_new = x + alpha * search_direction

    tau: float, optional
        Amount by which to reduce alpha (step size) on each iteration.
        alpha_new = alpha * tau
        Default is 0.5

    tol: float, optional
        Tolerance criterion to stop iteration (i.e. stop whenever cost
        function doesn't improve more than tol). Default is 1e-6.

    max_count: int, optional
        Stop when line search iterations exceed max_count. Default is 1000.
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
        """Take multiple update steps along search direction.

        Parameters
        ----------
        params: np.ndarray
            Parameters to be updated

        grads: np.ndarray
            Gradient of objective function w.r.t. each parameter

        cost: Callable
            Objective function y = f(x) where x = params

        learning_rate: float, optional
            The maximum allowed step size. Default is 0.05
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
    """Find optimum using gradient-based optimization.

    Parameters
    ----------
    line_search: LineSearch
        The algorithm to use for determine best search direction
        given current value of gradient and searching for optimum
        along that direction.
    """

    def __init__(
        self,
        line_search: LineSearch,
    ):  # noqa D107
        self.line_search = line_search
        self.vars_history: list[np.ndarray] | None = None
        self.cost_history: list[np.ndarray] | None = None

    def minimize(
        self,
        x: np.ndarray,
        f: Callable,
        dfdx: Callable,
        alpha: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> np.ndarray:
        """Minimize single objective function.

        Parameters
        ----------
        x: np.ndarray
            Parameters to be updated

        f: Callable
            Objective function y = f(x)

        alpha: float
            Learning rate
            Default is 0.01

        max_iter: int
            Maximum number of optimizer iterations allowed
            Default is 100

        verbose: bool
            Send progress output to standard out
            Default is False

        epoch: int
            The epoch in which this optimization is being run (only used
            for printing progress output messages)
            Default is None

        batch: int
            The batch in which this optimization is being run (only used
            for printing progress output messages)
            Default is None
        """
        # Stopping criteria (Vanderplaats, ch. 3, p. 121)
        converged = False
        N1 = 0
        N1_max = 100
        N2 = 0
        N2_max = 100

        epsilon_absolute = 1e-6  # absolute error criterion
        epsilon_relative = 1e-6  # relative error criterion

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
                denominator = max(abs(cost_history[-1]), 1e-6)
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
    """Search for optimum using gradient descent.

    Parameters
    ----------
    tau: float, optional
        Backtracking line search parameter. Amount by which
        to reduce alpha (step size) on each iteration.
        alpha_new = alpha * tau
        Default is 0.5

    tol: float, optional
        Backtracking line search parameter. Tolerance criterion
        to stop iteration (i.e. stop whenever cost function
        doesn't improve more than tol). Default is 1e-6.

    max_count: int, optional
        Backtracking line search parameter. Stop when line search
        iterations exceed max_count. Default is 1000.
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
    """Search for optimum using ADAM algorithm.

    Parameters
    ----------
    beta_1: float
        Exponential decay rate for estimates of first moment vector
        in adam, should be in [0, 1). Only used when solver="adam"
        Default is 0.9

    beta_2: float
        Exponential decay rate for estimates of second moment vector
        in adam, should be in [0, 1). Only used when solver="adam"
        Default is 0.99

    tau: float, optional
        Backtracking line search parameter. Amount by which
        to reduce alpha (step size) on each iteration.
        alpha_new = alpha * tau
        Default is 0.5

    tol: float, optional
        Backtracking line search parameter. Tolerance criterion
        to stop iteration (i.e. stop whenever cost function
        doesn't improve more than tol). Default is 1e-6.

    max_count: int, optional
        Backtracking line search parameter. Stop when line search
        iterations exceed max_count. Default is 1000.
    """

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 0.5,
        tol: float = 1e-6,
        max_count: int = 1_000,
    ):  # noqa D107
        line_search = Backtracking(
            update=ADAM(beta_1, beta_2),
            tau=tau,
            tol=tol,
            max_count=max_count,
        )
        super().__init__(line_search)
