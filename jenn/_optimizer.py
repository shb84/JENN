"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List

EPS = np.finfo(float).eps  # small number to avoid division by zero


class Update(ABC):

    @abstractmethod
    def _update(self, params: List[np.ndarray], grads: List[np.ndarray],
                alpha: float) -> List[np.ndarray]:
        raise NotImplementedError

    def __call__(self, params: List[np.ndarray], grads: List[np.ndarray],
                 alpha: float) -> List[np.ndarray]:
        return self._update(params, grads, alpha)


class GD(Update):

    def _update(self, params: List[np.ndarray], grads: List[np.ndarray],
                alpha: float):
        new_params = []
        for k, x in enumerate(params):
            new_params.append(x - alpha * grads[k])
        return new_params


class ADAM(Update):

    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.99):
        """
        Initialize ADAM updating 
        
        Parameters
        ----------
        beta_1: float
            Exponential decay rate for estimates of first moment vector
            in adam, should be in [0, 1). Only used when solver=’adam’
            Default = 0.9

        beta_2: float
            Exponential decay rate for estimates of second moment vector
            in adam, should be in [0, 1). Only used when solver=’adam’
            Default = 0.99
        """
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._v = None
        self._s = None
        self._t = 0
        self._grads = None

    def _update(self, params: List[np.ndarray], grads: List[np.ndarray],
                alpha: float) -> List[np.ndarray]:
        """
        Take a single step in direction of improvement according to ADAM

        Parameters
        ----------
        params: List[np.ndarray]
            List of parameters to be updated

        grads: List[np.ndarray]
            Gradient of objective function w.r.t. each parameter

        alpha: float
            Learning rate

        Returns
        -------
        new_params: List[np.ndarray]
            List of updated parameters
        """

        beta_1 = self.beta_1
        beta_2 = self.beta_2

        v = self._v
        s = self._s
        t = self._t

        if not v:
            v = [np.zeros(param.shape) for param in params]
        if not s:
            s = [np.zeros(param.shape) for param in params]

        a = [id(x) for x in grads]
        b = []
        if self._grads:
            b = [id(x) for x in self._grads]

        if a != b:
            self._grads = grads
            t += 1  # only update for new search directions

        new_params = []
        for k, x in enumerate(params):
            v[k] = beta_1 * v[k] + (1. - beta_1) * grads[k]
            s[k] = beta_2 * s[k] + (1. - beta_2) * np.square(grads[k])
            v_corrected = v[k] / (1. - beta_1 ** t)
            s_corrected = s[k] / (1. - beta_2 ** t) + EPS
            new_params.append(x - alpha * v_corrected / np.sqrt(s_corrected))

        self._v = v
        self._s = s
        self._t = t

        return new_params


class LineSearch(ABC):

    def __init__(self, update: Update):
        self.update = update

    @abstractmethod
    def search(self, params: List[np.ndarray], grads: List[np.ndarray],
               cost: callable, learning_rate: float):
        raise NotImplementedError


class Backtracking(LineSearch):

    def __init__(self, update: Update, tau: float = 0.5,
                 tol: float = 1e-6, max_count: int = 1_000):
        super().__init__(update)
        self.tau = tau
        self.tol = tol
        self.max_count = max_count

    def search(self,
               params: List[np.ndarray], grads: List[np.ndarray],
               cost: callable, learning_rate: float = 0.1):
        """
        Take multiple update steps along search direction determined by update

        Parameters
        ----------
        params: List[np.ndarray]
            List of parameters to be updated

        grads: List[np.ndarray]
            Gradient of objective function w.r.t. each parameter

        cost: callable
            Objective function y = f(x) where x = params

        learning_rate: float
            The maximum allowed step size
        """
        tau = self.tau
        tol = self.tol
        x0 = self.update(params, grads, alpha=0)
        f0 = cost(x0)
        tau = max(0., min(1., tau))
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

    def __init__(self, line: LineSearch):
        self.line_search = line
        self.vars_history = None
        self.cost_history = None

    def minimize(self, x: List[np.ndarray], f: callable,
                 alpha: float = 0.01, max_iter: int = 100,
                 verbose: bool = False, epoch: int = None,
                 batch: int = None) -> List[np.ndarray]:
        """
        Minimize single objective function

        Parameters
        ----------
        x: List[np.ndarray]
            List of parameters to be updated

        f: callable
            Objective function y = f(x)

        alpha: float
            Learning rate
            Default = 0.01

        max_iter: int
            Maximum number of optimizer iterations allowed
            Default = 100

        verbose: bool
            Send progress output to standard out
            Default = False

        epoch: int
            The epoch in which this optimization is being run (only used
            for printing progress output messages)
            Default = None

        batch: int
            The batch in which this optimization is being run (only used
            for printing progress output messages)
            Default = None
        """
        line = self.line_search

        # Stopping criteria (Vanderplaats, ch. 3, p. 121)
        converged = False
        N1 = 0
        N1_max = 100
        N2 = 0
        N2_max = 100

        epsilon_absolute = 1e-6  # absolute error criterion
        epsilon_relative = 1e-6  # relative error criterion

        cost_history = []
        vars_history = []

        # Iterative update
        for i in range(0, max_iter):
            y = f(x)[0].squeeze()

            cost_history.append(y)
            vars_history.append(x)

            x = line.search(params=x, cost=lambda x: f(x)[0], grads=f(x)[1],
                            learning_rate=alpha)

            if verbose:
                if epoch is not None and batch is not None:
                    e = epoch
                    b = batch
                    print(f"epoch = {e:d}, batch = {b:d}, iter = {i:d}, cost = {y:6.3f}")
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
                dF2 =  numerator / denominator

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

    def __init__(self):
        super().__init__(line=Backtracking(update=GD()))


class ADAMOptimizer(Optimizer):

    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.99):
        adam = ADAM(beta_1, beta_2)
        super().__init__(line=Backtracking(update=adam))
