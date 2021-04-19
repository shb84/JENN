"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Union, List
from ._fwd_prop import L_model_forward, L_grads_forward
from ._bwd_prop import L_model_backward
from ._utils import (mini_batches, grad_check, finite_diff, goodness_of_fit,
                     DataConverter, scale_factor)
from ._loss import squared_loss, gradient_enhancement, regularization
from ._optimizer import Optimizer, Backtracking, ADAM, GD

from importlib.util import find_spec

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

EPS = np.finfo(float).eps  # small number to avoid division by zero


class GENNBase:
    """ Gradient-Enhanced Neural Net (GENN) """

    def __init__(self,
                 hidden_layer_sizes: Union[List[int], tuple] = (100,),
                 activation: str = "relu", *,
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 gamma: int = 1,
                 batch_size: int = None,
                 learning_rate: str = "constant",
                 learning_rate_init: float = 0.001,
                 num_epochs: int = 1,
                 max_iter: int = 100,
                 shuffle: bool = True,
                 random_state: int = None,
                 tol: float = 1e-4,
                 verbose: bool = False,
                 warm_start: bool = False,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 is_finite_difference: bool = False,
                 is_grad_check: bool = False):
        """
        Initialize object

        Parameters
        ----------
        hidden_layer_sizes: List[int]
            The ith element represents the number of neurons
            in the ith hidden layer
            Default = (110,)

        activation:str
            Activation function for the hidden layer.
            option = {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
            Default = ’tanh’

        solver: str
            The type of solver to use for gradient-based optimization
            Options = {‘sgd’, ‘adam’}
            Default = 'adam'

        alpha: float
            L2 penalty (regularization term) parameter
            Default = 1e-4

        gamma:
            Parameter controlling influence of partials
            squared loss term during training
            Default = 1

        batch_size: str or int
            Size of mini-batches for stochastic optimizers.
            When set to “auto”, batch_size=min(200, n_samples)
            Default = 'auto'

        learning_rate: str
            Learning rate schedule for weight updates.
            Options =
                ‘constant’ = constant learning rate given by learning_rate_init
                ‘backtracking’ = learning is optimized during each line search
            Default = 'constant'

        learning_rate_init: float
            The initial learning rate used. It controls the step-size
            during line search, when updating the neural net parameters.
            Default = 1e-3

        num_epochs: int
            Number of passes through data
            Default = 1

        max_iter: int
            Number of optimizer iterations. Note that optimizer is called once
            per batch for each epoch. For example: 10 epochs, 5 batches, 100
            max_iter results in 10 * 5 * 100 = 5_000 function calls to train.
            Default = 100

        shuffle: bool
            Whether to shuffle batch samples for each epoch
            Default = True

        random_state: int
            Fix the random state to ensure repeatable results
            Default = None

        tol: float
            Optimizer tolerance
            Default = 1e-4

        verbose: bool
            Whether to print progress messages to stdout
            Default = False

        warm_start: bool
            When set to True, reuse the solution of the previous call to fit
            as initialization, otherwise, just erase the previous solution
            Default = False

        beta_1: float
            Exponential decay rate for estimates of first moment vector
            in adam, should be in [0, 1). Only used when solver=’adam’
            Default = 0.9

        beta_2: float
            Exponential decay rate for estimates of second moment vector
            in adam, should be in [0, 1). Only used when solver=’adam’
            Default = 0.99

        is_finite_difference: bool
            This option is intended for debugging. Finite difference is
            used instead of backpropagation during training.
            Default = False

        is_grad_check: bool
            This option is intended for debugging. The analytical gradient
            computed using back-prop is checked against finite difference at
            every optimizer iteration.
            Default = False

        normalize: bool
            Normalize the training data according to mean and variance. This
            can often greatly help optimization as it makes the optimization
            problem easier to navigate and solve efficiently.
            Default = False
        """

        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.num_epochs = num_epochs
        self.max_iter = max_iter
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.is_finite_difference = is_finite_difference
        self.is_grad_check = is_grad_check

        # Dimensionality of inputs and outputs
        self._n_x = None
        self._n_y = None

        # Parameters
        self._W = None  # coefficients
        self._b = None  # intercepts
        self._a = None  # activations

        # Training history
        self._cost_history = None

    def _cost(self, W: List[np.ndarray], b: List[np.ndarray],
              X: np.ndarray, Y: np.ndarray, J: np.ndarray = None,
              is_grad: bool = True):
        """

        Parameters
        ----------
        W: List[np.ndarray]
        b: List[np.ndarray]
        X: np.ndarray
        Y: np.ndarray
        J: np.ndarray
        is_grad: bool

        Return
        ------
        c, dW, db : Tuple[np.float, List[np.ndarray], List[np.ndarray]]
            c = cost
            dW = d(cost)/dW = derivative of cost w.r.t. neural net weights
            db = d(cost)/db = derivative of cost w.r.t. neural net biases
        """
        # Predict
        Y_pred, caches = L_model_forward(X, W, b, self._a)
        J_pred, J_caches = L_grads_forward(X, W, b, self._a)

        # Cost function
        c = 0
        c = c + squared_loss(Y, Y_pred)
        c = c + regularization(W, X.shape[1], self.alpha)
        if J is not None:
            c = c + gradient_enhancement(J_pred, J, self.gamma)

        # Cost function gradient
        dW = []
        db = []
        if is_grad:
            dW, db = L_model_backward(Y_pred, Y, caches, J_pred, J, J_caches,
                                      self.alpha, self.gamma)
        return c, dW, db

    def _initialize(self):
        """ Initialize neural net using "He" initialization """
        if not self._n_y:
            raise ValueError(f'Missing output layer: n_y = {self._n_y}')

        if not self.hidden_layer_sizes:
            raise ValueError(
                "Neural net does not seem to have any hidden layers")

        np.random.seed(self.random_state)

        layer_dims = [self._n_x] + self.hidden_layer_sizes + [self._n_y]

        # Parameters
        self._W = []
        self._b = []
        number_layers = len(layer_dims)
        for l in range(1, number_layers):
            W = np.random.randn(layer_dims[l], layer_dims[l - 1]) \
                * np.sqrt(1. / layer_dims[l - 1])
            b = np.zeros((layer_dims[l], 1))
            self._W.append(W)
            self._b.append(b)

    def _train(self, X: np.ndarray, Y: np.ndarray, J: np.ndarray = None):
        """
        Internal support function in charge of training the neural net

        Parameters 
        ----------
        X: matrix of shape (n_x, m) where n_x = no. of inputs
                                          m = no. of training examples
        Y: matrix of shape (n_y, m) where n_y = no. of outputs
        J: np.ndarray of size (n_y, n_x, m) where J = Jacobian
                                                      dY1/dX1 = J[0, 0, :]
                                                      dY1/dX2 = J[0, 1, :]
                                                      ...
                                                      dY2/dX1 = J[1, 0, :]
                                                      dY2/dX2 = J[1, 1, :]
                                                      ...

        Returns
        -------
        cost_history : defaultdict(dict)
            Cost function values for each epoch and batch:
                cost_history["epoch_e"]["batch_b"] = c
                where e is the epoch number
                      b is the batch number
                      c is the cost as an np.ndarray (c[i] = cost of i^th iter)
        """
        if X.ndim != 2:
            raise ValueError(f'X.ndim = {X.ndim} but X.shape should (n_x, m)')

        if Y.ndim != 2:
            raise ValueError(f'Y.ndim = {Y.ndim} but X.shape should (n_y, m)')

        self._n_x = X.shape[0]
        self._n_y = Y.shape[0]

        # Initialize
        if not self.warm_start or self._W is None or self._b is None:
            self._initialize()
        else:
            # Check that user did not change dimensions
            if len(self._W) - 1 != len(self.hidden_layer_sizes):
                raise ValueError(
                    f'warm start flag raised, but number of hidden layers '
                    f'no longer matches previously trained neural net.')
            if self._n_x != X.shape[0]:
                raise ValueError(
                    f'warm start flag raised, but new input layer does '
                    f'not match shape of previous input layer.')
            if self._n_y != Y.shape[0]:
                raise ValueError(
                    f'warm start flag raised, but new output layer does '
                    f'not match shape of previous output layer.')

        hidden_activation = [self.activation] * len(self.hidden_layer_sizes)
        output_activation = ['identity']
        self._a = hidden_activation + output_activation

        cost_history = defaultdict(dict)

        n_x, m = X.shape
        n_y, _ = Y.shape

        if J is None:
            J = np.zeros((n_y, n_x, m))
            self.gamma = 0  # Make sure gradient-enhancement is off if J = None

        for e in range(self.num_epochs):
            batches = mini_batches(X, self.batch_size,
                                   random_state=self.random_state,
                                   shuffle=self.shuffle)
            for b, batch in enumerate(batches):

                X_batch = X[:, batch]
                Y_batch = Y[:, batch]
                J_batch = J[:, :, batch]

                def f(x, is_grad=True):
                    """ Cost function and gradient """
                    W = x[:len(self._W)]
                    b = x[len(self._W):]
                    y, dW, db = self._cost(W, b, X_batch, Y_batch, J_batch,
                                           is_grad)
                    dy_dx = dW + db
                    return y, dy_dx

                def f_FD(x):
                    """ Debug: cost function with finite difference grad """
                    y = f(x, is_grad=False)[0]
                    dy_dx = finite_diff(x, lambda x: f(x, is_grad=False)[0])
                    return y, dy_dx

                def f_check(x):
                    """ Debug: cost function with FD vs. analytic grad """
                    grad_check(x, f=lambda x: f(x, is_grad=False)[0],
                               dfdx=lambda x: f(x, is_grad=True)[1])
                    if self.is_finite_difference:
                        return f_FD(x)
                    else:
                        return f(x)

                cost = f
                if self.is_grad_check:
                    cost = f_check
                elif self.is_finite_difference:
                    cost = f_FD

                if self.solver == 'adam':
                    update = ADAM(self.beta_1, self.beta_2)
                elif self.solver == 'sgd':
                    update = GD()
                else:
                    raise ValueError(f'solver = {self.solver} not recognized')

                if self.learning_rate == 'constant':
                    line_search = Backtracking(update,
                                               max_count=0, tol=self.tol)
                elif self.learning_rate == 'backtracking':
                    line_search = Backtracking(update, tol=self.tol)
                else:
                    raise ValueError(f'learning_rate = '
                                     f'{self.learning_rate} not recognized')

                optimizer = Optimizer(line_search)

                params = optimizer.minimize(x=self._W + self._b, f=cost,
                                            alpha=self.learning_rate_init,
                                            max_iter=self.max_iter,
                                            verbose=self.verbose,
                                            epoch=e, batch=b)

                self._W = params[:len(self._W)]
                self._b = params[len(self._W):]

                key1 = 'epoch_' + str(e)
                key2 = 'batch_' + str(b)
                cost_history[key1][key2] = np.array(optimizer.cost_history)

        self._cost_history = cost_history


class JENN(GENNBase):

    def __init__(self, *args, **kwargs):
        self._data_converter = None
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray, J: np.ndarray = None,
            is_normalize: bool = False):
        """
        Minimize augmented squared loss in order to learn neural net parameters

        Parameters
        ----------
        X: np.ndarray
            Training data features
            shape = (m, n_x) where m = no. examples, n_x = no. features

        Y: np.ndarray
            Training data responses
            shape = (m, n_y) where n_y = no. responses

        J: np.ndarray
            Training data Jacobian, J = dY/dX
            shape = (m, n_x, n_y)

        is_normalize: bool
            Option to normalize data by mean and variance before fitting
            Default = False
        """
        X = X.T  # The algorithm assumes X.shape = (n_x, m)
        Y = Y.T  # The algorithm assumes Y.shape = (n_y, m)
        if J is not None:
            J = J.T  # The algorithm assumes J.shape = (n_y, n_x, m)

        if is_normalize:
            mu_x, sigma_x = scale_factor(X)
            mu_y, sigma_y = scale_factor(Y)
            self._data_converter = DataConverter(mu_x, sigma_x, mu_y, sigma_y)
        else:
            self._data_converter = None

        if self._data_converter is not None:
            X_norm = self._data_converter.X_norm(X)
            Y_norm = self._data_converter.Y_norm(Y)
            J_norm = self._data_converter.J_norm(J)
            self._train(X_norm, Y_norm, J_norm)
        else:
            self._train(X, Y, J)

    def predict(self, X: np.ndarray):
        """
        Predict response(s) using trained neural network

        Parameters
        ----------
            X: np.ndarray
                Training data features
                shape = (m, n_x) where m = no. examples, n_x = no. features

        Returns
        -------
            Y: np.ndarray
                Training data responses
                shape = (m, n_y) where n_y = no. responses
        """
        X = X.T  # The algorithm assumes X.shape = (n_x, m)
        if self._data_converter is not None:
            X_norm = self._data_converter.X_norm(X)
            Y_norm = L_model_forward(X_norm, self._W, self._b, self._a,
                                     store_cache=False)
            Y = self._data_converter.Y(Y_norm)
        else:
            Y = L_model_forward(X, self._W, self._b, self._a,
                                store_cache=False)
        return Y.T  # The algorithm returns Y.shape = (n_y, m)

    def jacobian(self, X: np.ndarray):
        """
        Predict Jacobian using trained neural network

        Parameters
        ----------
        X: np.ndarray
            Training data features
            shape = (m, n_x) where m = no. examples, n_x = no. features

        Returns
        -------
        J: np.ndarray
            Training data Jacobian, J = dY/dX of shape = (m, n_x, n_y)
        """
        X = X.T  # The algorithm assumes X.shape = (n_x, m)
        if self._data_converter is not None:
            X_norm = self._data_converter.X_norm(X)
            J_norm = L_grads_forward(X_norm, self._W, self._b, self._a,
                                     store_cache=False)
            J = self._data_converter.J(J_norm)
        else:
            J = L_grads_forward(X, self._W, self._b, self._a,
                                store_cache=False)
        return J.T  # The algorithm returns J.shape = (n_y, n_x, m)

    def goodness_fit(self, X: np.ndarray, Y_true: np.ndarray,
                     title: str = None, legend: str = None,
                     show_error: bool = True):
        """
        Plot goodness of fit

        Parameters
        ----------
        X: np.ndarray
            Inputs, shape = (m, n_x) where m = no. examples, n_x = no. features

        Y_true: np.ndarray
            Outputs, shape = (m, n_y) where m = no. examples, n_y = no. outputs
        """
        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib must be installed.")

        if self._n_x != X.shape[1]:
            raise ValueError(f'X.shape[1] = {X.shape[1]} '
                             f'but expected {self._n_x}')

        if self._n_y != Y_true.shape[1]:
            raise ValueError(f'Y_true.shape[1] = {Y_true.shape[1]} '
                             f'but expected {self._n_y}')

        Y_pred = self.predict(X)

        figs = []
        for i in range(self._n_y):
            y_pred = Y_pred[:, i].ravel()
            y_true = Y_true[:, i].ravel()
            if title is None:
                title = f'Goodness of Fit: Y[{i}]'
            fig = goodness_of_fit(y_pred, y_true, title, legend, show_error)
            figs.append(fig)

        return figs

    def training_history(self, title: str = 'Training History'):

        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib must be installed.")

        if not self.training_history:
            return None

        epochs = list(self._cost_history.keys())
        if len(epochs) > 1:
            avg_costs = []
            for epoch in epochs:
                batches = self._cost_history[epoch].keys()
                avg_batch_costs = []
                for batch in batches:
                    avg_batch_cost = self._cost_history[epoch][batch].mean()
                    avg_batch_costs.append(avg_batch_cost)
                avg_cost = sum(avg_batch_costs) / len(batches)
                avg_costs.append(avg_cost)
            plt.plot(range(len(epochs)), avg_costs)
            plt.xlabel('epoch')
            plt.ylabel('avg cost')
        elif len(self._cost_history['epoch_0']) > 1:
            avg_cost = []
            batches = self._cost_history['epoch_0'].keys()
            for batch in batches:
                avg_cost.append(self._cost_history['epoch_0'][batch].mean())
            plt.plot(range(len(batches)), avg_cost)
            plt.xlabel('batch')
            plt.ylabel('avg cost')
        else:
            cost = self._cost_history['epoch_0']['batch_0']
            plt.plot(np.arange(cost.size), cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
        plt.title(title)
