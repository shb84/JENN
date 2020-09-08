import numpy as np
from collections import defaultdict
from typing import Tuple, Union, List

from ._fwd_prop import L_model_forward, L_grads_forward
from ._utils import (mini_batches, grad_check, finite_diff,
                     goodness_of_fit, rsquare, normalize_data)
from ._loss import cost
from ._optimizer import Optimizer, Backtracking, ADAM, GD

from importlib.util import find_spec

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

EPS = np.finfo(float).eps  # small number to avoid division by zero


class GENN:
    """ Gradient-Enhanced Neural Net (GENN) """

    def __init__(self,
                 hidden_layer_sizes: Union[List[int], tuple] = (100,),
                 activation: str = "relu", *,
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 gamma: int = 1,
                 batch_size: int or str = 'auto',
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
                 is_grad_check: bool = False,
                 normalize: bool = False):
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
        self.normalize = normalize

        # Dimensionality of inputs and outputs
        self._n_x = None
        self._n_y = None

        # Parameters
        self._W = None  # coefficients
        self._b = None  # intercepts
        self._a = None  # activations

        # Normalization
        self._mu_x = np.array([[0]])
        self._sigma_x = np.array([[1]])
        self._mu_y = np.array([[0]])
        self._sigma_y = np.array([[1]])

        # Training history
        self._cost_history = None

    @property
    def coefficients(self):
        return self._W

    @property
    def intercepts(self):
        return self._W

    def _preprocess(self, X: np.ndarray, Y: np.ndarray, J: np.ndarray = None):
        """
        Check, transpose, and normalize input data as assumed by GENN algorithm

        Scikit learn assumes  X.shape = (n_samples, n_features)
                              Y.shape = (n_samples, n_responses)
                              J.shape = (n_samples, n_features, n_responses)
        However, GENN assumes X.shape = (n_features, n_samples)
                              Y.shape = (n_responses, n_samples)
                              J.shape = (n_responses, n_features, n_samples)
        where J = Jacobian

        Parameters
        ----------
        X: np.ndarray
            Training data features
            shape = (m, n_x) where m = no. examples, n_x = no. features

        Y: np.ndarray
            Training data responses
            shape = (m, n_x) where n_y = no. responses

        J: np.ndarray
            Training data Jacobians
            shape = (m, n_x, n_y)
            Default = None

        Returns
        -------
        X_norm, Y_norm, Z_norm : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Transposed and normalized inputs
        """
        # Force user to provide inputs with correct shapes
        if X.ndim != 2:
            message = f"Expected 2D array, but got shape X.shape = {X.shape}"
            raise ValueError(message)
        if Y.ndim != 2:
            message = f"Expected 2D array, but got shape Y.shape = {Y.shape}"
            raise ValueError(message)
        if J is not None and J.ndim != 3:
            message = f"Expected 3D array, but got shape J.shape = {J.shape}"
            raise ValueError(message)

        m, n_x = X.shape
        n_y = Y.shape[1]

        self._n_x = n_x
        self._n_y = n_y

        if J is not None:

            msg = f'Input dimensions do not agree. ' \
                  f'Expected: X.shape = (n_samples, n_features)' \
                  f'Expected: Y.shape = (n_samples, n_responses)' \
                  f'Expected: J.shape = (n_samples, n_features, n_responses)'

            if m != Y.shape[0]:
                raise ValueError(msg)
            if m != J.shape[0]:
                raise ValueError(msg)
            if n_x != J.shape[1]:
                raise ValueError(msg)
            if n_y != J.shape[2]:
                raise ValueError(msg)

        else:

            msg = f'Input dimensions do not agree. ' \
                  f'Expected: X.shape = (n_samples, n_features)' \
                  f'Expected: Y.shape = (n_samples, n_responses)'

            if X.shape[0] != Y.shape[0]:
                raise ValueError(msg)

        X = X.T
        Y = Y.T
        if J is not None:
            J = J.T

        self._mu_x = np.array([[0]])
        self._sigma_x = np.array([[1]])
        self._mu_y = np.array([[0]])
        self._sigma_y = np.array([[1]])

        if self.normalize:
            results = normalize_data(X, Y, J)
            X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y = results

            self._mu_x = mu_x
            self._mu_y = mu_y
            self._sigma_x = sigma_x
            self._sigma_y = sigma_y
            return X_norm, Y_norm, J_norm
        else:
            return X, Y, J

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
        W = []
        b = []
        number_layers = len(layer_dims)
        for l in range(1, number_layers):
            coefficients = np.random.randn(layer_dims[l],
                                           layer_dims[l - 1]) * np.sqrt(
                1. / layer_dims[l - 1])
            W.append(coefficients)

            intercepts = np.zeros((layer_dims[l], 1))
            b.append(intercepts)

        self._W = W
        self._b = b

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
        cost_history = defaultdict(dict)

        n_x, m = X.shape
        n_y, _ = Y.shape
        if J is None:
            J = np.zeros((n_y, n_x, m))

        batch_size = self.batch_size
        if self.batch_size == 'auto':
            batch_size = min(200, m)

        for e in range(self.num_epochs):
            batches = mini_batches(X, batch_size, seed=self.random_state,
                                   shuffle=self.shuffle)
            for b, batch in enumerate(batches):

                X_batch = X[:, batch]
                Y_batch = Y[:, batch]
                J_batch = J[:, :, batch]

                def f(x, is_grad=True):
                    """ Cost function and gradient """
                    W = x[:len(self._W)]
                    b = x[-len(self._b):]
                    a = self._a
                    lambd = self.alpha
                    gamma = self.gamma

                    y, dW, db = cost(W, b, a, X_batch, Y_batch, J_batch,
                                     lambd, gamma, is_grad)
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

                cost_fun = f
                if self.is_grad_check:
                    cost_fun = f_check
                elif self.is_finite_difference:
                    cost_fun = f_FD

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

                params = optimizer.minimize(x=self._W + self._b, f=cost_fun,
                                            alpha=self.learning_rate_init,
                                            max_iter=self.max_iter,
                                            verbose=self.verbose,
                                            epoch=e, batch=b)

                self._W = params[:len(self._W)]
                self._b = params[-len(self._b):]

                key1 = 'epoch_' + str(e)
                key2 = 'batch_' + str(b)
                cost_history[key1][key2] = np.array(optimizer.cost_history)
        return cost_history

    def fit(self, X: np.ndarray, Y: np.ndarray, J: np.ndarray = None):
        """
        Minimize augmented squared loss in order to learn neural net parameters

        Parameters
        ----------
        X: np.ndarray
            Training data features
            shape = (m, n_x) where m = no. examples, n_x = no. features

        Y: np.ndarray
            Training data responses
            shape = (m, n_x) where n_y = no. responses

        J: np.ndarray
            Training data Jacobian, J = dY/dX
            shape = (m, n_x, n_y)
        """
        X, Y, J = self._preprocess(X, Y, J)

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

        self._cost_history = self._train(X, Y, J)

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
                shape = (m, n_x) where n_y = no. responses
        """

        if X.shape[1] != self._n_x:
            msg = f'X.shape = {X.shape}. Expected (-1, {self._n_x}).'
            raise ValueError(msg)

        if X.shape[1] != self._n_x:
            if X.shape == (self._n_x, 1):
                X = X.T
            else:
                msg = f'X.shape = {X.shape}. Expected (-1, {self._n_x}).'
                raise ValueError(msg)

        X_norm = (X.T - self._mu_x) / (self._sigma_x + EPS)
        Y_norm = L_model_forward(X_norm, self._W, self._b, self._a,
                                 store_cache=False)

        Y = (self._sigma_y + EPS) * Y_norm + self._mu_y
        return Y.T

    def gradient(self, X: np.ndarray):
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
                Training data Jacobian, J = dY/dX
                shape = (m, n_x, n_y)
        """
        if X.ndim != 2:
            if X.ndim == 1 and X.shape[0] == self._n_x:
                X = X.reshape((1, -1))
            else:
                msg = f'X.ndim = {X.ndim}. Expected 2.'
                raise ValueError(msg)

        if X.shape[1] != self._n_x:
            if X.shape == (self._n_x, 1):
                X = X.T
            else:
                msg = f'X.shape = {X.shape}. Expected (-1, {self._n_x}).'
                raise ValueError(msg)

        X_norm = (X.T - self._mu_x) / (self._sigma_x + EPS)
        J_norm = L_grads_forward(X_norm, self._W, self._b, self._a,
                                 store_cache=False)

        J = J_norm * self._sigma_y / self._sigma_x

        return J.T

    def goodness_fit(self, x: np.ndarray, y_true: np.ndarray,
                     show_plot: bool = True, title: str = None,
                     legend: str = None) -> float:
        y_pred = self.predict(x)
        if show_plot:
            goodness_of_fit(y_pred, y_true, title, legend)
        return rsquare(y_pred, y_true)

    def training_history(self, show_plot: bool = False):

        if not MATPLOTLIB_INSTALLED and show_plot:
            raise ImportError("Matplotlib must be installed.")

        if not self.training_history:
            return None

        if show_plot:
            epochs = list(self._cost_history.keys())
            if len(epochs) > 1:
                avg_cost = []
                for epoch in epochs:
                    batches = self._cost_history[epoch].keys()
                    costs = []
                    for batch in batches:
                        costs.append(self._cost_history[epoch][batch].mean())
                    avg_cost.extend(sum(costs) / len(batches))
                plt.plot(range(len(epochs)), avg_cost)
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

        return self._cost_history
