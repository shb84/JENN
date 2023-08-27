import numpy as np
from ._activation import ACTIVATIONS
from ._parameters import Parameters
from ._cache import Cache
from ._fwd_prop import L_model_forward, L_grads_forward
from .._utils import mini_batches
from collections import defaultdict


class JENNBase:

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 *,
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
                 ):

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
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.solver = solver
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.num_epochs = num_epochs
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # kwargs
        self.random_state = random_state

        # Dimensionality of inputs and outputs
        self._n_x = None
        self._n_y = None

        # Parameters
        self.parameters = None

        # Training history
        self.cost_history = None

    def _train(self, X: np.ndarray, Y: np.ndarray, J: np.ndarray = None):

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f'X should be an array of two-dimensional shape')

        if Y.ndim != 2:
            raise ValueError(f'Y should be an array of two-dimensional shape')

        # Get training data shapes
        n_x, m = X.shape
        n_y, _ = Y.shape
        self._n_x = n_x
        self._n_y = n_y 
        
        # Get neural layer sizes and shapes 
        layer_sizes = [n_x] + self.hidden_layer_sizes + [n_y]
        layer_shapes = [(layer_size, m) for layer_size in layer_sizes]

        # Initialize
        self.parameters = Parameters(layer_sizes)
        cache = Cache(layer_shapes)
        cost_history = defaultdict(dict)

        if J is None:
            J = np.zeros((n_y, n_x, m))
            self.gamma = 0  # Make sure gradient-enhancement is off if J = None

        for e in range(self.num_epochs):
            batches = mini_batches(
                X, self.batch_size, self.shuffle, self.random_state)
            for b, batch in enumerate(batches):

                X_batch = X[:, batch]
                Y_batch = Y[:, batch]
                J_batch = J[:, :, batch]

        #         def f(x, is_grad=True):
        #             """ Cost function and gradient """
        #             W = x[:len(self._W)]
        #             b = x[len(self._W):]
        #             y, dW, db = self._cost(W, b, X_batch, Y_batch, J_batch,
        #                                    is_grad)
        #             dy_dx = dW + db
        #             return y, dy_dx
        #
        #         def f_FD(x):
        #             """ Debug: cost function with finite difference grad """
        #             y = f(x, is_grad=False)[0]
        #             dy_dx = finite_diff(x, lambda x: f(x, is_grad=False)[0])
        #             return y, dy_dx
        #
        #         def f_check(x):
        #             """ Debug: cost function with FD vs. analytic grad """
        #             grad_check(x, f=lambda x: f(x, is_grad=False)[0],
        #                        dfdx=lambda x: f(x, is_grad=True)[1])
        #             if self.is_finite_difference:
        #                 return f_FD(x)
        #             else:
        #                 return f(x)
        #
        #         cost = f
        #         if self.is_grad_check:
        #             cost = f_check
        #         elif self.is_finite_difference:
        #             cost = f_FD
        #
        #         if self.solver == 'adam':
        #             update = ADAM(self.beta_1, self.beta_2)
        #         elif self.solver == 'sgd':
        #             update = GD()
        #         else:
        #             raise ValueError(f'solver = {self.solver} not recognized')
        #
        #         if self.learning_rate == 'constant':
        #             line_search = Backtracking(update,
        #                                        max_count=0, tol=self.tol)
        #         elif self.learning_rate == 'backtracking':
        #             line_search = Backtracking(update, tol=self.tol)
        #         else:
        #             raise ValueError(f'learning_rate = '
        #                              f'{self.learning_rate} not recognized')
        #
        #         optimizer = Optimizer(line_search)
        #
        #         params = optimizer.minimize(x=self._W + self._b, f=cost,
        #                                     alpha=self.learning_rate_init,
        #                                     max_iter=self.max_iter,
        #                                     verbose=self.verbose,
        #                                     epoch=e, batch=b)
        #
        #         self._W = params[:len(self._W)]
        #         self._b = params[len(self._W):]
        #
        #         key1 = 'epoch_' + str(e)
        #         key2 = 'batch_' + str(b)
        #         cost_history[key1][key2] = np.array(optimizer.cost_history)
        #
        # self.cost_history = cost_history
