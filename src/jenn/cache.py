"""Array initialization."""
import numpy as np


class Cache:
    """
    Don't recompute quantities during training. This class stores previously
    computed quantities during forward prop in pre-allocated arrays that can
    be accessed during backprop. This allows the algorithm to be faster.

    Note that the attributes of this class are not protected. It's possible
    to overwrite them instead of updating them in place. To
    ensure that an array is updated in place, please use numpy syntax for it:

        e.g. cache = Cache(shapes)
             layer_1_activations = cache.A[1]
             layer_1_activations[:] = new_array_values  # note [:]
    """

    @property
    def m(self):
        return self.A[0].shape[1]

    @property
    def n_x(self):
        return self._layer_sizes[0]

    @property
    def n_y(self):
        return self._layer_sizes[-1]

    @property
    def layer_sizes(self):
        return [n for n in self._layer_sizes]

    def __init__(self, layer_sizes, m=1):
        self._layer_sizes = layer_sizes
        self.vectorize(m)

    def vectorize(self, m: int):
        """Vectorize preallocated neural net arrays

        Parameters
        ----------
        number_examples: int, optional
            Number of examples to use when preallocating arrays
            for vectorization. Default is 1.
        """
        self.Z = []  # store z = w a_prev + b
        self.Z_prime = []  # store z' = dz/dx[j] for all j = 1, .., n_x
        self.Z_prime_prime = []  # store z'' = d/dx[j] ( dz/dx[j] )
        self.A = []  # store a = g(z)
        self.A_prime = []  # store a' = da/dx[j] for all j = 1, .., n_x
        self.A_prime_prime = []  # store a'' = d/dx[j] ( da/dx[j] )
        self.G_prime = []  # store g' = da/dz
        self.G_prime_prime = []  # store g'' = d/dz( da/dz )
        self.J = []  # Jacobian w.r.t. x for each layer
        self.dA = []
        for i, n in enumerate(self.layer_sizes):
            self.Z.append(np.zeros((n, m)))
            self.Z_prime.append(np.zeros((n, self.n_x, m)))
            self.Z_prime_prime.append(np.zeros((n, self.n_x, m)))
            self.G_prime.append(np.zeros((n, m)))
            self.G_prime_prime.append(np.zeros((n, m)))
            self.A.append(np.zeros((n, m)))
            self.A_prime.append(np.zeros((n, self.n_x, m)))
            self.A_prime_prime.append(np.zeros((n, self.n_x, m)))
            self.J.append(np.zeros((n, self.n_x, m)))
            self.dA.append(np.zeros((n, m)))

    def __call__(self, X: np.ndarray):
        """Update cache upon new input if shape has changed."""
        if X.shape != self.A[0].shape:
            n_x = X.shape[0]
            if self.n_x != n_x:
                raise ValueError(
                    f'Neural net expected input array shape to be ({self.n_x}, *), not ({n_x}, *)')
            self.vectorize(X.shape[1])
        return self
