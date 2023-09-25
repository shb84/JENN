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
        return self.layer_sizes[0]

    @property
    def n_y(self):
        return self.layer_sizes[-1]

    @property
    def J(self):
        return self.A_prime[-1]

    def __init__(self, layer_sizes, m=1):
        self.layer_sizes = layer_sizes
        self.Z = []  # store z = w a_prev + b
        self.Z_prime = []  # store z' = dz/dx[j] for all j = 1, .., n_x
        self.Z_prime_prime = []  # store z'' = d/dx[j] ( dz/dx[j] )
        self.A = []  # store a = g(z)
        self.A_prime = []  # store a' = da/dx[j] for all j = 1, .., n_x
        self.A_prime_prime = []  # store a'' = d/dx[j] ( da/dx[j] )
        self.G_prime = []  # store g' = da/dz
        self.G_prime_prime = []  # store g'' = d/dz( da/dz )
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
            self.dA.append(np.zeros((n, m)))
