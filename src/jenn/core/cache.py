"""Neural net cache.

Store quantities computed during forward propagation, so they don't have
to be recomputed during backward propgation.
"""

import numpy as np


class Cache:
    """Neural net cache.

    This object stores neural net quantities computed during
    forward prop for each layer so they don't have to be
    recomputed during backprop. This allows the algorithm to be faster.

    Note that the attributes of this class are not protected. It's possible
    to overwrite them instead of updating them in place. To
    ensure that an array is updated in place, use proper numpy syntax:

        e.g. cache = Cache(shapes)
             layer_1_activations = cache.A[1]
             layer_1_activations[:] = new_array_values  # note [:]

    Parameters
    ----------
    layer_sizes: list[int]
        The number of "neurons" in each layer (including input and output layers).

    m: int, optional
        The number of examples (for array preallocation). Default is 1.

    Attributes
    ----------
    Z: list[np.ndarray]
        Store Z = W.T A_prev + b for each layer

    Z_prime: list[np.ndarray]
        Store Z' = d/dx[j] (Z) for all j = 1, .., n_x

    Z_prime_prime: list[np.ndarray]
        Store Z'' = d/dx[j] d/dx[j] (Z))

    A: list[np.ndarray]
        Store A = g(Z) where g is the activation function.

    A_prime: list[np.ndarray]
        Store A' = d/dx[j] (A) for all j = 1, .., n_x

    A_prime_prime: list[np.ndarray]
        Store A'' = d/dx[j] d/dx[j] ( A )

    G_prime: list[np.ndarray]
        Store g' = d/dz (A)

    G_prime_prime: list[np.ndarray]
        Store g'' = d/dz d/dz (A)

    dA: list[np.ndarray]
        Store d/dA (L) for backprop

    dA_prime: list[np.ndarray]
        Store d/dA' (L) for backprop
    """

    @property
    def m(self) -> int:
        """Return number of examples."""
        return self.A[0].shape[1]

    @property
    def n_x(self) -> int:
        """Return number of inputs."""
        return self.layer_sizes[0]

    @property
    def n_y(self) -> int:
        """Return number of outputs."""
        return self.layer_sizes[-1]

    @property
    def J(self) -> np.ndarray:
        """Return predicted partials."""
        return self.A_prime[-1]

    def __init__(self, layer_sizes: list[int], m: int = 1):  # noqa: D107
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
        self.dA_prime = []
        for n in self.layer_sizes:
            self.Z.append(np.zeros((n, m)))
            self.Z_prime.append(np.zeros((n, self.n_x, m)))
            self.Z_prime_prime.append(np.zeros((n, self.n_x, m)))
            self.G_prime.append(np.zeros((n, m)))
            self.G_prime_prime.append(np.zeros((n, m)))
            self.A.append(np.zeros((n, m)))
            self.A_prime.append(np.zeros((n, self.n_x, m)))
            self.A_prime_prime.append(np.zeros((n, self.n_x, m)))
            self.dA.append(np.zeros((n, m)))
            self.dA_prime.append(np.zeros((n, self.n_x, m)))
