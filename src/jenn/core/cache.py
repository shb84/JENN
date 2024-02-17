"""Cache.
=========

This module defines a convenience class to  all quantities 
computed during forward propagation, so they don't have to be 
recomputed again during backward propgation. See
`paper`_ for details and notation.
"""  # noqa: W291

import numpy as np


class Cache:
    r"""Neural net cache.

    A cache s neural net quantities computed during
    forward prop for each layer, so they don't have to be
    recomputed again during backprop. This makes the algorithm faster.

    .. warning::
        The attributes of this class are not protected. It's possible
        to overwrite them instead of updating them in place. To ensure
        that an array is updated in place, use the numpy `[:]` syntax:

        .. code-block:: python

            cache = Cache(shapes)
            layer_1_activations = cache.A[1]
            layer_1_activations[:] = new_array_values  # note [:]

    .. note::
        The variables and their symbols refer to the theory in the companion
        `paper`_ for this library.

    :param layer_sizes: number of nodes in each layer (including input/output layers)
    :param m: number of examples (used to preallocate arrays)

    :ivar Z:  :math:`Z^{[l]} \in \mathbb{R}^{n^{[l]}\times m}~\forall~ l = 1 \dots L`
    :vartype Z: list[numpy.ndarray]

    :ivar Z_prime:  :math:`{Z^\prime}^{[l]} \in \mathbb{R}^{n^{[l]}\times n_x \times m}~\forall~ l = 1 \dots L`
    :vartype Z_prime: list[numpy.ndarray]

    :ivar A:  :math:`A^{[l]} = g(Z^{[l]}) \in \mathbb{R}^{n^{[l]} \times m}~\forall~ l = 1 \dots L`
    :vartype A: list[numpy.ndarray]

    :ivar A_prime:  :math:`{A^\prime}^{[l]} = g^\prime(Z^{[l]})Z^{\prime[l]} \in \mathbb{R}^{n^{[l]}\times n_x \times m}`
    :vartype A_prime: list[numpy.ndarray]

    :ivar G_prime:  :math:`G^{\prime} = g^{\prime}(Z^{[l]}) \in \mathbb{R}^{n^{[l]} \times m}~\forall~ l = 1 \dots L`
    :vartype G_prime: list[numpy.ndarray]

    :ivar G_prime_prime:  :math:`G^{\prime\prime} = g^{\prime\prime}(Z^{[l]}) \in \mathbb{R}^{n^{[l]} \times m}`
    :vartype G_prime_prime: list[numpy.ndarray]

    :ivar dA: :math:`{\partial \mathcal{J}}/{dA^{[l]}}  \in \mathbb{R}^{n^{[l]} \times m}~\forall~ l = 1 \dots L`
    :vartype dA: list[numpy.ndarray]

    :ivar dA_prime: :math:`{\partial \mathcal{J}}/{dA^{\prime[l]}}  \in \mathbb{R}^{n^{[l]} \times n_x \times m}~\forall~ l = 1 \dots L`
    :vartype dA: list[numpy.ndarray]
    """

    @property
    def m(self) -> int:
        """Return number of examples."""
        return int(self.A[0].shape[1])

    @property
    def n_x(self) -> int:
        """Return number of inputs."""
        return int(self.layer_sizes[0])

    @property
    def n_y(self) -> int:
        """Return number of outputs."""
        return int(self.layer_sizes[-1])

    def __init__(self, layer_sizes: list[int], m: int = 1):  # noqa: D107
        self.layer_sizes = layer_sizes
        self.Z: list[np.ndarray] = []  #  z = w a_prev + b
        self.Z_prime: list[np.ndarray] = []  #  z' = dz/dx[j] for all j = 1, .., n_x
        self.A: list[np.ndarray] = []  #  a = g(z)
        self.A_prime: list[np.ndarray] = []  #  a' = da/dx[j] for all j = 1, .., n_x
        self.G_prime: list[np.ndarray] = []  #  g' = da/dz
        self.G_prime_prime: list[np.ndarray] = []  #  g'' = d/dz( da/dz )
        self.dA: list[np.ndarray] = []
        self.dA_prime: list[np.ndarray] = []
        for n in self.layer_sizes:
            self.Z.append(np.zeros((n, m)))
            self.Z_prime.append(np.zeros((n, self.n_x, m)))
            self.G_prime.append(np.zeros((n, m)))
            self.G_prime_prime.append(np.zeros((n, m)))
            self.A.append(np.zeros((n, m)))
            self.A_prime.append(np.zeros((n, self.n_x, m)))
            self.dA.append(np.zeros((n, m)))
            self.dA_prime.append(np.zeros((n, self.n_x, m)))
