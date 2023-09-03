import numpy as np
import math


def mini_batches(
        X: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        random_state: int = None,
) -> list:
    """
    Create randomized mini-batches by returning a list of tuples, where
    each tuple contains the indices of the training data points associated with
    that mini-batch

    Parameters
    ----------
        X: np.ndarray
            input features of the training data
            shape = (n_x, m) where m = num of examples and n_x = num of inputs

        batch_size: int
            mini batch size (if None, then batch_size = m)

        shuffle: bool
            Shuffle data points
            Default = True

        random_state: int
            Random seed (set to make runs repeatable)
            Default = None

    Returns
    -------
        mini_batches: list
            List of mini-batch indices to use for slicing data, where the index
            is in the interval [1, m]
    """
    np.random.seed(random_state)

    batches = []
    m = X.shape[1]
    if not batch_size:
        batch_size = m
    batch_size = min(batch_size, m)

    # Step 1: Shuffle the indices
    if shuffle:
        indices = list(np.random.permutation(m))
    else:
        indices = np.arange(m)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / batch_size))
    k = 0
    for _ in range(num_complete_minibatches):
        mini_batch = indices[k * batch_size:(k + 1) * batch_size]
        if mini_batch:
            batches.append(tuple(mini_batch))
        k += 1

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch = indices[(k + 1) * batch_size:]
        if mini_batch:
            batches.append(tuple(mini_batch))

    return batches
