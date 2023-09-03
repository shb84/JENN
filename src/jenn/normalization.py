import numpy as np


def avg(array):
    return np.mean(array, axis=1).reshape((-1, 1))


def std(array):
    return np.std(array, axis=1).reshape((-1, 1))


def safe_divide(value, eps=np.finfo(float).eps):
    if value == 0.0:
        return value + eps
    return value


def normalize(array, mu, sigma):
    return (array - mu) / safe_divide(sigma)


def denormalize(array, mu, sigma):
    return safe_divide(sigma) * array + mu


def normalize_partials(partials, sigma_x, sigma_y):
    if partials is None:
        return partials
    n_y, n_x, _ = partials.shape
    sigma_x = sigma_x.T.reshape((1, n_x, 1))
    sigma_y = sigma_y.reshape((n_y, 1, 1))
    return partials * safe_divide(sigma_x) / safe_divide(sigma_y)


def denormalize_partials(partials, sigma_x, sigma_y):
    n_y, n_x, _ = partials.shape
    sigma_x = sigma_x.T.reshape((1, n_x, 1))
    sigma_y = sigma_y.reshape((n_y, 1, 1))
    return partials * safe_divide(sigma_y) / safe_divide(sigma_x)
