"""Test fixtures"""
import pytest
import jenn
import numpy as np
from typing import Tuple


@pytest.fixture
def xor() -> Tuple[jenn.Dataset, jenn.Parameters]:
    """Return XOR test data and model parameters"""
    data = jenn.Dataset(
        X=np.array([[0, 1, 0, 1], [0, 0, 1, 1]]),
        Y=np.array([[0, 1, 1, 0]]),
    )
    layer_sizes = [2, 2, 1]
    parameters = jenn.Parameters(layer_sizes, output_activation='relu')
    parameters.b[1][:] = np.array([[0], [-1]])        # layer 1
    parameters.W[1][:] = np.array([[1, 1], [1, 1]])   # layer 1
    parameters.b[2][:] = np.array([[0]])              # layer 2
    parameters.W[2][:] = np.array([[1, -2]])          # layer 2
    return data, parameters


@pytest.fixture
def sinusoidal_data_1D(
        m_train=4, m_test=30, lb=-np.pi, ub=np.pi,
) -> Tuple[jenn.Dataset, jenn.Dataset]:
    """Generate 1D sinusoidal synthetic data."""

    def f(x):
        return x * np.sin(x)

    def dfdx(x):
        return np.sin(x) + x * np.cos(x)

    # Domain
    lb = -np.pi
    ub = np.pi

    # Training data
    m = m_train  # number of training examples
    n_x = 1  # number of inputs
    n_y = 1  # number of outputs
    X_train = np.linspace(lb, ub, m).reshape((n_x, m))
    Y_train = f(X_train).reshape((n_y, m))
    J_train = dfdx(X_train).reshape((n_y, n_x, m))

    # Test data
    m = m_test  # number of test examples
    X_test = lb + np.random.rand(m, 1).reshape((n_x, m)) * (ub - lb)
    Y_test = f(X_test).reshape((n_y, m))
    J_test = dfdx(X_test).reshape((n_y, n_x, m))

    training_data = jenn.Dataset(X_train, Y_train, J_train)
    test_data = jenn.Dataset(X_test, Y_test, J_test)

    return training_data, test_data

