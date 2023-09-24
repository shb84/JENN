"""Test that model learns correctly."""
import pytest
import jenn
import numpy as np


def test_sinuisoid(sinusoidal_data_1D):
    """Train a neural net against 1D sinuidal data."""
    training_data, test_data = sinusoidal_data_1D