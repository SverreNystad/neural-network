import numpy as np
import pytest
from src.data_loader import get_data
from src.losses import Loss, MeanSquaredError
from src.neural_network import Sigmoid, Activation, NeuralNetwork


def test_get_data():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    train_features = X_train.shape[1]
    test_features = X_test.shape[1]

    assert train_features == test_features


def test_activation_shapes():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    activation_function: Activation = Sigmoid()
    activated = activation_function(X_train)
    activated = activation_function(X_train)
    derived = activation_function.derivative(X_train)
    assert activated.shape == X_train.shape
    assert derived.shape == X_train.shape


def test_activation_derivative():
    x = np.array([0, 1])

    activation_function: Activation = Sigmoid()
    activated = activation_function(x)
    derived = activation_function.derivative(x)

    expected_derived = np.array([0.25, 0.19661193])
    assert np.isclose(derived, expected_derived).all()


def test_zero_loss():
    loss_function: Loss = MeanSquaredError()
    y_prediction = np.array([0, 1, 0, 1, 0])
    y_true = np.array([0, 1, 0, 1, 0])
    loss = loss_function(y_true, y_prediction)
    derived = loss_function.derived(y_true, y_prediction)
    expected_loss = 0
    expected_derived = np.array([0, 0, 0, 0, 0])

    assert np.isclose(loss, expected_loss)
    assert np.isclose(derived, expected_derived).all()


def test_some_loss():
    loss_function: Loss = MeanSquaredError()
    y_true = np.array([3.0, -0.5, 2.0, 7.0], dtype=float)
    y_prediction = np.array([2.5, 0.0, 2.0, 8.0], dtype=float)
    loss = loss_function(y_true, y_prediction)
    derived = loss_function.derived(y_true, y_prediction)
    expected_loss = 0.375
    expected_derived = np.array([-1.0, 1.0, 0.0, 2.0])

    assert np.isclose(loss, expected_loss)
    assert np.isclose(derived, expected_derived).all()
