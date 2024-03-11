from src.data_loader import get_data
from src.neural_network import Layer, NeuralNetwork, Sigmoid
import numpy as np

np.random.seed(0)


def test_2_1_1_network_output_shape():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    layers = [1]
    network = NeuralNetwork(2, layers, 1)
    network.train(X_train, y_train)
    y_pred = network.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]


def test_too_many_input_features():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    layers = [1]
    network = NeuralNetwork(3, layers, 1)
    try:
        network.predict(X_train)
    except ValueError:
        assert True
    else:
        assert False


def test_too_few_input_features():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    layers = [1]
    network = NeuralNetwork(1, layers, 1)
    try:
        network.predict(X_train)
    except ValueError:
        assert True
    else:
        assert False


def test_predict_output_shape():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    assert X_test.shape == (120, 2)
    layers = [1]
    network = NeuralNetwork(2, layers, 1)
    network.train(X_train, y_train)
    y_pred = network.predict(X_test)
    assert y_pred.shape == (120, 1)
