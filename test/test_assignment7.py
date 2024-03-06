from src.data_loader import get_data
from src.neural_network import Layer, NeuralNetwork, Sigmoid
import numpy as np

np.random.seed(0)


def test_2_1_1_network_output_shape():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    layers = [Layer(1, Sigmoid)]
    network = NeuralNetwork(2, layers, 1)
    network.train(X_train, y_train)
    y_pred = network.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_too_high_input_num():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    layers = [Layer(1, Sigmoid)]
    network = NeuralNetwork(3, layers, 1)
    try:
        network.predict(X_train)
    except ValueError:
        assert True
    else:
        assert False
