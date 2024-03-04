import pytest
from src.data_loader import get_data
from src.neural_network import Sigmoid, Activation, NeuralNetwork

def test_get_data():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    train_features = X_train.shape[1]
    test_features = X_test.shape[1]

    assert train_features == test_features


def test_activation():
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    activation_function: Activation = Sigmoid()
    activated = activation_function(X_train)
    assert activated.shape == X_train.shape

# def test_prediction():
#     X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
#     activation_function: Activation = Sigmoid()
    
#     nn: NeuralNetwork = NeuralNetwork(activation_function)
#     nn.train(X_train, y_train)

#     predictions = nn.forward(X_test)
#     assert predictions.shape == y_test.shape