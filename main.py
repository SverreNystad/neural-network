from src.data_loader import get_data
from src.neural_network import NeuralNetwork, Layer, Sigmoid, Neuron
import numpy as np

from src.visualizer import neuron_visualizer

if __name__ == "__main__":
    # np.random.seed(0)
    # X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # # Train the model
    # NeuralNetwork = NeuralNetwork(2, [Layer(1, Sigmoid)], 1)
    # NeuralNetwork.train(X_train, y_train)

    # # Make predictions
    # predictions = NeuralNetwork.predict(X_test)

    Neuron = Neuron(Sigmoid)
    neuron_visualizer(Neuron)
