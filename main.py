from src.activation import Activation, Sigmoid
from src.data_loader import get_data
from src.losses import Loss, MeanSquaredError
from src.neural_network import NeuralNetwork
import numpy as np

from src.visualizer import plot_loss_over_time, visualize_neuron_weights

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    activation: Activation = Sigmoid
    loss_function: Loss = MeanSquaredError

    # Train the model
    print("[INFO] Creating the model")
    neuralNetwork = NeuralNetwork(
        2, [2], 1, activation, loss_function, learning_rate=0.01, epochs=100
    )
    print(neuralNetwork.summary())
    print(f"[INFO] Training the model for {neuralNetwork.epochs} epochs")
    neuralNetwork.train(X_train, y_train)

    # Visualize the the loss over time
    plot_loss_over_time(neuralNetwork.logger)

    # Visualize the neurons classifying neuron
    visualize_neuron_weights(neuralNetwork.output_layer.neurons[0])

    # Make predictions on the training set
    predictions = neuralNetwork.predict(X_train)
    loss = loss_function()(y_train, predictions)
    print(f"MSE Loss on Training: {loss}")

    # Make predictions on the test set
    predictions = neuralNetwork.predict(X_test)
    loss = loss_function()(y_test, predictions)
    print(f"MSE Loss on test: {loss}")
