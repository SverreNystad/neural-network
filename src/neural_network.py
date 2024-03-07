import numpy as np

from src.activation import Activation


class HyperParameters:

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations


class Neuron:

    def __init__(self, activation: Activation, learning_rate: float = 0.01) -> None:
        self.activation = activation()
        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = 1000

        self.weights = None
        self.bias = None

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the neural network on the given input-output pairs
        This uses gradient descent with backpropagation to update the weights

        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, features).
            y (np.ndarray): Target values, a numpy array of shape (samples,).
        """

        # Initialize weights and bias
        samples, features = x.shape

        # Initialize weights
        self.weights = self._random_weights(features)
        self.bias = 0

        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = self.forward(x)

            # Compute loss
            dw = np.dot(x.T, (y - predictions))
            db = np.sum(y - predictions)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _random_weights(self, features: int) -> np.ndarray:
        """
        Returns a random weight matrix of shape (features, 1)

        Args:
            features (int): number of features in the input tensor
        Returns:
            np.ndarray: random weight matrix of shape (features, 1)
        """
        return np.random.rand(features, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass, predicting the output for the given input.
        Using the formula:
            a_j = g_j(Σ_i w_i,j a_i)
            a_j: the output of the unit j
            a_i: the output of the the units
            w_i,j: The weight attached to the link from unit i to unit j

            g_j: is the nonlinear activation function associated with unit j

        output = activation(W^T * input + bias)

        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, features).

        Returns:
            np.ndarray: The predicted output, a numpy array of shape (samples,).
        """
        y = np.dot(self.weights.T, x) + self.bias
        return self.activation(y)


class Layer:
    def __init__(self, neurons: int, activation: Activation) -> None:
        self.neurons = [Neuron(activation) for _ in range(neurons)]

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the layer on the given input-output pairs
        This uses gradient descent with backpropagation to update the weights

        Args:
            x (np.ndarray): input tensor to the layer
            y (np.ndarray): output tensor of the layer
        """
        for neuron in self.neurons:
            neuron.train(x, y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        Args:
            x (np.ndarray): input tensor to the layer
        Returns:
            np.ndarray: output tensor of the layer
        """
        return np.array([neuron.forward(x) for neuron in self.neurons])


class NeuralNetwork:

    def __init__(
        self, input_layer_size: int, hidden_layers: list[Layer], output_layer_size: int
    ) -> None:
        self.input_layer_size = input_layer_size
        self.hidden_layers = hidden_layers
        self.output_layer_size = output_layer_size

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the neural network using the provided input-output pairs.
        Using backpropagation and optimization algorithm to update the weights.

        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, input_layer_size).
            y (np.ndarray): Target values, a numpy array of shape (samples, output_layer_size).
        """
        if x.shape[1] != self.input_layer_size:
            raise ValueError(
                f"Input array must have {self.input_layer_size} features, but got {x.shape[1]}"
            )

        # Take the input tensor and forward it through each layer and store the activations
        # of each layer.
        activation = x
        activations = [x]
        for layer in self.hidden_layers:
            # Train the layer on the input-output pairs
            activation = layer.forward(activation)
            activations.append(x)

        # Backpropagation

        for layer in reversed(self.hidden_layers):
            # Backward
            pass
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input using the trained neural network.

        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, input_layer_size).

        Returns:
            np.ndarray: The predicted output, a numpy array of shape (samples, output_layer_size).
        """
        if x.shape[1] != self.input_layer_size:
            raise ValueError(
                f"Input array must have {self.input_layer_size} features, but got {x.shape[1]}"
            )

        for layer in self.hidden_layers:
            # Forward pass through the layer and update the input tensor
            # Before the next layer
            x = layer.forward(x)

        # The output tensor of the last hidden layer is the input tensor to the output layer
        assert x.shape[1] == self.output_layer_size
        return x
