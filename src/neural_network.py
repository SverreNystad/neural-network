import numpy as np

from src.activation import Activation, Sigmoid
from src.losses import Loss, MeanSquaredError


class Neuron:

    def __init__(
        self,
        weights: int,
        activation: Activation,
        cost_function: Loss,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Initializes the neuron with the given activation function and cost function

        """
        self.activation = activation()
        self.cost_function = cost_function()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = 1000

        self.weights: np.ndarray = self._random_weights(weights)
        self.bias: float = 0

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
        return self.activation(self.get_input(x))

    def get_input(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the input to the neuron

        This is often noted as z = W^T * x + b
        This input z is sometimes called the im
        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, features).

        Returns:
            np.ndarray: The input to the neuron, a numpy array of shape (samples,).
        """
        return np.dot(self.weights.T, x) + self.bias

    def update_weights(self, delta: np.ndarray, x: np.ndarray) -> None:
        """
        Updates the weights of the neuron using the given input and delta

        Args:
            delta (np.ndarray): The gradient of the loss function with respect to the output layer
            x (np.ndarray): The input to the neuron
        """
        self.bias -= self.learning_rate * np.sum(delta)
        if x.shape[0] != delta.shape[0]:
            # The input tensor must have the same number of features as the weights
            return
        self.weights -= self.learning_rate * np.dot(x.T, delta)


class Layer:
    def __init__(
        self, inputs: int, neurons: int, activation: Activation, cost_function: Loss
    ) -> None:
        self.neurons = [
            Neuron(inputs, activation, cost_function) for _ in range(neurons)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagates pass through the layer
        Args:
            x (np.ndarray): input tensor to the layer
        Returns:
            np.ndarray: output tensor of the layer
        """
        return np.array([neuron.forward(x) for neuron in self.neurons])

    def get_inputs(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the input to the layer

        Args:
            x (np.ndarray): Input features, a numpy array of shape (samples, features).

        Returns:
            np.ndarray: The input to the layer, a numpy array of shape (samples,).
        """
        return np.array([neuron.get_input(x) for neuron in self.neurons])

    def update_weights(self, delta: np.ndarray, x: np.ndarray) -> None:
        """
        Updates the weights of the layer using the given input and delta

        Args:
            delta (np.ndarray): The gradient of the loss function with respect to the output layer
            x (np.ndarray): The input to the layer
        """
        for neuron, delta_i in zip(self.neurons, delta):
            neuron.update_weights(delta_i, x)


class NeuralNetwork:

    def __init__(
        self,
        input_layer_size: int,
        hidden_layers: list[int],
        output_layer_size: int,
        activation: Activation = Sigmoid,
        cost_function: Loss = MeanSquaredError,
    ) -> None:
        self.activation: Activation = activation
        self.cost_function: Loss = cost_function
        self.learning_rate = 0.01

        self.input_layer_size = input_layer_size
        self._init_layers(input_layer_size, hidden_layers, output_layer_size)

    def _init_layers(
        self, input_layer_size: int, hidden_layers: list[int], output_layer_size: int
    ) -> None:
        """
        Initializes the layers of the neural network

        Args:
            input_layer_size (int): number of features in the input tensor
            hidden_layers (list[int]): number of neurons in each hidden layer
            output_layer_size (int): number of neurons in the output layer
        """
        self.hidden_layers = []
        inputs = input_layer_size
        for layer in hidden_layers:
            self.hidden_layers.append(
                Layer(inputs, layer, self.activation, self.cost_function)
            )
            inputs = layer
        self.output_layer = Layer(
            hidden_layers[-1], output_layer_size, self.activation, self.cost_function
        )

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
        for x_i, y_i in zip(x, y):
            self._train_single(x_i, y_i)

    def _train_single(self, x: np.ndarray, y: np.ndarray) -> None:

        # Take the input tensor and forward it through each layer and store the activations
        # of each layer.
        activation = x
        activations = [x]
        # outputs are the activations of each layer z = W^T * x + b
        for layer in self.hidden_layers:
            # Train the layer on the input-output pairs
            activation = layer.forward(activation)
            activations.append(activation)

        # Forward pass through the output layer
        activation = self.output_layer.forward(activation)
        activations.append(activation)

        # TODO: Might use for logging
        total_error = self.cost_function()(y, activations[-1])
        # Backwards pass
        # Compute the gradient of the loss function with respect to the output layer

        delta = (
            activations[-2]
            * self.activation().derivative(activations[-1])
            * 2
            * (activations[-1] - y)
        )

        self.output_layer.update_weights(delta, activations[-2])
        # Update the weights and biases of the output layer
        for i in range(len(self.hidden_layers), 0, -1):
            delta = (
                self.hidden_layers[i - 1].get_inputs(activations[i - 1])
                * self.activation().derivative(activations[i])
                * 2
                * (activations[i] - y)
            )
            self.hidden_layers[i - 1].update_weights(delta, activations[i - 1])

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
        predictions = np.zeros((x.shape[0], len(self.output_layer.neurons)))
        for i, x_i in enumerate(x):
            prediction = self._predict_single(x_i)
            predictions[i] = prediction
        return predictions

    def _predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input using the trained neural network.
        """
        activation = x
        for layer in self.hidden_layers:
            # Forward pass through the layer and update the input tensor
            # Before the next layer
            activation = layer.forward(activation)

        # Forward pass through the output layer
        activation = self.output_layer.forward(activation)

        return activation
