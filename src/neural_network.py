import numpy as np
from typing import Union
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Applies the logistic function element-wise

        Args:
            x (float or array): input to the logistic function
                the function is vectorized, so it is acceptable
                to pass an array of any shape.

        Returns:
            Element-wise sigmoid activations of the input
        """
        return 1 / (1 + np.exp(-x))

class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function
    """
    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return np,max(0, x)

class Softplus(Activation):
    """
    Softplus activation function

    Softplus is a smooth version of the ReLU function
    """

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return np.log(1 + np.e**(x))
    
class Tanh(Activation):
    """
    Hyperbolic tangent function
    Tanh is a scaled and shifted version of the Sigmoid activation
    Note that range of tanh is [-1, 1]
    """
    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        numerator = np.e**(2*x)-1
        denominator = np.e**(2*x)+1
        return numerator / denominator
    
class Loss(ABC):
    """
    The loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class MeanSquaredError(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the mean squared error

        Formula:
        mse = (1/n) * Σ(y_true - y_pred)^2

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels

        Returns:
            np.ndarray: mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)


class HyperParameters:

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations


class Neuron:

    def __init__(self, activation: Activation, learning_rate: float = 0.01) -> None:
        self.activation = activation
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
            x (np.ndarray): input tensor to the neural network
            y (np.ndarray): output tensor of the neural network
        """

        # Initialize weights and bias
        samples: int = x.shape[0]
        features: int = x.shape[1]

        # Initialize weights
        self.weights = self._random_weights(features)
        self.bias = 0

        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = self.forward(x)

            # Compute loss
            dw = np.dot(x.T, (y - predictions))
            db = np.sum(y - predictions)

        pass

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
        Predicts the output of the neural network for a given input
        Args:
            x (np.ndarray): input to the neural network
        Returns:
            np.ndarray: output of the neural network
        """
        y = np.dot(x, self.weights) + self.bias
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
        if x.shape[1] == self.input_layer_size:
            raise ValueError(
                f"Input array must have {self.input_layer_size} features, but got {x.shape[1]}"
            )
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
