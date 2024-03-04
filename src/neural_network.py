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
        return np.mean((y_true - y_pred)**2)
        

class Neuron:
    
    def __init__(self, activation: Activation, learning_rate: float = 0.01) -> None:
        self.activation = activation
        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = 1000

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