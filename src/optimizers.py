import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    The optimizer is responsible for updating the weights of the neural network
    """

    @abstractmethod
    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Updates the model's weights based on the gradients.

        Parameters:
            weights (np.ndarray): Current weights of the neural network.
            gradients (np.ndarray): Gradients of the loss function with respect to the weights.

        Returns:
            np.ndarray: Updated weights.
        """
        pass


class GradientDescent(Optimizer):

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return weights - gradients
