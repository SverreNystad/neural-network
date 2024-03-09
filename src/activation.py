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

    def derivative(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the derivative of the sigmoid function

        f = 1/(1+exp(-x))
        df = f * (1 - f)
        Args:
            x (float or array): input to the sigmoid function
                the function is vectorized, so it is acceptable
                to pass an array of any shape.

        Returns:
            Element-wise derivative of the sigmoid function
        """
        return self(x) * (1 - self(x))


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function
    """

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return np, max(0, x)

    def derivative(self, x: float | np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class Softplus(Activation):
    """
    Softplus activation function

    Softplus is a smooth version of the ReLU function
    """

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return np.log(1 + np.e ** (x))


class Tanh(Activation):
    """
    Hyperbolic tangent function
    Tanh is a scaled and shifted version of the Sigmoid activation
    Note that range of tanh is [-1, 1]
    """

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        numerator = np.e ** (2 * x) - 1
        denominator = np.e ** (2 * x) + 1
        return numerator / denominator
