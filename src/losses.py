import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    The loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss between the true labels and the predicted labels
        """
        pass

    @abstractmethod
    def derived(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss function
        """
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

    def derived(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred)


class RootMeanSquaredError(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the root mean squared error

        Formula:
        rmse = sqrt((1/n) * Σ(y_true - y_pred)^2)

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels

        Returns:
            np.ndarray: root mean squared error
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsoluteError(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the mean absolute error

        Formula:
        mae = (1/n) * Σ|y_true - y_pred|

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels

        Returns:
            np.ndarray: mean absolute error
        """
        return np.mean(np.abs(y_true - y_pred))


class CrossEntropy(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the cross entropy loss

        Formula:
        cross_entropy = -Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels

        Returns:
            np.ndarray: cross entropy loss
        """
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
