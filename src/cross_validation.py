from src.neural_network import Loss, MeanSquaredError, NeuralNetwork
import numpy as np


def cross_validate(
    model: NeuralNetwork,
    k: int,
    dataset: np.ndarray,
    loss_algorithm: Loss = MeanSquaredError(),
) -> list[np.ndarray]:
    """
    Trains and validates the model using k-fold cross validation
    """
    # 1. Split the data set into K subsets randomly
    # 2. For each one of the developed subsets of data points
    #   * Treat that subset as the validation set
    #   * Use all the rest subsets for training purpose
    #   * Training of the model and evaluate it on the validation set or test set
    #   * Calculate prediction error
    # 3. Repeat the above step K times i.e., until the model is not trained and tested on all subsets
    # 4. Generate overall prediction error by taking the average of prediction errors in every case
    data_splits = split_into_k_train_test_sets(k, dataset)
    errors = []

    for train, test in data_splits:
        model.train()
        predictions = model.predict(test)
        error = loss_algorithm(test, predictions)
        errors.append(error)

    return errors


def split_into_k_train_test_sets(
    k: int, dataset: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Splits the dataset into k folds and returns a list of tuples where each tuple
    contains the training and validation sets for a fold

    Args:
        k (int): number of folds
    Returns:
        list[tuple[np.ndarray, np.ndarray]]: list of tuples where each tuple contains
            the training and validation sets for a fold
    """
    pass
