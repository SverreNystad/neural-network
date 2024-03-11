from matplotlib import pyplot as plt
from src.neural_network import Neuron
import numpy as np
import math


def visualize_neuron_weights(
    neuron: Neuron,
) -> None:
    """
    Visualizes the weights of a neuron as an image.
    Args:
        neuron (Neuron): Neuron to visualize with weights and bias
    """

    # Get the weights of the neuron
    weights = neuron.weights

    # Calculate the maximum absolute weight to normalize the weights to [0, 255]
    max_abs_weight = np.max(np.abs(weights))

    # Normalize the weights to [0, 1] and then to [0, 255]
    normalized_weights = (weights / max_abs_weight * 0.5 + 0.5) * 255

    # Find the dimensions of the rectangle for visualization
    num_weights = len(weights)
    width = int(math.ceil(math.sqrt(num_weights)))
    height = int(math.ceil(num_weights / width))

    # If the number of weights is not a perfect multiple of the width,
    # we need to pad the weights array to fit into the rectangle
    padded_weights = np.pad(
        normalized_weights,
        (0, width * height - num_weights),
        "constant",
        constant_values=0,
    )

    weight_matrix = padded_weights.reshape(height, width)

    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(weight_matrix, cmap="gray", interpolation="nearest")
    ax.axis("off")  # Hide the axes
    plt.show()


def plot_loss_over_time(losses: np.ndarray) -> None:
    """
    Plots the loss over time
    Args:
        losses (np.ndarray): Array of loss values
    """
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.show()
