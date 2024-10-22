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

    def update_parameters(self, delta_weighs: np.ndarray, delta_bias: int) -> None:
        """
        Updates the weights of the neuron using the given input and delta

        Args:
            delta (np.ndarray): The gradient of the loss function with respect to the output layer
            x (np.ndarray): The input to the neuron
        """
        self.weights -= self.learning_rate * delta_weighs
        self.bias -= self.learning_rate * delta_bias


class Layer:
    def __init__(
        self,
        inputs: int,
        neurons: int,
        activation: Activation,
        cost_function: Loss,
        learning_rate: float,
    ) -> None:
        self._inputs = inputs
        self.neurons = [
            Neuron(inputs, activation, cost_function, learning_rate)
            for _ in range(neurons)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagates pass through the layer
        Args:
            x (np.ndarray): input tensor to the layer
        Returns:
            np.ndarray: output tensor of the layer
        """
        if x.shape[0] != self._inputs:
            # Correct number of features
            # (4, 1) != (4, 1, 1)
            x = x.reshape(self._inputs, 1)
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

    def update_parameters(
        self,
        delta_weights: np.ndarray,
        delta_biases: np.ndarray,
    ) -> None:
        """
        Updates the weights of the layer using the given input and delta

        Args:
            delta (np.ndarray): The gradient of the loss function with respect to the output layer
            x (np.ndarray): The input to the layer
        """
        for neuron, delta_w, delta_b in zip(self.neurons, delta_weights, delta_biases):
            neuron.update_parameters(delta_w, delta_b)


class NeuralNetwork:
    """
    A simple implementation of a feedforward neural network with customizable
    number of layers, activation functions, and loss function. This neural network
    supports basic training through backpropagation and can make predictions on new data.

    The neural network initializes with specified sizes for the input, hidden, and
    output layers, an activation function for the neurons, and a loss function
    for training evaluation. The architecture is flexible, allowing for a variable
    number of hidden layers and neurons within those layers.

    Parameters:
    - input_layer_size (int): The number of features in the input data.
    - hidden_layers (list[int]): A list containing the number of neurons in each hidden layer.
    - output_layer_size (int): The number of neurons in the output layer.
    - activation (Activation, optional): The activation function to be used by the neurons. Defaults to Sigmoid.
    - cost_function (Loss, optional): The loss function to be used for evaluating the training. Defaults to MeanSquaredError.
    - learning_rate (float, optional): The learning rate for the optimization algorithm. Defaults to 0.01.
    - epochs (int, optional): The number of epochs to train the neural network. Defaults to 100.

    Methods:
    - train(x, y): Trains the neural network on the given input-output pairs using backpropagation.
    - predict(x): Predicts the output for the given input using the trained neural network.

    Example usage:
    ```python
    >>> X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    >>> loss_function = MeanSquaredError()

    # Train a model with 2 input features, 2 hidden neurons, and 1 output neuron
    >>> neuralNetwork = NeuralNetwork(2, [2], 1)
    >>> neuralNetwork.train(X_train, y_train)

    # Make predictions on the training set
    >>> predictions = neuralNetwork.predict(X_train)
    >>> loss = loss_function(y_train, predictions)
    >>> print(f"MSE Loss on Training: {loss}")
    MSE Loss on Training: 0.06979377674222319

    >>> # Make predictions on the test set
    >>> predictions = neuralNetwork.predict(X_test)
    >>> loss = loss_function(y_test, predictions)
    >>> print(f"MSE Loss on test: {loss}")
    MSE Loss on test: 0.07305711106937894
    ```
    """

    def __init__(
        self,
        input_layer_size: int,
        hidden_layers: list[int],
        output_layer_size: int,
        activation: Activation = Sigmoid,
        cost_function: Loss = MeanSquaredError,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        self.activation: Activation = activation
        self.cost_function: Loss = cost_function
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.input_layer_size = input_layer_size
        self._init_layers(
            input_layer_size, hidden_layers, output_layer_size, learning_rate
        )

        self.logger = []

    def _init_layers(
        self,
        input_layer_size: int,
        hidden_layers: list[int],
        output_layer_size: int,
        learning_rate: float,
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
                Layer(inputs, layer, self.activation, self.cost_function, learning_rate)
            )
            inputs = layer

        self.output_layer = Layer(
            hidden_layers[-1],
            output_layer_size,
            self.activation,
            self.cost_function,
            learning_rate,
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
        for epoch in range(self.epochs):
            for x_i, y_i in zip(x, y):
                self._train_single(x_i, y_i)

            # Log the error every x epochs
            if epoch % 1 == 0:
                total_error = self.cost_function()(y, self.predict(x))
                self.logger.append(total_error)

    def _train_single(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the neural network using the provided input-output pairs.
        """
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

        # Backwards pass
        # Compute the gradient of the loss function with respect to the output layer
        delta_weights = (
            activations[-2]
            * self.activation().derivative(activations[-1])
            * 2
            * (activations[-1] - y)
        )

        delta_biases = (
            self.activation().derivative(activations[-1]) * 2 * (activations[-1] - y)
        )

        # Compute the gradient of the loss function with respect to the output layer
        delta_previous_activation = np.zeros_like(len(self.output_layer.neurons))
        for i, neuron in enumerate(self.output_layer.neurons):
            delta_previous_activation = (
                neuron.weights
                * self.activation().derivative(activations[-1])
                * 2
                * (activations[-1] - y)
            )
        # Update the weights and biases of the output layer
        self.output_layer.update_parameters(delta_weights, delta_biases)

        for i in range(len(self.hidden_layers), 0, -1):
            delta_weights = (
                self.hidden_layers[i - 1].get_inputs(activations[i - 1])
                * self.activation().derivative(activations[i])
                * 2
                * (activations[i] - y)
            )
            delta_biases = (
                self.activation().derivative(activations[i]) * 2 * (activations[i] - y)
            )
            self.hidden_layers[i - 1].update_parameters(delta_weights, delta_biases)

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

    def summary(self):
        """
        Prints a summary of the neural network architecture, including the layer types,
        number of neurons in each layer, activation functions, and total trainable parameters.
        """

        summary_str = "Neural Network Architecture Summary:\n"
        summary_str += "===================================\n"
        summary_str += f"Input Layer:\n  Size: {self.input_layer_size} neurons\n\n"

        # Initialize input size with input layer size
        inputs = self.input_layer_size
        total_params = 0

        # Calculate parameters for each hidden layer
        for i, layer in enumerate(self.hidden_layers):
            # Weights + biases
            layer_params = (inputs * len(layer.neurons)) + len(layer.neurons)
            total_params += layer_params
            summary_str += f"Hidden Layer {i + 1}:\n  Size: {len(layer.neurons)} neurons\n  Activation: {type(layer.neurons[0].activation).__name__}\n  Params: {layer_params}\n"
            inputs = len(layer.neurons)

        # Calculate parameters for output layer
        output_layer_params = (inputs * len(self.output_layer.neurons)) + len(
            self.output_layer.neurons
        )
        total_params += output_layer_params
        summary_str += f"Output Layer:\n  Size: {len(self.output_layer.neurons)} neurons\n  Activation: {type(self.output_layer.neurons[0].activation).__name__}\n  Params: {output_layer_params}\n"

        summary_str += f"Total Trainable Parameters: {total_params}\n"

        return summary_str
