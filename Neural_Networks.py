import numpy as np
import pickle

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input


class Sigmoid(Layer):
    def forward(self, input):
        self.sigmoid = 1 / (1 + np.exp(-input))
        return self.sigmoid

    def backward(self, input, grad_output):
        grad_sigmoid = self.sigmoid * (1 - self.sigmoid)
        return grad_output * grad_sigmoid


class Tanh(Layer):
    def forward(self, input):
        self.tanh = np.tanh(input)
        return self.tanh

    def backward(self, input, grad_output):
        grad_tanh = 1 - self.tanh ** 2
        return grad_output * grad_tanh


class Softmax(Layer):
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def backward(self, input, grad_output):
        return grad_output


class BinaryCrossEntropy(Layer):
    def forward(self, input, y):
        self.input = input
        self.y = y
        eps = 1e-9  # to avoid taking log of zero
        return -y * np.log(input + eps) - (1 - y) * np.log(1 - input + eps)

    def backward(self, input, y):
        eps = 1e-9  # to avoid division by zero
        return -(y / (input + eps) - (1 - y) / (1 - input + eps))


class CategoricalCrossEntropy(Layer):
    def forward(self, input, y):
        self.input = input
        self.y = y
        eps = 1e-9
        return -np.sum(y * np.log(input + eps), axis=-1)

    def backward(self, input, y):
        eps = 1e-9
        return -(y / (input + eps))


class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        activations = []
        for layer in self.layers:
            activations.append(layer.forward(input))
            input = activations[-1]
        return activations

    def backward(self, input, grad_output):
        for layer_input, layer in zip(reversed(input), reversed(self.layers)):
            grad_output = layer.backward(layer_input, grad_output)
        return grad_output

    def save_weights(self, path):
        weights = [layer.weights for layer in self.layers if isinstance(layer, Linear)]
        biases = [layer.biases for layer in self.layers if isinstance(layer, Linear)]
        with open(path, 'wb') as f:
            pickle.dump((weights, biases), f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights, biases = pickle.load(f)
        for layer, weight, bias in zip(self.layers, weights, biases):
            if isinstance(layer, Linear):
                layer.weights = weight
                layer.biases = bias
