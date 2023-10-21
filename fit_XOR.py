import numpy as np
from sklearn.metrics import accuracy_score

from Neural_Networks import Sequential,Linear,Sigmoid,Softmax,Tanh,BinaryCrossEntropy,CategoricalCrossEntropy

"""
I have used two different architectures with different activation functions like
Linear, sigmoid, linear, sigmoid
linear, tanh, linear, sigmoid

i have saved weights as well in a pickle format.
"""


def generate_XOR_easy():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)  # Keep y as a 1D array

    return X, y


X, y = generate_XOR_easy()

network_01 = Sequential([
    Linear(2, 10),
    Sigmoid(),
    Linear(10, 1),
    Sigmoid()
])
network_02 = Sequential([
    Linear(2, 10),
    Tanh(),
    Linear(10, 1),
    Sigmoid()
])

networks = [network_01, network_02]


def train(network, X, y):
    y = y.reshape(-1, 1)
    for i in range(10000):
        layer_activations = network.forward(X)
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]
        loss = BinaryCrossEntropy().forward(logits, y)
        grad_loss = BinaryCrossEntropy().backward(logits, y)
        network.backward(layer_inputs[:-1], grad_loss)


def predict(network, X):
    y_pred = network.forward(X)[-1]
    y_pred = np.round(y_pred)
    return y_pred

i=0
print("--------------------------------------------------------------------")
for network in networks:
    print(f'Training network {i+1}...')
    print(f"Network {i+1} Summary:")
    for layers in network.layers:
        layer_name = str(layers)
        a = layer_name.index(".")
        b = layer_name.index("object")
        print(layer_name[a+1:b])
    train(network, X, y)
    predictions = predict(network, X)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y, predictions.squeeze()))
    # saving weights..
    network.save_weights("XOR_solved")
    print("--------------------------------------------------------------------")
    i+=1