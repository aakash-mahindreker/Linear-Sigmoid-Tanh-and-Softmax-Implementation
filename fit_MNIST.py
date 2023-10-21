import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from Neural_Networks import Sequential,Linear,Sigmoid,Softmax,Tanh,BinaryCrossEntropy,CategoricalCrossEntropy

"""
In this training i have used three different architectures... as follows..
Linear, Sigmoid, Linear Softmax
Linear, Tanh, Linear Softmax
Linear, Softmax, Linear Softmax

I have saved individual model weights for your reference.
"""

# load data Mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


network_1 = Sequential([
    Linear(784, 100),
    Sigmoid(),
    Linear(100, 10),
    Softmax()
])
network_2 = Sequential([
    Linear(784, 100),
    Tanh(),
    Linear(100, 10),
    Softmax()
])
network_3 = Sequential([
    Linear(784, 100),
    Softmax(),
    Linear(100, 10),
    Softmax()
])
networks = [network_1, network_2, network_3]


def train(network, x_train, y_train, x_val, y_val, batch_size=128, max_epochs=10, early_stopping=5):
    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = np.inf
    counter = 0

    for epoch in range(max_epochs):
        # batch training
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        train_losses = []
        for batch in batches:
            x_batch = x_train[batch]
            y_batch = y_train[batch]
            layer_activations = network.forward(x_batch)
            layer_inputs = [x_batch] + layer_activations
            logits = layer_activations[-1]
            loss = CategoricalCrossEntropy().forward(logits, y_batch)
            train_losses.append(np.mean(loss))
            grad_loss = CategoricalCrossEntropy().backward(logits, y_batch)
            network.backward(layer_inputs[:-1], grad_loss)

        # validation
        val_activations = network.forward(x_val)
        val_logits = val_activations[-1]
        val_loss = CategoricalCrossEntropy().forward(val_logits, y_val)

        # average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
        if counter >= early_stopping:
            break

    return history


# split your data into train and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# train your networks and plot the losses
print("--------------------------------------------------------------------")
for i, network in enumerate(networks):
    print(f'Training network {i+1}...')
    print(f"Network {i+1} Summary:")
    for layers in network.layers:
        layer_name = str(layers)
        a = layer_name.index(".")
        b = layer_name.index("object")
        print(layer_name[a+1:b])
    history = train(network, x_train, y_train, x_val, y_val)

    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Network {i + 1} Losses over time')
    plt.legend()
    plt.show()

    test_activations = network.forward(x_test)
    test_logits = test_activations[-1]
    test_pred = np.argmax(test_logits, axis=-1)
    test_true = np.argmax(y_test, axis=-1)
    test_acc = accuracy_score(test_true, test_pred)
    print(f'Network {i + 1} test accuracy: {test_acc}')
    # saving all weights...
    path = f"MNIST_model{i}"
    network.save_weights(path)
    print("--------------------------------------------------------------------")
