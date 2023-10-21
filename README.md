# Linear-Sigmoid-Tanh-and-Softmax-Implementation

A layer class as a parent is implemented and the classes Linear, Sigmoid, Tanh and Softmax are implemented which inherits the layer class. A sequential class is implemented which will contain a list of layers.

The functions for error calculations are implemented which are Binary Cross Entropy and Categorical Cross Entropy.

The Sequential model is implemented on XOR and the weights are saved as XOR_solved (pickle file). 

Two networks implemented as follows:
<img width="736" alt="image" src="https://github.com/aakash-mahindreker/Linear-Sigmoid-Tanh-and-Softmax-Implementation/assets/70765660/d7762bca-e144-49df-983c-c5cd6fa58a1b">

The Sequential model for handwritten digit recognition and the weights are saved as MNIST_model1, MNIST_model2, etc. 

## The result is as follows:

### 1. Network 1: Has layers: Linear, Sigmoid, Linear and Softmax. 

The accuracy of the Network 1 is 0.1032.
<img width="796" alt="image" src="https://github.com/aakash-mahindreker/Linear-Sigmoid-Tanh-and-Softmax-Implementation/assets/70765660/3371c643-1e65-4921-99df-28a8bb6985c1">

## 2. Network 2: Has 4 layers: Linear, Tanh, Linear and Softmax. 

The accuracy of the Network 2 is 0.1467.
<img width="792" alt="image" src="https://github.com/aakash-mahindreker/Linear-Sigmoid-Tanh-and-Softmax-Implementation/assets/70765660/2ed51dd3-f522-4a7e-a620-1361899548f3">

## 3. Network 3: Has layers Linear, softmax, linear and softmax. 

The accuracy of the Network 2 is 0.1028.
<img width="808" alt="image" src="https://github.com/aakash-mahindreker/Linear-Sigmoid-Tanh-and-Softmax-Implementation/assets/70765660/12f2a50c-da7c-4bfb-b692-102c90d14e41">
