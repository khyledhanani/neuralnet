
# Neural Network Implementation

## Description
This project provides a simple implementation of a feedforward neural network with backpropagation, built using Python and NumPy. It supports configurable number of hidden layers and nodes, and includes both the Sigmoid and ReLU activation functions. The network can be trained using either stochastic or batch gradient descent. 
It is still a work in progress, I am working on adding batch normalization and additional optimizers like Adam and Adadelta..

## Usage

### Creating a Neural Network
First, import the necessary libraries and the `NeuralNetwork` class:

```python
import numpy as np
from neuralnet import NeuralNetwork 
```

Create an instance of the `NeuralNetwork`:

```python
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
hidden_layers = 2
hidden_nodes = 5

nn = NeuralNetwork(inputs, outputs, hidden_layers, hidden_nodes, relu, relu_derivative)
```

### Training the Network
Set the learning rate and the number of epochs, and specify the gradient descent type:

```python
learning_rate = 0.01
epochs = 1000
gd_type = 'stochastic'  # Choose 'stochastic' or 'batch'

nn(learning_rate, epochs, gd_type)
```

### Testing the Network
You can feed data into the network using:

```python
input_data = [1, 0]
result = nn.feedforward(input_data)
print("Output:", result)
```

### Contributions
All contributions are welcome of course!
