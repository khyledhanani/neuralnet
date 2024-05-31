import numpy as np

class NeuralNetwork:
    def __init__(self, inputs: list, outputs: list, hidden_layers: int, hidden_nodes: list, activation_function: callable, activation_derivative: callable):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.layer_sizes = [self.inputs.shape[1]] + hidden_nodes + [self.outputs.shape[1]]
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.random.randn(1, size) for size in self.layer_sizes[1:]]
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def feedforward(self, input_data):
        layer_output = input_data
        self.layer_outputs = [layer_output] 
        for weight, bias in zip(self.weights, self.biases):
            layer_output = self.activation_function(np.dot(layer_output, weight) + bias)
            self.layer_outputs.append(layer_output)
        return layer_output

    def backprop(self, input_data, output_data):
        self.feedforward(input_data)
        delta = (self.layer_outputs[-1] - output_data) * self.activation_derivative(self.layer_outputs[-1])
        gradients_w = []
        gradients_b = []
        for i in reversed(range(len(self.weights))):
            gradients_w.insert(0, np.dot(self.layer_outputs[i].T, delta))
            gradients_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return gradients_w, gradients_b
    
    '''
    Activation Functions
    - Sigmoid & RELU
    '''

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x) 
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
