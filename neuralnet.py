import numpy as np

class NeuralNetwork:

    def __init__(self, inputs: list, outputs: list, hidden_layers: int, hidden_nodes: int, activation_function: callable, activation_derivative: callable):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.normal(size=(len(self.inputs[0]), hidden_nodes, hidden_layers))
        self.biases = np.random.normal(size=(hidden_nodes, hidden_layers))
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative


    def feedforward(self, input_data):
        layer_output = input_data
        self.layer_outputs = [layer_output]  # Initialize as a Python list
        for weight, bias in zip(self.weights, self.biases):
            layer_output = self.activation_function(np.dot(layer_output, weight) + bias)
            self.layer_outputs.append(layer_output)
        return layer_output

    
    def backprop(self, input_data: np.array, output_data: np.array):
        gradients_w = []
        gradients_b = []
        error = output_data - self.feedforward(input_data)
        for i in reversed(range(len(self.weights))):
            dA_dZ = self.activation_derivative(self.layer_outputs[i + 1])
            dZ_dW = self.layer_outputs[i]
            delta = error * dA_dZ
            gradient_w = np.outer(dZ_dW, delta)
            gradients_w.append(gradient_w)
            gradients_b.append(delta)
            error = np.dot(self.weights[i], delta)
        
        return gradients_w[::-1], gradients_b[::-1]  # Reverse to match layer order

    
    def __call__(self, l_rate, epochs, gd_type):
        indices = np.arange(len(self.input_data))  
        total_loss = 0

        for epoch in range(epochs):
            if gd_type == 'stochastic':
                np.random.shuffle(indices)  # Shuffle indices for SGD
            
            total_gradient_w = 0
            total_gradient_b = 0

            for i in indices:
                output_layer = self.feedforward(self.input_data[i], self.activation_function)
                total_loss += np.linalg.norm(self.output_data[i] - output_layer)
                gradients_w, gradients_b = self.backprop(self.input_data[i], self.output_data[i], self.activation_function)

                if gd_type == 'batch':
                    # For batch gradient descent, accumulate gradients over the entire dataset
                    if total_gradient_w is None:
                        total_gradient_w = gradients_w
                        total_gradient_b = gradients_b
                    else:
                        total_gradient_w += gradients_w
                        total_gradient_b += gradients_b
                elif gd_type == 'stochastic':
                    # For stochastic gradient descent, update weights and biases immediately
                    self.weights -= l_rate * gradients_w
                    self.biases -= l_rate * gradients_b

            if gd_type == 'batch':
                # Update weights and biases at end of epoch
                self.weights -= l_rate * total_gradient_w
                self.biases -= l_rate * total_gradient_b

            print(f'Epoch {epoch} complete, loss: {total_loss / len(self.input_data)}')

    """
    Activation Functions:
    - Sigmoid
    - ReLU
    """
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
    

