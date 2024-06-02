import numpy as np

class NeuralNetwork:

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, hidden_layers: int, hidden_nodes: int, activation_function: callable, activation_derivative: callable):
        self.inputs = inputs
        self.outputs = outputs
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases for each layer
        self.weights = [np.random.normal(size=(inputs.shape[1], hidden_nodes))]
        self.biases = [np.random.normal(size=(hidden_nodes))]
        
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.normal(size=(hidden_nodes, hidden_nodes)))
            self.biases.append(np.random.normal(size=(hidden_nodes)))
        
        self.weights.append(np.random.normal(size=(hidden_nodes, outputs.shape[1])))
        self.biases.append(np.random.normal(size=(outputs.shape[1])))

    def feedforward(self, input_data):
        layer_output = input_data
        self.layer_outputs = [layer_output] 
        for weight, bias in zip(self.weights, self.biases):
            layer_output = self.activation_function(np.dot(layer_output, weight) + bias)
            self.layer_outputs.append(layer_output)
        return layer_output
    
    def backprop(self, input_data: np.ndarray, output_data: np.ndarray):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Forward pass
        layer_output = self.feedforward(input_data)
        error = output_data - layer_output
        
        # Backward pass
        for i in reversed(range(len(self.weights))):
            dA_dZ = self.activation_derivative(self.layer_outputs[i + 1])
            delta = error * dA_dZ
            gradients_w[i] = np.outer(self.layer_outputs[i], delta)
            gradients_b[i] = delta
            error = np.dot(delta, self.weights[i].T)
        
        return gradients_w, gradients_b
    
    def __call__(self, l_rate, epochs, gd_type):
        indices = np.arange(len(self.inputs))  
        for epoch in range(epochs):
            total_loss = 0

            if gd_type == 'stochastic':
                np.random.shuffle(indices)  
            
            total_gradients_w = [np.zeros_like(w) for w in self.weights]
            total_gradients_b = [np.zeros_like(b) for b in self.biases]

            for i in indices:
                output_layer = self.feedforward(self.inputs[i])
                total_loss += np.linalg.norm(self.outputs[i] - output_layer)
                gradients_w, gradients_b = self.backprop(self.inputs[i], self.outputs[i])

                if gd_type == 'batch':
                    total_gradients_w = [tw + gw for tw, gw in zip(total_gradients_w, gradients_w)]
                    total_gradients_b = [tb + gb for tb, gb in zip(total_gradients_b, gradients_b)]
                elif gd_type == 'stochastic':
                    self.weights = [w - l_rate * gw for w, gw in zip(self.weights, gradients_w)]
                    self.biases = [b - l_rate * gb for b, gb in zip(self.biases, gradients_b)]

            if gd_type == 'batch':
                self.weights = [w - l_rate * tw / len(self.inputs) for w, tw in zip(self.weights, total_gradients_w)]
                self.biases = [b - l_rate * tb / len(self.inputs) for b, tb in zip(self.biases, total_gradients_b)]

            print(f'Epoch {epoch} complete, loss: {total_loss / len(self.inputs)}')

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