import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

# Dummy neural network structure
class NeuralNetwork:
    def __init__(self):
        # Example: 3 layers (input, hidden, output)
        self.layers = [3, 5, 2]  # 3 input neurons, 5 hidden neurons, 2 output neurons
        self.activations = [np.zeros(l) for l in self.layers]
        self.weights = [np.random.randn(self.layers[i], self.layers[i-1]) if i > 0 else np.random.randn(self.layers[i], 1) for i in range(len(self.layers))]
        self.gradients = [np.zeros_like(w) for w in self.weights]

    def feedforward(self, inputs):
        self.activations[0] = inputs
        for i in range(1, len(self.layers)):
            self.activations[i] = np.dot(self.weights[i], self.activations[i-1])
        return self.activations[-1]  # Output layer

    def backpropagate(self, target):
        # For simplicity, using a dummy gradient update for backpropagation
        error = self.activations[-1] - target
        for i in reversed(range(1, len(self.layers))):
            self.gradients[i] = np.outer(error, self.activations[i-1])  # Simplified gradient calculation
            error = np.dot(self.weights[i].T, error)  # Propagate error backwards

    def update_weights(self, learning_rate=0.01):
        for i in range(1, len(self.layers)):
            self.weights[i] -= learning_rate * self.gradients[i]

# Neural Network GUI with Tkinter and matplotlib for visualization
class TrainingGUI:
    def __init__(self, root, nn):
        self.root = root
        self.nn = nn
        self.root.title("Neural Network Training Visualization")

        # Set up tkinter canvas for network visualization
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        # Set up matplotlib for showing activations and gradients
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_matplotlib = FigureCanvasTkAgg(self.fig, root)
        self.canvas_matplotlib.get_tk_widget().pack()

        # Placeholder for training button
        self.train_button = tk.Button(root, text="Start Training", command=self.start_training)
        self.train_button.pack()

    def draw_network(self):
        self.canvas.delete("all")
        
        # Draw neurons as circles
        layer_positions = [(100, 50 + i * 60) for i in range(len(self.nn.layers))]  # Example positions
        for i, layer_size in enumerate(self.nn.layers):
            y_pos = layer_positions[i][1]
            for j in range(layer_size):
                self.canvas.create_oval(50 + i * 150, y_pos + j * 30, 80 + i * 150, y_pos + j * 30 + 20, fill="blue", outline="black")

        # Draw edges between layers
        for i in range(1, len(self.nn.layers)):
            for j in range(self.nn.layers[i]):
                for k in range(self.nn.layers[i-1]):
                    self.canvas.create_line(80 + (i-1) * 150, 50 + (i-1) * 30 + k * 30, 50 + i * 150, 50 + i * 30 + j * 30)

    def update_feedforward(self):
        self.nn.feedforward(np.random.randn(self.nn.layers[0]))  # Random input for simulation
        self.draw_network()
        
        # Update neuron activations (show color change based on activation value)
        for i, activation in enumerate(self.nn.activations):
            for j, value in enumerate(activation):
                color = self.get_activation_color(value)
                self.canvas.create_oval(50 + i * 150, 50 + i * 60 + j * 30, 80 + i * 150, 50 + i * 60 + j * 30 + 20,
                                        fill=color, outline="black")

    def update_backpropagation(self):
        # Simulate backpropagation and gradient flow
        target = np.random.randn(self.nn.layers[-1])  # Dummy target
        self.nn.backpropagate(target)
        
        # Update edge thickness based on gradient magnitude
        for i in range(1, len(self.nn.layers)):
            for j in range(self.nn.layers[i]):
                for k in range(self.nn.layers[i-1]):
                    gradient_magnitude = np.linalg.norm(self.nn.gradients[i][:, k])
                    self.canvas.create_line(80 + (i-1) * 150, 50 + (i-1) * 30 + k * 30,
                                            50 + i * 150, 50 + i * 30 + j * 30,
                                            width=gradient_magnitude * 2, fill="red")

    def get_activation_color(self, value):
        # Map activation to color
        if value < 0:
            return "red"
        elif value > 0.5:
            return "green"
        else:
            return "yellow"

    def start_training(self):
        def training_loop():
            for epoch in range(10):  # Simulate 10 epochs
                self.update_feedforward()  # Show feedforward
                time.sleep(0.5)  # Simulate training time
                self.update_backpropagation()  # Show backpropagation
                time.sleep(0.5)
        
        thread = threading.Thread(target=training_loop)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    nn = NeuralNetwork()
    root = tk.Tk()
    gui = TrainingGUI(root, nn)
    gui.draw_network()  # Draw the initial network
    root.mainloop()
