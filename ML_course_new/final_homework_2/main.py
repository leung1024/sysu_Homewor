import numpy as np
from neural_network import NeuralNetwork

def main():
    # create a 2-2-1 network
    network_size = np.array([2, 2, 1])
    weights = np.array([[0.3, -0.4, -2],
                        [-0.1, 1.0, 0.5]])
    biases = np.array([[0.5, -0.5]])
    net_obj = NeuralNetwork()