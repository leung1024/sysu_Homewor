#=============================================================================
#
# Author: lujunliang Arron - j_ackson123@163.com
#
# QQ : 2817022753
#
# Last modified: 2017-12-29 14:14
#
# Filename: neural_network.py
#
# Description: 
#
#=============================================================================
import numpy as np


class NeuralNetwork():

    def __init__(self, network_sizes, weights=[], biases=[]):
        self.sizes = network_sizes
        self.layer_sizes = len(network_sizes)
        if weights and biases:
            self.weights = weights
            self.biases = biases
        else:
            pass
            # self.weights = np.array([np.random.randn(y, x) for x, y in zip(network_sizes[:-1], sizes[1:])])
            # self.biases = np.array([np.random.randn(y, 1) for y in network_sizes[1:]])

    def format_check(self, *args):
        for val in args:
            if type(val) != np.narray:
                raise TypeError

    def activation_function(self):
        return numpy.tanh()

    def feedforward(self, input_val):
        self.format_check(input_val)
        for layer_num in range(self.layer_sizes):
            input_val = np.dot(input_val.T, self.weights[layer_num]) + self.biases[layer_num]
        output_val = input_val
        return output_val



            
        
