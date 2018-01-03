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

    def __init__(self, network_sizes, w_b=[]):
        self.sizes = network_sizes
        self.layer_sizes = len(network_sizes)
        self.output = []
        self.raw_output = []
        if w_b:
            [self.format_check(item) for item in w_b]
            self.w_b = w_b
        else:
            self.w_b = np.array([])
            for layer_index in range(self.layer_sizes - 1):
                cur_layer_size = network_sizes[layer_index] + 1
                post_layer_size = network_sizes[layer_index+1] + 1
                self.w_b = np.append(self.w_b, 2*np.random.rand(cur_layer_size, post_layer_size) - 1)
            # self.weights = np.array([np.random.randn(y, x) for x, y in zip(network_sizes[:-1], sizes[1:])])
            # self.biases = np.array([np.random.randn(y, 1) for y in network_sizes[1:]])

    def format_check(self, *args):
        for val in args:
            if type(val) != np.ndarray:
                raise TypeError

    def transfer_function(self, input_val):
        a = 1.716
        b = 2/3
        input_val *= b
        return a * np.tanh(input_val)

    def feedforward(self, input_val):
        self.format_check(input_val)
        for layer_args in self.w_b:
            input_val = np.insert(input_val, 0, 1)
            input_val = np.dot(input_val.T, layer_args)
            self.raw_output.append(input_val)
            input_val = self.transfer_function(input_val)
            self.a_output.append(input_val)
        output_val = input_val
        return output_val

    # def stochastic_backpropagation(self, theta, eta, m):
        



            
        
