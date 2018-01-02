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
        if weights and biases:
            [self.format_check(item) for item in w_b]
            self.w_b = w_b
        else:
            pass
            # self.weights = np.array([np.random.randn(y, x) for x, y in zip(network_sizes[:-1], sizes[1:])])
            # self.biases = np.array([np.random.randn(y, 1) for y in network_sizes[1:]])

    def format_check(self, *args):
        for val in args:
            if type(val) != np.narray:
                raise TypeError

    def activation_function(self, *args):
        return numpy.tanh()

    def feedforward(self, input_val):
        self.format_check(input_val)
        for layer_args in self.w_b:
            input_val = np.dot(input_val.T, layer_args)
        output_val = numpy.tanh(input_val)
        return output_val



            
        
