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
        self.t_output = []
        self.raw_output = []
        if w_b:
            [self.format_check(item) for item in w_b]
            self.w_b = w_b
        else:
            self.w_b = []
            for layer_index in range(self.layer_sizes - 1):
                cur_layer_size = network_sizes[layer_index] + 1
                post_layer_size = network_sizes[layer_index+1]
                self.w_b.append(2*np.random.rand(cur_layer_size, post_layer_size) - 1)

            # self.weights = np.array([np.random.randn(y, x) for x, y in zip(network_sizes[:-1], sizes[1:])])
            # self.biases = np.array([np.random.randn(y, 1) for y in network_sizes[1:]])

    def format_check(self, *args):
        for val in args:
            if type(val) != np.ndarray:
                raise TypeError

    def cost_function(self, exp_val, act_val):
        return sum((act_val-exp_val)**2) / 2

    def transfer_function(self, input_val):
        a = 1.716
        b = 2/3
        input_val *= b
        return a * np.tanh(input_val)

    def net_function(self, input_val, w_b):
        input_val = np.insert(input_val, 0, 1)
        return np.dot(input_val.T, w_b)

    def feedforward(self, input_val):
        self.format_check(input_val)
        self.t_output = []
        self.raw_output = []
        for layer_args in self.w_b:
            input_val = np.insert(input_val, 0, 1)
            input_val = np.dot(input_val.T, layer_args)
            self.raw_output.append(input_val)
            input_val = self.transfer_function(input_val)
            self.t_output.append(input_val)
        output_val = input_val
        return output_val

    def stochastic_backpropagation(self, input_val, act_val, theta, eta, pattern_num=2):
        m = 0
        input_val_len = len(input_val)
        if pattern_num <= 0 or pattern_num > input_val_len:
            print('Please use a vaild pattern_num')
            raise ValueError
        while True:
            import pdb;pdb.set_trace()
            cur_output = self.feedforward(input_val)
            j_val = self.cost_function(cur_output, act_val)
            if j_val < theta or input_val_len < m:
                break
            # choosen pattern
            pattern_index = np.random.randint(0, input_val_len-1, [pattern_num])
            # try to update weight
            f_net_k = self.raw_output[1]
            f_net_j = np.cosh(self.net_function(input_val, self.w_b[0]))
            delta_k = (act_val - cur_output) * f_net_k
            delta_j = f_net_j * np.dot(delta_k, self.w_b[1][1:,:])
            omega_1 = self.w_b[0][1:,:]
            omega_1[pattern_index] += eta * input_val[pattern_index].T * delta_j
            omega_2 = self.w_b[1][1:,:]
            omega_2 += eta * self.t_output[0].T * delta_k
            self.w_b[0][1:,:] = omega_1
            self.w_b[1][1:,] = omega_2







            
        
