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
import matplotlib.pyplot as plt
import random


class NeuralNetwork():

    def __init__(self, network_sizes, w_b=[]):
        self.sizes = network_sizes
        self.layer_sizes = len(network_sizes)
        self.t_output = []
        self.raw_output = []
        self.j_val_data = []
        self.training_step = 0
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
        if input_val.shape != (1, input_val.size):
            input_val = np.reshape(input_val,[1, len(input_val)])
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

    def plot_learning_curve(self):
        plt.figure()
        plt.plot(range(len(self.j_val_data)), self.j_val_data)
        plt.show()

    def stochastic_backpropagation(self, data_set, label, theta, eta, max_step=100):
        step = 0
        
        # input_val_len = len(input_val)
        # input_val = np.reshape(input_val,[1, input_val_len])
        # if pattern_num <= 0 or pattern_num > input_val_len:
        #     print('Please use a vaild pattern_num')
        #     raise ValueError
        while True:
            # choose a pattern
            if len(data_set[0]) != self.sizes[0]:
                print('Please use a vaild data_set')
                raise ValueError
            pattern_index = random.randint(0, len(data_set) - 1)
            input_val = data_set[pattern_index]
            act_val = np.array([label[pattern_index]])
            input_val = np.reshape(input_val,[1, len(input_val)])
            cur_output = self.feedforward(input_val)
            j_val = self.cost_function(cur_output, act_val)
            self.j_val_data.append(j_val)
            if j_val < theta or step > max_step:
                self.training_step = step
                print(j_val)
                print(j_val == np.float64('nan'))
                print(step)
                break
            # choosen pattern
            # pattern_index = random.sample(range(input_val_len), pattern_num)
            # pattern_index = sorted(pattern_index)
            # try to update weight
            d_hidden2output = 1 - np.tanh(self.t_output[1])**2 
            # d_input2hidden = 1 - np.tanh(self.net_function(input_val, self.w_b[0]))**2
            d_input2hidden = 1 - np.tanh(self.t_output[0])**2
            # import pdb;pdb.set_trace()
            d_costfunc_1 = (act_val - cur_output) * d_hidden2output
            d_costfunc_0 = d_input2hidden * np.dot(d_costfunc_1, self.w_b[1][1:,:])
            # update weight
            self.w_b[0][1:,:] += eta * input_val.T * d_costfunc_0
            self.w_b[1][1:,:] += eta * self.t_output[0].T * d_costfunc_1
            # omega_1 = self.w_b[0][1:,:]
            # # omega_1[pattern_index] -= eta * input_val.T[pattern_index] * delta_j
            # omega_1 -= eta * input_val.T * d_costfunc_0
            # omega_2 = self.w_b[1][1:,:]
            # omega_2 -= eta * self.t_output[0].T * d_costfunc_1
            # self.w_b[0][1:,:] = omega_1
            # self.w_b[1][1:,] = omega_2
            step += 1







            
        
