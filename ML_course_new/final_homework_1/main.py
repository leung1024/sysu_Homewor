#=============================================================================
#
# Author: lujunliang Arron - 2817022753@qq.com
#
# QQ : 2817022753
#
# Last modified: 2017-12-05 09:41
#
# Filename: main.py
#
# Description: 
#
#=============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
GD_LEARNING_RATE = 0.00001
SGD_LEARNING_RATE = 0.00005
N = 2000



def generate_uniformity_distribution(N):
    return np.random.uniform(0, 1, [2, N])

class Centroid:

    def __init__(self, data):
        self.x = data[0,0]
        self.y = data[1,0]
        self.sample_data = data

    def cal_x_gradient(self):
        return (-2) * (sum(self.sample_data[0,:]) - self.sample_data.shape[1]*self.x)

    def cal_y_gradient(self):
        return (-2) * (sum(self.sample_data[1,:]) - self.sample_data.shape[1]*self.y)

    def update_centroid(self, learning_rate):
        self.x_gradient = self.cal_x_gradient()
        self.x -= learning_rate * self.x_gradient
        self.y_gradient = self.cal_y_gradient()
        self.y -= learning_rate * self.y_gradient

    def reset(self):
        self.x = self.sample_data[0,0]
        self.y = self.sample_data[1,0]


def main():
    # 1. Generate 2000 points uniformly at random in the two-dimmensional unit squre
    data = generate_uniformity_distribution(N)
    plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    p1, = ax1.plot(data[0,:], data[1,:], 'b.', label='uniform distribution')

    # 2. Using gradient descent to find out the centroid
    c_p = Centroid(data)
    for i in range(100):
        c_p.update_centroid(GD_LEARNING_RATE)
        p2, = ax1.plot(c_p.x, c_p.y, 'r+', label='GD')

    plt.legend(handles=[p1, p2], loc='upper left')

    # SGD
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    p3, = ax2.plot(data[0,:], data[1,:], 'b.', label='uniform distribution')
    c_p.reset()
    for i in range(100):
        rand_index = np.random.randint(0, N-1, size=500)
        sample_data_x = []
        sample_data_y = []
        for index in rand_index:
            sample_data_x.append(data[0,index])
            sample_data_y.append(data[1,index])
        c_p.sample_data = np.array([sample_data_x, sample_data_y])
        c_p.update_centroid(SGD_LEARNING_RATE)
        p4, = ax2.plot(c_p.x, c_p.y, 'gx', label='SGD')

    plt.legend(handles=[p3, p4], loc='upper left')
    datacursor()
    plt.show()

main()