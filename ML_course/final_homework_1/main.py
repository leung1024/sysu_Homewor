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
LEARNING_RATE = 0.00001

# 1. Generate 2000 points uniformly at random in the two-dimmensional unit squre

def generate_uniformity_distribution(N=2000):
    return np.random.uniform(0, 1, [2, N])

# 2. Using gradient descent to find out the centroid

def cal_distance(self, point_1x, point_1y, point_2x, point_2y):
    return (point_2x - point_1x)**2 + (point_2y - point_1y)**2

class Centroid:

    def __init__(self, data):
        self.p_x = data[0,0]
        self.p_y = data[1,0]
        self.x_len = len(data[0,:])
        self.y_len = len(data[1,:])
        self.data = data

    def cal_x_gradient(self):
        return (-2) * (sum(self.data[0,:]) - self.x_len*self.p_x)

    def cal_y_gradient(self):
        return (-2) * (sum(self.data[1,:]) - self.y_len*self.p_y)

    def update_centroid(self):
        self.x_gradient = self.cal_x_gradient()
        self.p_x -= LEARNING_RATE * self.x_gradient
        self.y_gradient = self.cal_y_gradient()
        self.p_y -= LEARNING_RATE * self.y_gradient


def main():
    data = generate_uniformity_distribution()
    plt.figure()
    plt.plot(data[0,:], data[1,:], '.')
    step_x = []
    step_y = []
    c_p = Centroid(data)
    for i in range(100):
        c_p.update_centroid()
        step_x.append(c_p.p_x)
        step_y.append(c_p.p_y)
        # plt.plot(c_p.p_x, c_p.p_y, 'rx')

    plt.plot(step_x, step_y, 'r+')
    datacursor()
    plt.show()

main()