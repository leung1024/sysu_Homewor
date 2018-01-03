import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

N = 300

def Exercise1_a():
    # create a 2-2-1 network
    network_size = np.array([2, 2, 1])
    # set weights and bias
    w_b = [np.array([[0.5, -0.5],
                     [0.3, -0.4],
                     [-0.1, 1]]),
           np.array([[1],
                     [-2],
                     [0.5]])
           ]
    net_obj = NeuralNetwork([2,2,1], w_b)
    # generate sample points
    samples = np.random.uniform(-5, 5, [2, N])
    samples[1,:] *= -1
    plt.figure()
    for index in range(samples.shape[1]):
        x1 = samples[0, index]
        x2 = samples[1, index]
        rst = net_obj.feedforward(np.array([x1,x2]))
        if rst <= 0:
            plt.plot(x1, x2, 'or')
        else:
            plt.plot(x1, x2, 'ob')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def Exercise1_b():
    # create a 2-2-1 network
    network_size = np.array([2, 2, 1])
    # set weights and bias
    w_b = [np.array([[-1, 1],
                     [-0.5, 1.5],
                     [1.5, -0.5]]),
           np.array([[0.5],
                     [-1],
                     [1]])
           ]
    net_obj = NeuralNetwork([2,2,1], w_b)
    # generate sample points
    samples = np.random.uniform(-5, 5, [2, N])
    samples[1,:] *= -1
    plt.figure()
    for index in range(samples.shape[1]):
        x1 = samples[0, index]
        x2 = samples[1, index]
        rst = net_obj.feedforward(np.array([x1,x2]))
        if rst <= 0:
            plt.plot(x1, x2, 'or')
        else:
            plt.plot(x1, x2, 'ob')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# Exercise1_a()
# Exercise1_b()