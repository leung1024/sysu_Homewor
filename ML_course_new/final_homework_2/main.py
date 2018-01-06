import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

N = 300

DATASET = [[0.28 ,1.31 ,-6.2 ],
[0.07 ,0.58 ,-0.78],
[1.54 ,2.01 ,-1.63],
[-0.44, 1.18, -4.32],
[-0.81, 0.21, 5.73],
[1.52 ,3.16 ,2.77 ],
[2.20 ,2.42 ,-0.19],
[0.91 ,1.94 ,6.21 ],
[0.65 ,1.93 ,4.38 ],
[-0.26, 0.82,-0.96],
[0.011, 1.03,-0.21],
[1.27 ,1.28 , 0.08],
[0.13 ,3.12 , 0.16],
[-0.21, 1.23,-0.11], 
[-2.18, 1.39,-0.19],
[0.34 ,1.96 ,-0.16], 
[-1.38, 0.94, 0.45], 
[-0.12, 0.82, 0.17], 
[-1.44, 2.31, 0.14], 
[0.26 ,1.94 , 0.08],]
LABEL = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

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


def Exercise2_a():
    # create a 3-1-1 network
    network_size = np.array([3, 1, 1])
    # set weights and bias
    w_b = []
    net_obj = NeuralNetwork(network_size, w_b)
    # generate sample points
    samples = np.random.uniform(-5, 5, [2, N])
    samples[1,:] *= -1
    # for index in range(samples.shape[1]):
    #     x1 = samples[0, index]
    #     x2 = samples[1, index]
    rst = net_obj.feedforward(np.array([0.28, 1.31, -6.2]))
    net_obj.stochastic_backpropagation(DATASET, LABEL, 0, 0.1, max_step=5000)
    #     if rst <= 0:
    #         plt.plot(x1, x2, 'or')
    #     else:
    #         plt.plot(x1, x2, 'ob')
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.show()
    net_obj.plot_learning_curve()
    rst1 = net_obj.feedforward(np.array([0.28, 1.31, -6.2]))
    rst2 = net_obj.feedforward(np.array([0.26 ,1.94 , 0.08]))
    print(np.sign(rst1))
    print(np.sign(rst2))

# Exercise1_a()
# Exercise1_b()
Exercise2_a()