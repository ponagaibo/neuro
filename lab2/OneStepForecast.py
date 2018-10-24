import random

import math
import numpy as np
import matplotlib.pyplot as plt

a = 1.0
b = 4.5
h = 0.01
# d = 5
# mu = 0.01
n = 10

def myfunc(t):
    return math.sin(math.sin(t) * t * t - t)

def create_training_set():
    times = math.ceil((b - a) / h + 1)
    trainig_set = [None] * times
    x = [a + h * i for i in range(0, times)]
    for i in range(0, len(x)):
        trainig_set[i] = myfunc(x[i])
    return trainig_set

def init_weight(d):
    w = [None] * (d + 1)
    for i in range(0, d + 1):
        w[i] = random.uniform(-2, 2)
        # print("w[i]: " + str(w[i]))
    return w

def count_nets(training_set, weights, d, mu):
    total_error = 0
    net = [None] * (len(training_set) - d)
    max = 0
    place = 0
    for k in range(0, len(training_set) - d):
        net[k] = 1 * weights[d]
        for i in range(0, d):
            net[k] += training_set[d - i - 1 + k] * weights[i]
        error = training_set[d + k] - net[k]
        if (error > max):
            max = error
            place = k
        # print("k = " + str(k) + ", error = " + str(error))
        if (k > d):
            total_error += error * error
        dw = [None] * d
        for i in range(0, d):
            dw[i] = training_set[d - i - 1 + k] * error * mu
            weights[i] += dw[i]
        weights[d] += error * mu
    mse = math.sqrt(total_error / (len(training_set) - d))
    # print("mse: " + str(mse) + ", max: " + str(max) + " at " + str(place))
    return weights, net, mse

def draw(training_set, net, d):
    times = math.ceil((b - a) / h + 1)
    x = [a + h * i for i in range(0, times)]
    plt.plot(x, training_set, c='r', linewidth = 1, marker = "1", alpha = 0.5, label="Real function")

    times = math.ceil((b - a - d * h) / h + 1)
    x = [a + h * (d + i) for i in range(0, times)]
    plt.plot(x, net, c='m', linewidth = 1, marker = ".", alpha = 1, label="Predicted function")
    plt.grid(True)
    plt.legend()

def predict(weights, d):
    real_values = [None] * (n + d)
    net = [None] * n
    for i in range(-(d - 1), n + 1):
        real_values[i + d - 1] = myfunc(b + i * h)
    total_error = 0
    max = 0
    place = 0
    for k in range(0, n):
        net[k] = 1 * weights[d]
        for i in range(0, d):
            net[k] += real_values[d - i - 1 + k] * weights[i]
        error = real_values[d + k] - net[k]
        if (error > max):
            max = error
            place = k
        # print("k = " + str(k) + ", error = " + str(error))
        if (k > d):
            total_error += error * error
    mse = math.sqrt(total_error / n)
    # print("mse: " + str(mse) + ", max: " + str(max) + " at " + str(place))
    return net, mse

def draw_new_points(new_net):
    x = [b + h * (i + 1) for i in range(0, n)]
    y = [None] * n
    for i in range(0, n):
        y[i] = myfunc(x[i])
    plt.subplot(2,1,1)
    plt.plot(x, y, c='b', linewidth=1, marker="1", alpha=0.5, label="Real prolonged function")

    # new_x = [b + h * i for i in range(1, 11)]
    plt.plot(x, new_net, c='c', linewidth=1, marker=".", alpha=1, label="Predicted prolonged function")
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    error = [None] * n
    for i in range(0, n):
        error[i] = math.fabs(new_net[i] - y[i])
    plt.plot(x, error)
    plt.grid(True)

def main():
    training_set = create_training_set()
    d = 1
    mu = 0.01
    w = init_weight(d)
    times = 50
    er = 1
    k = []
    for i in range(0, times):
        w, k, er = count_nets(training_set, w, d, mu)
    print("weights: ")
    for i in range(len(w)):
        print("w[" + str(i) + "] = " + str(w[i]))
    print("MSE: " + str(er))
    draw(training_set, k, d)
    plt.show() # comment to have common plot

    d=3
    mu = 0.3
    training_set = create_training_set()
    w = init_weight(d)
    times = 600
    eps = 10 ** -6
    i = 0
    er = eps + 1
    while i < times and er > eps:
        w, k, er = count_nets(training_set, w, d, mu)
        i += 1
    print("weights: ")
    for i in range(len(w)):
        print("w[" + str(i) + "] = " + str(w[i]))
    print("MSE: " + str(er))
    new_net, er = predict(w, d)
    draw_new_points(new_net)
    plt.show()

main()