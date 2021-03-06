import random
import matplotlib.pyplot as plt
import numpy as np

values = [[1, 2.7, 4.3], [1, -3.8, 0.6], [1, -0.4, -4.9], [1, -1.7, -3.4], [1, 2.9, -1.9], [1, 0.2, -3.4]]
answers = [0, 0, 1, 1, 1, 1]
mu = 0.3

def check(w):
    newPoints = [[1, random.uniform(-5, 5), random.uniform(-5, 5)], [1, random.uniform(-5, 5), random.uniform(-5, 5)],
                [1, random.uniform(-5, 5), random.uniform(-5, 5)]]
    for i in range(len(newPoints)):
        net = 0.0
        for j in range(len(w)):
            net += w[j] * newPoints[i][j]
        plt.scatter(newPoints[i][1], newPoints[i][2], c='b')
        # print("point: " + str(newPoints[i][1]) + ";" + str(newPoints[i][2]) + ",  net: " + str(net))
        if net >= 0:
            plt.text(newPoints[i][1], newPoints[i][2], '1: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')
        else:
            plt.text(newPoints[i][1], newPoints[i][2], '0: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')

def draw(w, b):
    for i in range(len(values)):
        if answers[i] == 0:
            plt.scatter(values[i][1], values[i][2], c='r')
            plt.text(values[i][1], values[i][2], '0: (' + str(round(values[i][1], 2)) + ';' + str(round(values[i][2], 2)) + ')')
        else:
            plt.scatter(values[i][1], values[i][2], c='g')
            plt.text(values[i][1], values[i][2], '1: (' + str(round(values[i][1], 2)) + ';' + str(round(values[i][2], 2)) + ')')

    x=np.arange(-5,5,0.02)
    plt.plot(x,(-w[0] - w[1] * x) / w[2])
    plt.text(-5, (-w[0] - w[1] * (-5)) / w[2], 'plot: y=' + str(round(-w[1] / w[2], 2)) + 'x+'
             + str(round(-w[0] / w[2], 2)))
    if b == 1:
        check(w)

    plt.grid(True)
    plt.show()

def initWeight():
    w = [-1, 1, 1]
    for i in range(len(w)):
        w[i] = random.uniform(-2, 2)
    return w

def train(w, s):
    if s == 0:
        answers[2] = 0
    eps = 1.0
    steps = 0
    while eps > 0:
        eps = 0
        for i in range(len(values)):
            curr = 0.0
            for j in range(len(w)):
                curr += w[j] * values[i][j]
            if curr >= 0:
                out = 1
            else:
                out = 0
            err = answers[i] - out
            if (err != 0):
                eps += 1
                for k in range(len(w)):
                    coef = mu * values[i][k] * err
                    w[k] += coef
        steps += 1
        if steps > 500:
            print("Can not train perceptron")
            break
    print("steps: " + str(steps))
    return w

def main():
    w = initWeight()
    # set s = 0 for nonseparable points
    s = 1
    w = train(w, s)
    print("final weights: ")
    for i in range(len(w)):
        print(w[i])
    # set b = 1 to draw with checking random points
    b = 1
    draw(w, b)

main()