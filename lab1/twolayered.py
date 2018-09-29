import random
import matplotlib.pyplot as plt
import numpy as np

# values = [[1, 0.6, -2.4], [1, 2.4, 0], [1, 1.4, -2], [1, -3.7, -0.3],
#           [1, -1.4, 2.8], [1, 2.8, 1.6], [1, -3.7, -4.8], [1, 0.5, -2]]
# answers = [[1, 0], [1, 1], [1, 0], [0, 0],
#            [0, 1], [1, 1], [1, 0], [1, 0]]

values = [[1, -1.5, -0.6], [1, 4.6, -4.6], [1, 4.7, -3.2], [1, 1.6, 0.8],
          [1, 1.7, -1.4], [1, 1.2, 3.1], [1, -4.9, -4.2], [1, 4.7, 1.5]]
answers = [[0, 0], [0, 1], [0, 1], [1, 0],
           [0, 0], [1, 0], [0, 1], [1, 1]]

mu = 0.3

# plt.scatter(-1.5, -0.6, c='r') #0 0
#
# plt.scatter(1.7, -1.4, c='r')
#
# plt.scatter(4.6, -4.6, c='g') # 0 1
# plt.scatter(4.7, -3.2, c='g') # 0 1
# plt.scatter(-4.9, -4.2, c='g') # 0 1
#
# plt.scatter(1.6, 0.8, c='b') # 1 0
# plt.scatter(1.2, 3.1, c='b')
#
# plt.scatter(4.7, 1.5, c='y') # 1 1
# var12

def drawPoints():
    plt.scatter(-3.7, -0.3, c='r') #0 0

    plt.scatter(-1.4, 2.8, c='g') # 0 1

    plt.scatter(0.6, -2.4, c='b') # 1 0
    plt.scatter(1.4, -2, c='b')
    plt.scatter(-3.7, -4.8, c='b')
    plt.scatter(0.5, -2, c='b')

    plt.scatter(2.4, 0, c='y') # 1 1
    plt.scatter(2.8, 1.6, c='y') # 1 1

    plt.grid(True)
    plt.show()


def initWeight():
    w = [[1, 0], [0, 0], [-1, 1]]
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j] = random.uniform(-2, 2)
    return w

def train(w):
    eps1 = 1.0
    eps2 = 1.0
    steps = 0
    while eps1 > 0 or eps2 > 0:
        eps1 = 0
        eps2 = 0
        for i in range(len(values)):
            net1 = 0.0
            net2 = 0.0
            for j in range(len(w)):
                net1 += w[j][0] * values[i][j]
                net2 += w[j][1] * values[i][j]
            if net1 >= 0:
                out1 = 1
            else:
                out1 = 0
            err1 = answers[i][0] - out1
            if (err1 != 0):
                eps1 += 1
                for k in range(len(w)):
                    coef = mu * values[i][k] * err1
                    w[k][0] += coef

            if net2 >= 0:
                out2 = 1
            else:
                out2 = 0
            err2 = answers[i][1] - out2
            if (err2 != 0):
                eps2 += 1
                for k in range(len(w)):
                    coef = mu * values[i][k] * err2
                    w[k][1] += coef
        steps += 1
        if steps > 500:
            print("Can not train perceptron")
            break
    print("steps: " + str(steps))
    return w

def draw(w, b):
    for i in range(len(values)):
        if answers[i][0] == 0:
            if answers[i][1] == 0: #0 0
                plt.scatter(values[i][1], values[i][2], c='r')
                plt.text(values[i][1], values[i][2], '0 0: (' + str(round(values[i][1], 2)) + ';'
                         + str(round(values[i][2], 2)) + ')')
            else: # 0 1
                plt.scatter(values[i][1], values[i][2], c='g')
                plt.text(values[i][1], values[i][2], '0 1: (' + str(round(values[i][1], 2)) + ';'
                         + str(round(values[i][2], 2)) + ')')
        else:
            if answers[i][1] == 0: # 1 0
                plt.scatter(values[i][1], values[i][2], c='b')
                plt.text(values[i][1], values[i][2], '1 0: (' + str(round(values[i][1], 2)) + ';'
                         + str(round(values[i][2], 2)) + ')')
            else: # 1 1
                plt.scatter(values[i][1], values[i][2], c='y')
                plt.text(values[i][1], values[i][2], '1 1: (' + str(round(values[i][1], 2)) + ';'
                         + str(round(values[i][2], 2)) + ')')
    x=np.arange(-5,5,0.02)
    plt.plot(x,(-w[0][0] - w[1][0] * x) / w[2][0])
    plt.text(-5, (-w[0][0] - w[1][0] * (-5)) / w[2][0], 'plot1: y=' + str(round(-w[1][0] / w[2][0], 2)) + 'x+'
             + str(round(-w[0][0] / w[2][0], 2)))
    plt.plot(x,(-w[0][1] - w[1][1] * x) / w[2][1])
    plt.text(-5, (-w[0][1] - w[1][1] * (-5)) / w[2][1], 'plot1: y=' + str(round(-w[1][1] / w[2][1], 2)) + 'x+'
             + str(round(-w[0][1] / w[2][1], 2)))

    if b == 1:
        check(w)
    plt.grid(True)
    plt.show()

def check(w):
    newPoints = [[1, random.uniform(-5, 5), random.uniform(-5, 5)], [1, random.uniform(-5, 5), random.uniform(-5, 5)],
                [1, random.uniform(-5, 5), random.uniform(-5, 5)], [1, random.uniform(-5, 5), random.uniform(-5, 5)],
                [1, random.uniform(-5, 5), random.uniform(-5, 5)]]
    for i in range(len(newPoints)):
        net1 = 0.0
        net2 = 0.0
        for j in range(len(w)):
            net1 += w[j][0] * newPoints[i][j]
            net2 += w[j][1] * newPoints[i][j]
        plt.scatter(newPoints[i][1], newPoints[i][2], c='m')
        # print("point: " + str(newPoints[i][1]) + ";" + str(newPoints[i][2]) + ",  net: " + str(net))
        if net1 >= 0:
            if net2 >= 0:
                plt.text(newPoints[i][1], newPoints[i][2], '1 1: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')
            else:
                plt.text(newPoints[i][1], newPoints[i][2], '1 0: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')
        else:
            if net2 >= 0:
                plt.text(newPoints[i][1], newPoints[i][2], '0 1: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')
            else:
                plt.text(newPoints[i][1], newPoints[i][2], '0 0: (' + str(round(newPoints[i][1], 2)) + ';' + str(round(newPoints[i][2], 2)) + ')')

def main():
    w = initWeight()
    train(w)
    print("final weights: ")
    for i in range(len(w)):
        for j in range(len(w[i])):
            print("i: " + str(i) + ", j: " + str(j) + ", w: " + str(w[i][j]))
    # set b = 1 to draw with checking random points
    b = 1
    draw(w, b)

main()