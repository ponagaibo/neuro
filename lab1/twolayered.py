import random
import matplotlib.pyplot as plt
import numpy as np

values = [[1, -1.5, -0.6], [1, 4.6, -4.6], [1, 4.7, -3.2], [1, 1.6, 0.8],
          [1, 1.7, -1.4], [1, 1.2, 3.1], [1, -4.9, -4.2], [1, 4.7, 1.5]]
answers = [0, 0, 1, 1, 1, 1]
mu = 0.3

plt.scatter(3.6, 1.3, c='r') #0 0
plt.scatter(1, -1.2, c='r')
plt.scatter(2.2, -1.3, c='r')
plt.scatter(3.4, 2.3, c='r')

plt.scatter(-1.5, 4.9, c='g') # 0 1

plt.scatter(-3.6, -4.8, c='b') # 1 0
plt.scatter(-0.8, -3.2, c='b')

plt.scatter(-2.8, 1.5, c='y') # 1 1

plt.grid(True)
plt.show()

