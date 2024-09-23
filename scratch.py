import math

import numpy as np
import torch

x = np.linspace(0, 1, 100)
x = x * np.pi - np.pi / 2
print(x)


y = [(math.sin(x) + 1) / 2 for x in x]

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()
