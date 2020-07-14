import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def f():
    plt.plot(np.random.rand(10))

for i in range(3):
    f()

plt.legend(['{}'.format(i) for i in range(3)])
plt.show()
