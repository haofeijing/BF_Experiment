import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.plot([1, 2, 3])
plt.plot([4, 5, 6])
plt.legend(['small', 'large'])
plt.savefig('test.png')

