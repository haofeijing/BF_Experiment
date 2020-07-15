import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = torch.tensor([1, 2, 3, 4]).float().cuda(dev)
b = a > 2

print(b)

c = np.array([1, 2, 3, 4])
d = c > 2
print(d)
