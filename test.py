import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = torch.tensor([[1/2, 2/3, 3/4, 4]])
b = torch.tensor([[5, 6, 7, 8]]).float()

print(a.type())
print(b.type())

