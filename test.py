import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
import time

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])
print(1 + a)


