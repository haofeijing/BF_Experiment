import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = np.array([1.2, 3.4, 1.2])
unique_ele, count_ele = np.unique(a, return_counts=True)
print(np.asarray((unique_ele, count_ele)))
