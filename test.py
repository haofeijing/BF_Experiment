import numpy as np
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


x = np.exp(3)
y = torch.tensor(3).float().cuda(dev)
print(x * y)


