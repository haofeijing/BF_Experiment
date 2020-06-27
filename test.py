import numpy as np

A = np.random.randn(2, 1, 2)
shape = A.shape
B = np.pad(A, ((0, 0), (0, 0), (1, 0)), 'constant', constant_values=0)
print(B)
