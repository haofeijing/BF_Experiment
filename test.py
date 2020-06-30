import numpy as np

def f(x, y):
    return x + 1

def g(x, h):
    return h(x, 1)

print(np.max([np.nan]) * 1)
