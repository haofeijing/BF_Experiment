import torch
import numpy as np
import scipy.stats
import torch.nn as nn
np.random.seed(234198)
from matplotlib import pyplot as plt


# Simulating geometric Brownian motion
def stock_sim_path(S, alpha, delta, sigma, T, N, n):
    """Simulates geometric Brownian motion."""
    h = T/n
    # uncomment below for deterministic trend. or, can pass it in as alpha as an array
    alpha = alpha # + np.linspace(0, 0.1, 500).reshape((n,N))
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))


class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1)
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.sigmoid(out)

        return out


def NN(n, x, s, tau_n_plus_1):
    epochs = 50
    model = NeuralNet(s.d, s.d + 40, s.d + 40)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        F = model.forward(X[n])
        optimizer.zero_grad()
        criterion = loss(F, S, X, n, tau_n_plus_1)
        criterion.backward()
        optimizer.step()

    return F, model


if __name__ == "__main__":
    T = 2
    days = int(250*T)
    stock_path = stock_sim_path(100, 0.05, 0, .15, T, 1, days)
    print(stock_path)
    # plt.plot(np.arange(days), stock_path)
    # plt.show()
