import torch
import numpy as np
import scipy.stats
import torch.nn as nn
np.random.seed(234198)
from matplotlib import pyplot as plt


class stock:
    def __init__(self, Ks, Kb, T, r):
        self.Ks = Ks
        self.Kb = Kb
        self.T = T
        self.r = r



    def g(self, n, m, X):
        return np.log(self.X[n, m, :]*(1 - self.Ks) + self.r*(self.T - n))


# Simulating geometric Brownian motion
def stock_sim_path(S, alpha, delta, sigma, T, N, n, M):
    """Simulates geometric Brownian motion."""
    h = T/n
    # uncomment below for deterministic trend. or, can pass it in as alpha as an array
    alpha = alpha # + np.linspace(0, 0.1, 500).reshape((n,N))
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N, M)).cumsum(axis = 0))




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


def loss(y_pred, s, x, n, tau):
    r_n = torch.zeros((s.M))
    for m in range(0, s.M):
        r_n[m] = -s.g(n, m, x) * y_pred[m] - s.g(tau[m], m, x) * (1 - y_pred[m])

    return (r_n.mean())


def NN(n, x, s, tau_n_plus_1):
    epochs = 50
    model = NeuralNet(s.d, s.d + 40, s.d + 40)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    F = None
    for epoch in range(epochs):
        F = model.forward(x[n])
        optimizer.zero_grad()
        criterion = loss(F, s, x, n, tau_n_plus_1)
        criterion.backward()
        optimizer.step()

    return F, model


def training():
    T = 2
    days = int(250*T)
    M = 50
    stock_path = stock_sim_path(100, 0.05, 0, .15, T, 1, days, M)

    X = torch.from_numpy(stock_path).float()
    print(X)
    # for m in range(M):
    #     plt.plot(np.arange(days), stock_path[:, :, m])
    # plt.show()

    mods = [None] * days  # models (para) at each time step
    tau_mat = np.zeros((days + 1, M))
    tau_mat[days, :] = days

    f_mat = np.zeros((days + 1, M))
    f_mat[days, :] = 1

    # %%
    for n in range(days - 1, -1, -1):
        probs, mod_temp = NN(n, X, S, torch.from_numpy(tau_mat[n + 1]).float())
        mods[n] = mod_temp
        np_probs = probs.detach().numpy().reshape(M)  # probs for each asset in each path at time n
        print(n, ":", np.min(np_probs), " , ", np.max(np_probs))

        f_mat[n, :] = (np_probs > 0.5) * 1.0

        tau_mat[n, :] = np.argmax(f_mat, axis=0)


if __name__ == "__main__":
    training()
