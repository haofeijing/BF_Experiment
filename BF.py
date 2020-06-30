import torch
import numpy as np
import scipy.stats
import torch.nn as nn

np.random.seed(234198)
from matplotlib import pyplot as plt


class stock:
    def __init__(self, Ks, Kb, T, r, d, M):
        self.Ks = Ks
        self.Kb = Kb
        self.T = T              # number of total time steps
        self.r = r              # risk-free interest rate
        self.d = d              # number of stocks
        self.M = M              # number of sample paths

    def g(self, n, m, X, *args):
        return np.log(X[int(n), :, m] * (1 - self.Ks) + self.r * (self.T - n))

    def g_t(self, n, m, X, V, t):
        return self.r * (n - t) - np.log(X[int(n), :, m] * (1 + self.Kb)) + \
                np.max(V[t, int(n):, m])


    def h(self, n, m, X, V, t):
        return np.log(X[int(n), :, m] * (1 - self.Ks)) \
               + np.max(V[t, int(n):, m])

    # Simulating geometric Brownian motion
    def stock_sim_path(self, S, alpha, delta, sigma, T):
        """Simulates geometric Brownian motion."""
        h = T / self.T
        # uncomment below for deterministic trend. or, can pass it in as alpha as an array
        alpha = alpha  # + np.linspace(0, 0.1, 500).reshape((n,N))
        mean = (alpha - delta - .5 * sigma ** 2) * h
        vol = sigma * h ** .5
        cumsum = (mean + vol * np.random.randn(self.T, self.d, self.M)).cumsum(axis=0)
        return S * np.exp(np.pad(cumsum, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=0))


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


def loss(y_pred, s, x, n, tau, f, V, t):
    r_n = torch.zeros(s.M)
    for m in range(0, s.M):
        r_n[m] = -f(n, m, x, V, t) * y_pred[m] - f(tau[m], m, x, V, t) * (1 - y_pred[m])

    return r_n.mean()


def NN(n, x, s, tau_n_plus_1, f, V, t):
    epochs = 50
    model = NeuralNet(s.d, s.d + 10, s.d + 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    F = None
    for epoch in range(epochs):
        F = model.forward(x[n].T)
        optimizer.zero_grad()
        criterion = loss(F, s, x, n, tau_n_plus_1, f, V, t)
        criterion.backward()
        optimizer.step()

    return F, model


def getTauAndV(S, X, f, V, t):
    mods = [None] * S.T  # models (para) at each time step
    tau_mat = np.zeros((S.T + 1, S.M))
    tau_mat[S.T, :] = S.T

    f_mat = np.zeros((S.T + 1, S.M))
    f_mat[S.T, :] = 1

    V_mat_test = np.zeros((S.T + 1, S.M))
    # V_est_test = np.zeros(S.T + 1)

    for m in range(0, S.M):
        V_mat_test[S.T, m] = f(S.T, m, X, V, t)  # set V_T value for each path

    # %%
    for n in range(S.T - 1, t - 1, -1):
        probs, mod_temp = NN(n, X, S, torch.from_numpy(tau_mat[n + 1]).float(), f, V, t)
        mods[n] = mod_temp
        np_probs = probs.detach().numpy().reshape(S.M)  # probs for each asset in each path at time n
        print(n, ":", np.min(np_probs), " , ", np.max(np_probs))

        f_mat[n, :] = (np_probs > 0.5) * 1.0

        tau_mat[n, :] = np.argmax(f_mat, axis=0)

        for m in range(0, S.M):
            V_mat_test[n, m] = f(n, m, X, V, t)

    return tau_mat, V_mat_test, mods

def getValueForAllTime(S, f, initValue=None):
    VForAllTime = [None] * S.T
    tauForAllTime = [None] * S.T
    modsForAllTime = [None] * S.T
    for time in range(S.T - 1, -1, -1):
        tau, V, mods = getTauAndV(S, X, f, V=initValue, t=time)
        print(tau)
        print(V)
        VForAllTime[time] = V
        tauForAllTime[time] = tau
        modsForAllTime[time] = mods
    VforAllTime = np.array(VForAllTime)
    tauForAllTime = np.array(tauForAllTime)
    return tauForAllTime, VforAllTime, modsForAllTime


if __name__ == "__main__":
    T = 2
    # days = int(10 * T) - 1
    days = 10
    M = 100      # number of sample paths
    d = 1        # number of stocks
    S = stock(0, 0.02, days, 0.05, d, M)
    stock_path = S.stock_sim_path(100, 0.05, 0, .15, T)
    X = torch.from_numpy(stock_path).float()

    # number of buying decisions, assume initial state i = 0, flat
    N_tau = 1

    # store all tau
    allDecisions = []
    allMods = []

    # training
    tau, V, mods = getValueForAllTime(S, S.g)
    allDecisions.append(tau)
    allMods.append(mods)

    for n in range(N_tau - 1):
        tau, V, mods = getValueForAllTime(S, S.g_t, V)
        allDecisions.append(tau)
        allMods.append(mods)

        tau, V, mods = getValueForAllTime(S, S.h, V)
        allDecisions.append(tau)
        allMods.append(mods)

    tau, V, mods = getValueForAllTime(S, S.g_t, V)
    allDecisions.append(tau)
    allMods.append(mods)


    # printing results for training data
    for m in range(S.M):
        print('Decisions for sample {}:'.format(m))
        decision = 0
        for decisions in allDecisions:
            if decision >= S.T:
                break
            decision = int(decisions[decision, decision+1, m])
            print(decision, end=' ')
        print()


