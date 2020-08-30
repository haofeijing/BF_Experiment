import numpy as np
import torch
import torch.nn as nn
import scipy.stats
import scipy.stats as si
import matplotlib.pyplot as plt
import time
import sys


np.random.seed(234198)
torch.manual_seed(234198)

# dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

days = 500
path = "European/days={}".format(days)
# path = "European/test"

print("start")
# path for txt file
sys.stdout = open("{}/output.txt".format(path), "w")


class stock:
    def __init__(self, T, K, sigma, delta, So, r, N, M, d):
        self.T = T                      # ending time step
        self.K=K                        # price purchased
        self.sigma=sigma*torch.ones(d).to(dev)    # asset volatility
        self.delta=delta                # dividend yield
        self.So=So*torch.ones(d).to(dev)           # price at time 0
        self.r=r                        # marker risk free return
        self.N=N                        # number of steps for stopping times
        self.M=M                        # number of training sample paths
        self.d=d                        # number of assets

    def GBM(self, days, alpha):
        # """Return the price of d assets from time 0 to N in all M paths"""
        #
        # dt=self.T/days               # delta t
        # So_vec=self.So*torch.ones((1,self.M, self.d)).to(dev)    # initial price x_0 for each asset in each path
        #
        # # set Z value for each asset in each path at each time step
        #
        # Z = torch.normal(0, 1, (days,self.M, self.d)).to(dev)
        #
        # # calculate price for each asset at each time step from 1 to N in each path
        # s=self.So*torch.exp(torch.cumsum((self.r-self.delta-0.5*self.sigma**2)
        #                                  *dt+self.sigma*torch.sqrt(torch.tensor(dt).float().to(dev))*Z, dim=0))
        #
        # # all price from time 0 to N for each asset in each path
        # s = torch.cat((So_vec, s), dim=0)

        """Simulates geometric Brownian motion."""
        h = self.T / days
        # uncomment below for deterministic trend. or, can pass it in as alpha as an array
        alpha = alpha # + np.linspace(0, 0.1, 500).reshape((n,N))
        mean = (alpha - self.delta - .5 * self.sigma ** 2) * h
        vol = self.sigma * h ** .5
        return self.So * torch.exp((mean + vol * torch.randn(days, self.M, self.d).to(dev)).cumsum(dim=0))
        # return s


    def g(self,t_n,m,X):
        max1=torch.max(X[int(t_n),m,:].float()-self.K)
        tmp1 = torch.exp(torch.tensor(-self.r*int(t_n) * (self.T/(days - 1))).to(dev))
        tmp2 = torch.relu(max1)
        return tmp1 * tmp2



#%%
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2, q3):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1)
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, q3)
        self.a4 = nn.Linear(q3, 1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.relu(out)
        out = self.a4(out)
        out = self.sigmoid(out)

        return out

T_n = [0, days-1]
N = len(T_n)-1
alpha = 0.05

def loss(y_pred,s, x, t_n, tau):
    g_n = torch.from_numpy(np.fromiter((s.g(t_n,m,x) for m in range(0,s.M)), float)).to(dev)
    g_tau = torch.from_numpy(np.fromiter((s.g(T_n[int(tau[m])],m,x) for m in range(0,s.M)), float)).to(dev)
    r_torch = - g_n * y_pred.view(-1) - g_tau * (1 - y_pred.view(-1))

    return r_torch.mean()


def bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    delta = si.norm.cdf(d1)
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call, delta


#%%

T = 2
K = 105
S0 = 100
r = 0.05
sigma = 0.2
delta = 0
M = 20000
M_test = 20000
d = 1


S=stock(T,K,sigma,delta,S0,r,N,M,d)
X=S.GBM(days=days, alpha=alpha) # training data

S_test = stock(T,K,sigma,delta,S0,r,N,M_test,d)
Y=S_test.GBM(days=days, alpha=alpha)  # test data

# print(Y)

# analytic solution
call, delta = bs(S0, K, T, r, sigma)
print(call)

#%%
# initialization for test data
tau_mat_test=torch.zeros((S_test.N+1,S_test.M)).to(dev)
tau_mat_test[S_test.N,:]=S_test.N

f_mat_test=torch.zeros((S_test.N+1,S_test.M)).to(dev)
f_mat_test[S_test.N,:]=1

V_mat_test=torch.zeros((S_test.N+1,S_test.M)).to(dev)
V_est_test=torch.zeros(S_test.N+1).to(dev)

for m in range(0,S_test.M):
    V_mat_test[S_test.N,m]=S_test.g(T_n[S_test.N],m,Y)  # set V_N value for each path

# print(V_mat_test[S_test.N, :])

# initialization for training
tau_mat=torch.zeros((S.N+1,S.M)).to(dev)
tau_mat[S.N,:]=S.N

f_mat=torch.zeros((S.N+1,S.M)).to(dev)
f_mat[S.N,:]=1



def NN(t_n,x,s, tau_n_plus_1):
    epochs=100
    model=NeuralNet(s.d,s.d+40,s.d+40,s.d+40).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        F = model.forward(x[t_n])
        optimizer.zero_grad()
        criterion = loss(F,s,x,t_n,tau_n_plus_1)
        criterion.backward()
        optimizer.step()

        # training loss
        train_losses.append(criterion.item())

        # validation loss
        model.eval()
        pred_y = model(Y[t_n])

        eval_losses.append(loss(pred_y, S_test, Y, t_n, tau_mat_test[n+1]))


    plt.clf()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(eval_losses)), eval_losses)
    plt.title('loss_{}'.format(n))
    plt.legend(['train', 'validation'])
    plt.savefig('{}/loss_{}.png'.format(path, n))


    return F,model


#%%
for n in range(S.N-1,-1,-1):
    probs, mod_temp=NN(T_n[n], X, S,tau_mat[n+1])

    print(n, ":", torch.min(probs).item()," , ", torch.max(probs).item())

    f_mat[n,:]=(probs.view(S.M) > 0.5)*1.0

    tau_mat[n,:]=torch.argmax(f_mat, dim=0)

    # prediction
    pred = mod_temp(Y[T_n[n]])
    f_mat_test[n, :] = (pred.view(S_test.M) > 0.5) * 1.0

    tau_mat_test[n, :] = torch.argmax(f_mat_test, dim=0)
    for m in range(0,S_test.M):
        # V_mat_test[n,m]=np.exp((T_n[n]-T_n[int(tau_mat_test[n,m])])*(-S_test.r*S_test.T/S_test.N))*S_test.g(tau_mat_test[n,m],m,Y)
        V_mat_test[n,m]=S_test.g(T_n[int(tau_mat_test[n,m])],m,Y)

    torch.cuda.empty_cache()

# plt.title('loss')
# plt.legend(['n = {}'.format(i) for i in range(S.N-1, -1, -1)])
# plt.savefig('smeilogy.png')

#%%


#%%
V_est_test=torch.mean(V_mat_test, dim=1)
V_std_test=torch.std(V_mat_test, dim=1)
V_se_test=V_std_test/(torch.sqrt(torch.tensor(S_test.M).float().to(dev)))

z=scipy.stats.norm.ppf(0.975)
lower=V_est_test[0] - z*V_se_test[0]
upper=V_est_test[0] + z*V_se_test[0]

print(V_est_test[0].item())
print(V_se_test[0].item())
print(lower.item())
print(upper.item())

# plt.clf()
# decisions = tau_mat_test[0, :]
# values, counts = torch.unique(decisions, return_counts=True)
# print(values, counts, sep='\n')



sys.stdout.close()


