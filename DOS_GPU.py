import numpy as np
import torch
import torch.nn as nn
import scipy.stats
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

np.random.seed(234198)
torch.manual_seed(234198)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class stock:
    def __init__(self, T, K, sigma, delta, So, r, N, M, d):
        self.T = T                      # ending time step
        self.K=K                        # price purchased
        self.sigma=sigma*torch.ones(d).cuda(dev)    # asset volatility
        self.delta=delta                # dividend yield
        self.So=So*torch.ones(d).cuda(dev)           # price at time 0
        self.r=r                        # marker risk free return
        self.N=N                        # number of time steps
        self.M=M                        # number of sample paths
        self.d=d                        # number of assets

    def GBM(self):
        """Return the price of d assets from time 0 to N in all M paths"""

        dt=self.T/self.N                # delta t
        So_vec=self.So*torch.ones((1,S.M, S.d)).cuda(dev)    # initial price x_0 for each asset in each path

        # set Z value for each asset in each path at each time step

        Z = torch.normal(0, 1, (self.N,self.M, self.d)).cuda(dev)

        # calculate price for each asset at each time step from 1 to N in each path
        s=self.So*torch.exp(torch.cumsum((self.r-self.delta-0.5*self.sigma**2)
                                         *dt+self.sigma*torch.sqrt(torch.tensor(dt).float().cuda(dev))*Z, dim=0))

        # all price from time 0 to N for each asset in each path
        s = torch.cat((So_vec, s), dim=0)
        return s


    def g(self,n,m,X):
        max1=torch.max(X[int(n),m,:].float()-self.K)
        tmp1 = torch.exp(torch.tensor(-self.r*(self.T/self.N)*int(n)).cuda(dev))
        tmp2 = torch.max(max1,torch.tensor([0.0]).cuda(dev))
        return tmp1 * tmp2




#%%
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2, q3):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1).cuda(dev)
        self.relu = nn.ReLU().cuda(dev)
        self.a2 = nn.Linear(q1, q2).cuda(dev)
        self.a3 = nn.Linear(q2, q3).cuda(dev)
        self.a4 = nn.Linear(q3, 1).cuda(dev)   # extra layer
        self.sigmoid=nn.Sigmoid().cuda(dev)


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

def loss(y_pred,s, x, n, tau):
    r_n=torch.zeros((s.M)).cuda(dev)
    for m in range(0,s.M):

        r_n[m]=-s.g(n,m,x)*y_pred[m] - s.g(tau[m],m,x)*(1-y_pred[m])

    return(r_n.mean())

#%%

S=stock(3,100,0.2,0.1,90,0.05,9,5000,10)
X=S.GBM() # training data

Y=S.GBM()  # test data
#%%

tau_mat_test=torch.zeros((S.N+1,S.M)).cuda(dev)
tau_mat_test[S.N,:]=S.N

f_mat_test=torch.zeros((S.N+1,S.M)).cuda(dev)
f_mat_test[S.N,:]=1

V_mat_test=torch.zeros((S.N+1,S.M)).cuda(dev)
V_est_test=torch.zeros(S.N+1).cuda(dev)

for m in range(0,S.M):
    V_mat_test[S.N,m]=S.g(S.N,m,Y)  # set V_N value for each path




tau_mat=torch.zeros((S.N+1,S.M)).cuda(dev)
tau_mat[S.N,:]=S.N

f_mat=torch.zeros((S.N+1,S.M)).cuda(dev)
f_mat[S.N,:]=1



def NN(n,x,s, tau_n_plus_1):
    epochs=100
    model=NeuralNet(s.d,s.d+40,s.d+40,s.d+40).cuda(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        F = model.forward(x[n])
        optimizer.zero_grad()
        criterion = loss(F,S,x,n,tau_n_plus_1)
        criterion.backward()
        optimizer.step()

        # training loss
        train_losses.append(criterion.item())

        # validation loss
        model.eval()
        pred_y = model(Y[n])

        eval_losses.append(loss(pred_y, S, Y, n, tau_mat_test[n+1]))


    plt.clf()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(eval_losses)), eval_losses)
    plt.title('loss_{}'.format(n))
    plt.legend(['train', 'validation'])
    plt.savefig('1e-4/loss_{}'.format(n))


    return F,model


#%%
for n in range(S.N-1,-1,-1):
    probs, mod_temp=NN(n, X, S,tau_mat[n+1])

    print(n, ":", torch.min(probs).item()," , ", torch.max(probs).item())

    f_mat[n,:]=(probs.view(S.M) > 0.5)*1.0

    tau_mat[n,:]=torch.argmax(f_mat, dim=0)

    # prediction
    pred = mod_temp(Y[n])
    f_mat_test[n, :] = (pred.view(S.M) > 0.5) * 1.0

    tau_mat_test[n, :] = torch.argmax(f_mat_test, dim=0)
    for m in range(0,S.M):
        V_mat_test[n,m]=torch.exp((n-tau_mat_test[n,m])*(-S.r*S.T/S.N))*S.g(tau_mat_test[n,m],m,Y)
#%%



#%%
V_est_test=torch.mean(V_mat_test, dim=1)
V_std_test=torch.std(V_mat_test, dim=1)
V_se_test=V_std_test/(torch.sqrt(torch.tensor(S.M).float().cuda(dev)))

z=scipy.stats.norm.ppf(0.975)
lower=V_est_test[0] - z*V_se_test[0]
upper=V_est_test[0] + z*V_se_test[0]

print(V_est_test[0].item())
print(V_se_test[0].item())
print(lower.item())
print(upper.item())
