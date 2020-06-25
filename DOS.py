import numpy as np
import torch
import torch.nn as nn
np.random.seed(234198)

import scipy.stats

class stock:
    def __init__(self, T, K, sigma, delta, So, r, N, M, d):
        self.T = T                      # ending time step
        self.K=K                        # price purchased
        self.sigma=sigma *np.ones(d)    # asset volatility
        self.delta=delta                # dividend yield
        self.So=So*np.ones(d)           # price at time 0
        self.r=r                        # marker risk free return
        self.N=N                        # number of time steps
        self.M=M                        # number of sample paths
        self.d=d                        # number of assets
    
    def GBM(self):
        """Return the price of d assets from time 0 to N in all M paths"""
        
        dt=self.T/self.N                # delta t
        So_vec=self.So*np.ones((1,S.M, S.d))    # initial price x_0 for each asset in each path

        # set Z value for each asset in each path at each time step
        Z=np.random.standard_normal((self.N,self.M, self.d))

        # calculate price for each asset at each time step from 1 to N in each path
        s=self.So*np.exp(np.cumsum((self.r-self.delta-0.5*self.sigma**2)*dt+self.sigma*np.sqrt(dt)*Z, axis=0))

        # all price from time 0 to N for each asset in each path
        s=np.append(So_vec, s, axis=0)
        return s
    
    
    def g(self,n,m,X):
        max1=torch.max(X[int(n),m,:].float()-self.K)
        
        return np.exp(-self.r*(self.T/self.N)*n)*torch.max(max1,torch.tensor([0.0]))
       

#%%
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1) 
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, 1)  
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.sigmoid(out)
        
        return out
    
def loss(y_pred,s, x, n, tau):
    r_n=torch.zeros((s.M))
    for m in range(0,s.M):
        
        r_n[m]=-s.g(n,m,x)*y_pred[m] - s.g(tau[m],m,x)*(1-y_pred[m])
    
    return(r_n.mean())
    
#%%

S=stock(3,100,0.2,0.1,90,0.05,9,5000,10)   

X=torch.from_numpy(S.GBM()).float()  # transform numpy array to tensor
#%%

def NN(n,x,s, tau_n_plus_1):
    epochs=50
    model=NeuralNet(s.d,s.d+40,s.d+40)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in range(epochs):
        F = model.forward(X[n])
        optimizer.zero_grad()
        criterion = loss(F,S,X,n,tau_n_plus_1)
        criterion.backward()
        optimizer.step()
    
    return F,model

mods=[None]*S.N                 # models (para) at each time step
tau_mat=np.zeros((S.N+1,S.M))
tau_mat[S.N,:]=S.N

f_mat=np.zeros((S.N+1,S.M))
f_mat[S.N,:]=1

#%%
for n in range(S.N-1,-1,-1):
    probs, mod_temp=NN(n, X, S,torch.from_numpy(tau_mat[n+1]).float())
    mods[n]=mod_temp
    np_probs=probs.detach().numpy().reshape(S.M)   # probs for each asset in each path at time n
    print(n, ":", np.min(np_probs)," , ", np.max(np_probs))

    f_mat[n,:]=(np_probs > 0.5)*1.0

    tau_mat[n,:]=np.argmax(f_mat, axis=0)

#%% 
Y=torch.from_numpy(S.GBM()).float()  # test data

tau_mat_test=np.zeros((S.N+1,S.M))
tau_mat_test[S.N,:]=S.N

f_mat_test=np.zeros((S.N+1,S.M))
f_mat_test[S.N,:]=1

V_mat_test=np.zeros((S.N+1,S.M))
V_est_test=np.zeros(S.N+1)

for m in range(0,S.M):
    V_mat_test[S.N,m]=S.g(S.N,m,Y)  # set V_N value for each path
    
V_est_test[S.N]=np.mean(V_mat_test[S.N,:])  # necessary?



for n in range(S.N-1,-1,-1):
    mod_curr=mods[n]
    probs=mod_curr(Y[n])
    np_probs=probs.detach().numpy().reshape(S.M)

    f_mat_test[n,:]=(np_probs > 0.5)*1.0

    tau_mat_test[n,:]=np.argmax(f_mat_test, axis=0)
    
    # calculate V_n for each path
    for m in range(0,S.M):
        V_mat_test[n,m]=np.exp((n-tau_mat_test[n,m])*(-S.r*S.T/S.N))*S.g(tau_mat_test[n,m],m,X) 
    # where is this formula from?

#%%
V_est_test=np.mean(V_mat_test, axis=1)
V_std_test=np.std(V_mat_test, axis=1)
V_se_test=V_std_test/(np.sqrt(S.M))

z=scipy.stats.norm.ppf(0.975)
lower=V_est_test[0] - z*V_se_test[0]
upper=V_est_test[0] + z*V_se_test[0]

print(V_est_test[0])
print(V_se_test[0])
print(lower)
print(upper)
