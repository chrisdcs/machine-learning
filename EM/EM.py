# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:54:08 2021

A demo for EM
And Anderson Acceleration applied for EM

@author: Ding Chi
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.GMM import GMM
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal as norm

#%% Generate dataset by GMM

# the demo only has n clusters
k = 3
D = 2
N = 300

mu = np.array([[10,3],
               [1,1],
               [5,4]])

cov = np.ndarray((3),dtype=object)
cov[0] = np.eye(2)

cov[1] = np.array([[1.5,0],
                    [0,1.5]])

cov[2] = np.eye(2) * 2


# generate data
# mixing coefficients pi for GMM

pi = np.array([1/3,1/3,1/3])
gmm = GMM(mu,cov,pi)
X = gmm.sample(N)
real_gaussian = [norm(mu[i],cov[i]) for i in range(k)]

# 0. Create dataset
# Stratch dataset to get ellipsoid 

# X,Y = make_blobs(cluster_std=1.5,random_state=20,n_samples=N,centers=k)
# X = np.dot(X,np.random.RandomState(0).randn(2,2))

# k=5
# X,Y = make_blobs(cluster_std=1,random_state=10,n_samples=N,centers=k)
# X = np.dot(X,np.random.RandomState(0).randn(2,2))

# k = 4
# X,Y = make_blobs(n_samples = N,centers = 4,cluster_std=0.6,random_state=0)

#%% EM algorithm

# initialize random guess for mean
# mu_guess = X[np.random.choice(X.shape[0],k,False)]
mu_guess = np.array([[3,5],
                     [2,0.4],
                     [4,3]])

# initialize random guess for covariance matrix
shape = k,D,D

# cov_guess = np.full(shape,np.cov(X,rowvar=False))
cov_guess = np.full(shape,np.eye(2))
for i in range(k):
    cov_guess[i] = np.eye(D)

# initial guess for mixing coefficients
pi_guess = 1/k * np.ones((k))

# guess distribution functions
guess = [norm(mu_guess[i],cov_guess[i]) for i in range(k)]

# log likelihood evaluation
likelihood = np.zeros((k,N))
for i in range(k):
    likelihood[i] = guess[i].pdf(X)

log = np.sum(np.log(likelihood.T @ pi_guess))

log_list = []
log_list.append(log)
#%% start the EM step
while True:
    
    # E step
    marginal = (likelihood.T @ pi_guess).reshape(-1,1)
    # marginal += (marginal < 1e-20) * 1e-20
    marginal[marginal==0] += 1e-20
    
    posterior = likelihood.T * pi_guess/marginal
    # if (posterior == np.nan).any():
    #     posterior = np.nan_to_num(posterior)
    
    # M step
    Nk = posterior.sum(0)
    
    # update mean: the fixed point iteration
    for i in range(k):
        mu_guess[i] = 1/Nk[i] * posterior[:,i] @ X


    Xmu = np.ndarray((k),dtype=object)
    for i in range(k):
        Xmu[i] = X - mu_guess[i]

    
    # update covariance matrix: the fixed point iteration
    for i in range(k):
        cov_guess[i] = 1/Nk[i] * np.sum(posterior[:,i].reshape(-1,1,1) * 
                                        Xmu[i][:,:,None] @ Xmu[i][:,None,:],0)
    
    pi_guess = Nk/N
    
    for i in range(k):
         guess[i].mean = mu_guess[i]
         guess[i].cov = cov_guess[i]

    
    for i in range(k):
        likelihood[i] = guess[i].pdf(X)

    log = np.sum(np.log(likelihood.T @ pi_guess))
    log_list.append(log)
    
    if np.array(log_list[-10:]).std() < 1e-1:
        break
    
#%%
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T
        
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.scatter(X[:,0],X[:,1])
for m,c in zip(mu_guess,cov_guess):
    multi_normal = norm(mean=m,cov=c)
    ax.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    ax.set_title('Final state')

plt.show()
#%%
# plt.plot(log_list)