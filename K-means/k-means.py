# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:18:36 2021

@author: m1390
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


n = 2
N = 100
X,Y = make_blobs(n_samples = N,centers = n,cluster_std=0.6)

class Kmeans(object):
    
    def __init__(self,n_clusters,X):
        self.n_clusters = n_clusters
        self.X = X
        
        self.clusters = self.X[np.random.randint(0,X.shape[0],self.n_clusters)]
        
    def fit(self,n_iters = 50):
        for _ in range(n_iters):
            # first assign class to data points
            D = np.linalg.norm(self.X[:,None,:] - self.clusters,2,-1)
            index = np.argmin(D,-1).reshape(-1,1)
            
            for i in range(self.n_clusters):
                mask = index == i
                self.clusters[i] = np.sum(self.X * mask,0)/np.sum(mask)
            
            
            
k = Kmeans(2, X)
k.fit()
plt.scatter(X[:,0],X[:,1])
for i in range(n):
    plt.scatter(k.clusters[i][0],k.clusters[i][1])