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
        
    def fit(self,n_iters = 10):
        for _ in range(n_iters):
            # first assign class to data points
            D = np.linalg.norm(self.X[:,None,:] - self.clusters,2,-1)
            index = np.argmin(D,-1).reshape(-1,1)
            
            for i in range(self.n_clusters):
                mask = index == i
                self.clusters[i] = np.sum(self.X * mask,0)/np.sum(mask)
            
    def predict(self,x):
        return np.argmin(np.linalg.norm(self.clusters - x,2,-1))
            
# k = Kmeans(2, X)
# k.fit()
# plt.scatter(X[:,0],X[:,1])
# for i in range(n):
#     plt.scatter(k.clusters[i][0],k.clusters[i][1])
    
    
import cv2
img = cv2.imread('camera-man.png',0)
h0,w0 = img.shape

"""
# it is like normalization, if the image size is too large, spatial coordinates 
# will take over in the 2 norm instead of intensity
# But simply normalize without resizing makes computation heavy, therefore the 
# image resize idea.
"""
ratio = .3
img = cv2.resize(img,(int(img.shape[0]*ratio),int(img.shape[1]*ratio)))
h,w = img.shape
X = np.zeros((h*w,3))
for i in range(h):
    for j in range(w):
        X[i*w + j] = np.array([img[i,j],i,j])

n_clusters = 3
color = np.array([np.random.randint(0,255,3).tolist() for i in range(n_clusters)])
k = Kmeans(n_clusters, X)
k.fit(50)

mask = np.zeros(img.shape+(3,)).astype(np.uint8)
for i in range(h):
    for j in range(w):
        mask[i,j] = color[k.predict(np.array([img[i,j],i,j]))]
        
mask = cv2.resize(mask,(h0,w0),interpolation=1)
plt.imshow(mask)