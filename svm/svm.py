# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:35:41 2021

@author: Ding Chi
"""

import numpy as np

class SVM:
    
    def __init__(self, learning_rate = 0.001, lambda_param=0.01, n_iters=1000):
        
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
        
    def fit(self, X, y):
        # set labels as -1 or +1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * ((x_i @ self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param  * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        # computes the hyper-plane
        linear_output = X @ self.w - self.b
        
        # binary classification (+1 or -1)
        return np.sign(linear_output)
    