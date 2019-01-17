# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "Deepak"
__date__ = "$11 Jan, 2019 10:56:22 AM$"

import numpy as np
"""ADAptive LInear NEuron classifier.
Parameters
------------
eta : float
Learning rate (between 0.0 and 1.0)
n_iter : int
Passes over the training dataset.
Attributes
-----------
w_ : 1d-array
Weights after fitting.
errors_ : list
Number of misclassifications in every epoch."""
class AdalineGD(object):
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
      
    """Fit training data.
    Calculate the errors on entire dataset to minimize cost function by adjusting weights
Parameters
----------
X : {array-like}, shape = [n_samples, n_features]
Training vectors, where n_samples
is the number of samples and
n_features is the number of features.
y : array-like, shape = [n_samples]
Target values.

Returns
-------
self : object"""
    def fit(self, X, Y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
            
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (Y - output)
            self.w_[1:] += self.eta * (X.T.dot(errors))
            self.w_[0] += self.eta * errors.sum()
            cost = ((errors**2).sum())/2
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        """Calculate net input
        Dot product of 2 arrays"""
        return np.dot(X, self.w_[1:])+self.w_[0]
        
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X)>= 0.01, 1, -1)

# Standardization feature scaling by converting standard mean of features to 0 and standard deviation to 1
def feature_scaling(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i]-X[:,i].mean())/X[:,i].std()
    return X_std
#Loading Iris dataset directly from UCI machine learning repository into a dataframe object
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', -1, 1)
X=df.iloc[0:100, [0,2]].values

eta = 0.01
#X_std = feature_scaling(X)
X_std = X
while(eta > 0.0002):
    ada = AdalineGD(eta=eta, n_iter=20)
    ada.fit(X_std, y)
    plt.plot(range(1, len(ada.cost_)+1), np.log10(ada.cost_), marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('log(sum-squared-errors)')
    plt.title('Adaline - Learning rate {0} with data scaling'.format(eta))
    plt.show()
    eta -= 0.001