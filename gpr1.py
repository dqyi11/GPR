'''
Created on Dec 8, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from GP import *


if __name__ == '__main__':
    
    sigma_n = 0.3 # noise variance
    f = 0 # f = 0 (turn off periodicity)
    X = np.array([-1.5, -1, -0.75, -0.4, -0.25, 0])
    y = 0.55 * np.array([-3, -2, -0.6, -0.4, 1, 1.6])
    
    #vars = [1, 0, 1, 0] # 0 means "known/given"
    
    kernel = SquaredExponentialKernel(sigma = 0.3, l = 0.2)
    
    (m,C) = train(X, kernel, sigma_n)
    
    Xstar = np.arange(-1,1,0.01)
    (Mu, Sigma) = predict(Xstar, X, kernel, C, y, sigma_n) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(X, y, yerr=sigma_n*np.ones(len(y)), color='r', fmt='o')  
    ax.fill_between(Xstar, Mu,Mu-Sigma,Mu-Sigma, alpha=0.2,facecolor='#0000FF')
              
    plt.show()