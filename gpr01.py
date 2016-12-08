'''
Created on Dec 7, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt

class Kernel(object):
    """
    Kernel from Bishop's Pattern Recognition and Machine Learning pg. 307 Eqn. 6.63.
    """
    def __init__(self,*args):
        self.thetas = args

    def __call__(self,x,y):
        exponential = self.thetas[0] * np.exp( -0.5 * self.thetas[1] * np.sum( (x - y)**2 ) )
        linear = self.thetas[3] * np.dot(x,y)
        constant = self.thetas[2]
        return exponential + constant + linear

class OrnsteinKernel(object):
    """
    Ornstein-Uhlenbeck process kernel.
    """
    def __init__(self,theta):
        self.theta = theta

    def __call__(self,x,y):
        return np.exp(-self.theta * sum(abs(x-y)))

def covariance(kernel, data):
    return np.reshape([kernel(x,y) for x in data for y in data], (len(data),len(data)))

def draw_multivariate_gaussian(mean,C):
    ndim = len(mean)
    z = np.random.standard_normal(ndim)
    
    # Better numerical stabability than cholskey decomposition for
    # near-singular matrices C.
    [U,S,V] = np.linalg.svd(C)
    A = U * np.sqrt(S)

    return mean + np.dot(A,z)

def train(data,kernel):
    mean = np.zeros(len(data))
    C = covariance(kernel,data)
    return (mean,C)

def predict(x, data, kernel, C, t):
    """
    The prediction equations are from Bishop pg 308. eqns. 6.66 and 6.67.
    """

    k = [kernel(x,y) for y in data]
    Cinv = np.linalg.inv(C)
    m = np.dot(np.dot(k,Cinv),t)
    sigma = kernel(x,x) - np.dot(np.dot(k,Cinv),k)
    return (x,m,sigma)

#kernel = OrnsteinKernel(1.0)
kernel = Kernel(1.0, 64.0, 0.0, 0.0)

# Some sample training points.
xpts = np.random.rand(10) * 2 - 1

# In the context of Gaussian Processes training means simply
# constructing the kernel (or Gram) matrix.
(m,C) = train(xpts, kernel)

# Now we draw from the distribution to sample from the gaussian prior.
t = draw_multivariate_gaussian(m,C)

# Instead of regressing against some known function, lets just see
# what happens when we predict based on the sampled prior. This seems
# to be what a lot of other demo code does.

# Explore the results of GP regression in the target domain.
predictions = [predict(i,xpts,kernel,C,t) for i in np.arange(-1,1,0.01)]

plt.clf()
plt.hold(1)

x = np.array([prediction[0] for prediction in predictions])
y = np.array([prediction[1] for prediction in predictions])
sigma = np.array([prediction[2] for prediction in predictions])
plt.fill_between(x,y,y-sigma,y+sigma, alpha=0.2,facecolor='#0000FF')
plt.scatter(xpts, t, color='r', marker="o")

plt.show()