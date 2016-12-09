'''
Created on Dec 8, 2016

@author: daqingy
'''
from abc import abstractmethod
import numpy as np

class Kernel(object):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def calc(self,x,y):
        pass
    
class SquaredExponentialKernel(Kernel):
    """
    Squared exponential kernel
    """    
    def __init__(self, sigma, l):
        self.sigma = sigma
        self.l = l

    def calc(self, x, y):
        return self.sigma * np.exp( -0.5 * (1./self.l**2) * np.sum( (x-y)**2 ) )
    

def covariance(kernel, data1, data2):
    K = [kernel.calc(x,y) for x in data1 for y in data2]
    return np.reshape(K, (len(data1),len(data2))) 

def train(training_input, kernel, sigma = 0):
    # sigma = 0 means no additive noise
    training_data_num = len(training_input)
    mean = np.zeros(training_data_num)
    C = covariance(kernel,training_input, training_input) + sigma * np.eye(training_data_num)
    return (mean,C)

def predict(testing_input, training_input, kernel, C, training_output, simga = 0):
    # sigma = 0 means no additive noise
    testing_data_num = len(testing_input)
    k = covariance(kernel, testing_input, training_input)
    Cinv = np.linalg.inv(C)
    Mu = np.dot(np.dot(k,Cinv),training_output)
    Sigma = covariance(kernel, testing_input,testing_input) + simga * np.eye(testing_data_num) - np.dot(np.dot(k,Cinv),k)
    return (Mu,Sigma)



