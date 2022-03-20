import numpy as np
import random
from math import sqrt, exp, pi
from sklearn.base import BaseEstimator

class GravityRegression(BaseEstimator):
    def __init__(self, bandwidth=None, func="exponential", gaussian_bandwidth=4):
        self.data_set = None
        self.data_values = None
        self.sum = None
        self.bandwidth = bandwidth
        self.func = func
        self.gaussian_bandwidth = gaussian_bandwidth
        self.count = None

        return

    def get_params(self, deep=True):
        return {
            "gaussian_bandwidth": self.gaussian_bandwidth,
            "func": self.func,
            "bandwidth": self.bandwidth
        }

    def euclidean_distance(self, x1,x2):
        # euclidean distance of the given point and all datasets calculation
        return ((x2-x1)**2).sum()

    def fit(self, X,y, sample_weight=None):
        self.data_set = X
        self.data_values = y
        #normalisation
        self.sum = np.sum(y)
        self.count = y.shape[0]
        return

    def predict(self, X):
        #for each point
        ret = np.zeros([X.shape[0]])
        for i in range(len(X)):
            sum = 0
            mean_distance = 0
            count = 0
            for j in range(len(self.data_values)):
                d_j = self.euclidean_distance(X[i], self.data_set[j])
                #consider only
                if self.bandwidth is None or d_j < self.bandwidth :
                    #weight: 1/(1-x)^2
                    #check also sqrt(2*pi)*exp(-x^2/2)
                    sum += self.data_values[j]
                    if self.func == "homographic":
                        d_j = 1 / (1+d_j/self.gaussian_bandwidth)**2
                    elif self.func == "exponential":
                        d_j = exp(-(d_j/self.gaussian_bandwidth)**2/2)

                    count += 1
                    mean_distance += d_j
                    ret[i] = ret[i] + (self.data_values[j] * d_j)

            if mean_distance > 0 and count > 1:
                #mean_distance /= count
                ret[i] = ret[i] / mean_distance
                pass

        return ret

    def w_predict(self, X):
        # for each point
        ret = np.zeros([X.shape[0]])
        for i in range(len(X)):
            sum = 0
            mean_distance = 0
            count = 0
            for j in range(len(self.data_values)):
                d_j = self.euclidean_distance(X[i], self.data_set[j])
                # consider only
                if self.bandwidth is None or d_j < self.bandwidth:
                    sum += self.data_values[j]
                    d_j += 1
                    d_j = d_j ** 2
                    count+=1
                    mean_distance += d_j
                    d_j = 1/d_j
                    ret[i] = ret[i] * 1/d_j

            if sum > 0:

                # mean_distance /= count
                ret[i] /= count
                pass
        return ret


    def __str__(self):
        if self.bandwidth is not None:
            return "Gravity b = " + ("%.2f" % self.bandwidth) +" " + self.func
        else:
            return "Gravity " + self.func