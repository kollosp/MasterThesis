
from scipy import interpolate
import numpy as np
class NNInterpolation:
    def __init__(self):
        return

    def fit(self, X, y):
        #calc distance matrix
        self.f = interpolate.NearestNDInterpolator(X, y)
        return

    def predict(self,X):
        return np.array([self.f(X[i,0], X[i,1]) for i in range(len(X))])

    def __str__(self):
        return "Nearest neighbour"
