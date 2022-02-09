from scipy import interpolate
import numpy as np

class LinearInterpolation:
    f = None

    def __init__(self):
        return

    def fit(self, X, y):
        #calc distance matrix
        self.f = interpolate.interp2d(X[:,0], X[:,1], y, kind='cubic')
        return

    def predict(self,X):
        return np.array([self.f(X[i,0], X[i,1])[0] for i in range(len(X))])

    def __str__(self):
        return "Linear interpolation"
