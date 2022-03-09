from scipy import interpolate
import numpy as np

class LinearInterpolation:
    f = None

    def __init__(self):
        return

    def fit(self, X, y, sample_weight=None):
        #calc distance matrix

        #add border points
        xx = X[:, 0]
        xx_linear_space = np.linspace(xx.min(), xx.max(), 5)
        xy = X[:, 1]
        xy_linear_space = np.linspace(xy.min(), xy.max(), 5)

        border = []

        for _, _x in enumerate(xx_linear_space):
            border.append(np.array([_x, xy_linear_space[0]]))
            border.append(np.array([_x, xy_linear_space[-1]]))
        for _, _y in enumerate(xy_linear_space):
            border.append(np.array([xx_linear_space[0], _y]))
            border.append(np.array([xx_linear_space[-1], _y]))

        border = np.array(border)
        values = np.zeros((border.shape[0]))

        X = np.concatenate((X,border))
        y = np.concatenate((y,values))

        #self.f = interpolate.interp2d(X[:,0], X[:,1], y, kind='cubic')
        self.f = interpolate.bisplrep(X[:,0], X[:,1], y)
        return

    def predict(self,X):
        #return np.array([self.f(X[i,0], X[i,1])[0] for i in range(len(X))])
        return np.array([interpolate.bisplev(X[i,0], X[i,1], self.f) for i in range(len(X))])

    def __str__(self):
        return "Linear interpolation"

    def get_params(self, deep=True):
        return {}
