
from scipy import interpolate
import numpy as np

class KMeansInterpolation:
    def __init__(self, k_mean=5):
        self.k_mean = k_mean
        return

    def fit(self, X, y, sample_weight=None):
        #calc distance matrix
        self.X = X
        self.y = y
        return self

    def predict(self,X):
        ret = np.zeros((X.shape[0]))
        for i,_ in enumerate(X):
            #distances = np.array([[np.linalg.norm(xx-X[i]),index] for xx,index in enumerate(self.X)],dtype=[("mag", float),("index", int)])
            #
            distances = np.array([(np.linalg.norm(xx-X[i]),index) for index,xx in enumerate(self.X)],dtype=[("mag", float),("index", int)])
            distances = np.sort(distances, order="mag")
            indexes = [x[1] for x in distances[0:self.k_mean]]
            #print(indexes, self.y)
            ret[i] = np.mean(self.y[indexes])


        return ret

    def __str__(self):
        return "K(" + str(self.k_mean) + ") regression"

    def get_params(self, deep=True):
        return {}

    def set_params(self):
        return {}