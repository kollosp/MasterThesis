import numpy as np
import random

class MeanRegression:
    def __init__(self):
        return

    def fit(self, X,y, sample_weight=None):
        self.y = y
        return

    def predict(self, X):
        d = np.ones(len(X))
        d.fill(self.y.mean())
        return d

    def __str__(self):
        return "Mean regression"

    def get_params(self, deep=True):
        return {}