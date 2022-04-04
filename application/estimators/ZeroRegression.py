import numpy as np
import random

class ZeroRegression:
    def __init__(self):
        return

    def fit(self, X,y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


    def __str__(self):
        return "Always Zeros"

    def get_params(self, deep=True):
        return {}