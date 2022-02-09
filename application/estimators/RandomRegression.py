import numpy as np
import random

class RandomRegression:
    def __init__(self):
        return

    def fit(self, X,y):
        return

    def predict(self, X):
        return np.array([random.uniform(0,1) for x in range(len(X))])

    def __str__(self):
        return "Random"