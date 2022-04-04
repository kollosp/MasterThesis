import numpy as np
from sklearn.base import clone
from math import floor

class BaggingRegression:
    def __init__(self, base_estimator=None, n_estimators=5, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_features = 0
        np.random.seed(self.random_state)
        self.subspaces = []
        self.estimators = []
        return


    def fit(self, X,y):
        self.estimators = []
        self.n_features = np.unique(y).shape[0]
        if self.n_estimators == 1:
            self.subspaces = np.array([[i for i in range(0, X.shape[0])]])
        else:
            self.subspaces = np.random.randint(0, X.shape[0], (self.n_estimators, floor(self.max_samples*X.shape[0])))

        for i in range(self.n_estimators):
            es = clone(self.base_estimator)
            es.fit(X[self.subspaces[i]], y[self.subspaces[i]])
            self.estimators.append(es)

        return self

    def predict(self, X):
        pred = []
        #for each estimator
        for i, es in enumerate(self.estimators):
            #predict using all estimators
            pred.append(es.predict(X))

        pred = np.array(pred).astype(float)
        p = np.zeros([X.shape[0]])
        #vote for result
        for i in range(X.shape[0]):
            p[i] = pred[:, i].mean()

        return p

    def __str__(self):
        return "Bagging " + str(self.n_estimators) + "x(" + str(self.base_estimator) + ")"