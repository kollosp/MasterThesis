import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.metrics import r2_score
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
class Processing:

    @staticmethod
    def processOne(X,y,methods,n_splits):
        # Procedura uczenia na zbiorze
        results = np.zeros([len(methods), n_splits])

        # repeted StratifiedKFold 5x2cv - zamiast tstudenta mozna zastosowac ftest
        skf = KFold(n_splits, shuffle=False)
        fold_index = 0

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #5x1 cross validation
            #print(X_train.shape, X_test.shape)

            for clf_index in range(len(methods)):
                #print(clf_index,methods[clf_index])
                methods[clf_index].fit(X_train, y_train)
                #evaluate interpolation
                y_pred = methods[clf_index].predict(X_test)
                results[clf_index, fold_index] = r2_score(y_test, y_pred)

            fold_index += 1

        return results


    @staticmethod
    def process(X,y, methods, n_splits=5):
        results = np.zeros([len(y), len(methods), n_splits])

        for index in range(len(y)):
            results[index] = Processing().processOne(X[index], y[index], methods, n_splits=n_splits)

        return results



