import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.metrics import r2_score, make_scorer
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures



from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

class Processing:
    @staticmethod
    def process(X,y, methods, n_splits=5,n_repeats=4, chart_verbose=True):
        results = np.zeros([len(methods), n_repeats * n_splits])

        # repeted StratifiedKFold 5x2cv - zamiast tstudenta mozna zastosowac ftest
        skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
        fold_index = 0

        fig, ax = None, None
        if chart_verbose is True:
            fig, ax = plt.subplots(len(methods), n_repeats * n_splits + 2)


        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 5x1 cross validation
            for clf_index in range(len(methods)):
                # print(clf_index,methods[clf_index])
                methods[clf_index].fit(X_train, y_train)
                # evaluate interpolation
                y_pred = methods[clf_index].predict(X_test)
                results[clf_index, fold_index] = r2_score(y_test, y_pred)
                print("---")
                print(y_test)
                print(y_pred)
                print(results[clf_index, fold_index])

                if chart_verbose is True:
                    ax[clf_index, fold_index].scatter(X_train[:, 1], X_train[:, 0], c=y_train)

            fold_index += 1

        if chart_verbose is True:
            plt.show()

        return results


    @staticmethod
    def processMLP(X,y,estimator,train_sizes, n_splits=5,n_repeats=10):

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234),
            n_jobs=4,
            train_sizes=train_sizes,
            return_times=True,
            scoring=make_scorer(r2_score),
            #exploit_incremental_learning = True
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean,fit_times_std

        pass