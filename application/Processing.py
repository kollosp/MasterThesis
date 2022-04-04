import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.metrics import r2_score, make_scorer,mean_squared_error,max_error,mean_absolute_error
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import sys


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import random

from XExtension import XExtension

#metric = r2_score
#metric = max_error
metric = mean_absolute_error

class Processing:
    @staticmethod
    def process(x,y, methods, filtering = None, normalization = None, n_splits=5, n_repeats=1, chart_manager=None):

        #filtering passed arguments to limit range of computation
        if filtering:
            x, y = filtering.apply(x,y)

        results = np.zeros([len(methods), n_repeats * n_splits])

        # repeted StratifiedKFold 5x2cv - zamiast tstudenta mozna zastosowac ftest
        skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
        fold_index = 0

        fig, ax = None, None
        if chart_manager is not None:
            fig, ax, _ = chart_manager.create_figure((1, len(methods)), "Processing")

        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # normalization of x and y data
            if normalization is not None:
                #making grid instead of samples
                x_train, y_train = normalization.apply(x_train, y_train)
                #normalization only on x = rescaling x into 0..1 range
                x_test = normalization.apply(x_test)

            # 5x1 cross validation
            for i, method in enumerate(methods):
                #fit estimator
                method.fit(x_train, y_train)

                # evaluate estimator
                y_pred = method.predict(x_test)

                results[i, fold_index] = r2_score(y_test, y_pred)

            fold_index += 1


        #printing results
        if chart_manager is not None:
            #normaliza full arguments set
            x_train, y_train = normalization.apply(x, y)

            #fit each estimator
            for i, method in enumerate(methods):
                method.fit(x_train, y_train)

                #plot estimator grid
                chart_manager.mesh_plot(ax[i], method, (26,26), en_norm=False)
                chart_manager.plot_description(ax[i], en_coordinates=True)
                chart_manager.text_plot(ax[i], [
                    "μ(R^2):" + ("%.3f" % np.mean(results[i])),
                    "σ:" + ("%.3f" % np.std(results[i])),
                    "MEA" + ("%.3f" % ((1-results[i].mean())/len(results[i]))),
                ])

        return results

    @staticmethod
    def learning_curve(X,y,methods,train_sizes, y_his = None, normalization=None, n_splits=5, verbose=True, chart_manager=None, x_extension=None):
        '''
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=1234),
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
        '''
        np.set_printoptions(threshold=sys.maxsize, suppress=True)

        train_scores = np.zeros((len(methods), len(train_sizes)))
        test_scores = np.zeros((len(methods), len(train_sizes)))
        fit_scores = np.zeros((len(methods), len(train_sizes))) #test to learning set (normalized)
        train_scores_std = np.zeros((len(methods), len(train_sizes)))
        test_scores_std = np.zeros((len(methods), len(train_sizes)))
        fit_scores_std = np.zeros((len(methods), len(train_sizes))) #test to learning set (normalized)

        fig, ax, _ = 0,0,0
        if chart_manager is not None:
            fig, ax, _ = chart_manager.create_figure((n_splits, len(train_sizes)), "CV selection")

        #for each training size
        for i, train_size in enumerate(train_sizes):
            indexes = random.sample(range(X.shape[0]), int(train_size*X.shape[0]))

            filtered_x = X[indexes]
            filtered_y = y[indexes]

            #cross validation
            skf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

            if verbose is True:
                print("Cross validation:", i+1, " Using ", int(100*train_size), "% of available data pieces")
                print("Indexes to be used:", indexes)

            loop_counter = 1
            train_scores_list = np.zeros((len(methods), n_splits))
            test_scores_list = np.zeros((len(methods), n_splits))
            fit_scores_list = np.zeros((len(methods), n_splits))
            for train_index, test_index in skf.split(filtered_x, filtered_y):
                x_train, x_test = filtered_x[train_index], filtered_x[test_index]
                y_train, y_test = filtered_y[train_index], filtered_y[test_index]

                x_train_no_normalization = x_train.copy()
                y_train_no_normalization = y_train.copy()

                if verbose is True:
                    print("\tIteration", loop_counter, "Training set:", len(x_train), "| Test set: ", len(x_test))

                # normalization of x and y data
                if normalization is not None:
                    # making grid instead of samples
                    x_train_no_normalization = normalization.apply(x_train_no_normalization)

                    x_train_ext, y_train_ext = normalization.apply(x_train, y_train)
                    x_train = normalization.apply(x_train)
                    x_train = np.append(x_train, x_train_ext,axis=0)
                    y_train = np.append(y_train, y_train_ext,axis=0)
                    #x_train = normalization.apply(x_train)
                    # normalization only on x = rescaling x into 0..1 range
                    x_test = normalization.apply(x_test)

                    if verbose is True:
                        print("\tNormalization: Learning set size: ", x_train.shape[0])

                if chart_manager is not None:
                    chart_manager.scatter_plot(ax[loop_counter-1,i], x_train, 'b')
                    chart_manager.scatter_plot(ax[loop_counter-1,i], x_test, 'r')
                    chart_manager.scatter_plot(ax[loop_counter-1,i], x_train_no_normalization, 'g')

                if y_his is not None and x_extension is not None:
                    x_train_no_normalization = x_extension.apply(x_train_no_normalization, y_his, cv_indexes=train_index)
                    x_test = x_extension.apply(x_test, y_his, cv_indexes=test_index)
                    x_train = x_extension.apply(x_train, y_his, x_train=filtered_x[train_index], normalization=normalization, cv_indexes=train_index)

                    print(np.append(x_train, np.array([y_train]).T, axis=1))

                for j, method in enumerate(methods):
                    # fit estimator

                    method.fit(x_train, y_train)

                    #plot learning curves instead of cv
                    #if chart_manager is not None:
                    #    #for mlp with adam solver
                    #    print(method.loss_)
                    #    chart_manager.plot_line(ax[loop_counter - 1, i], list(range(len(method.loss_curve_))),method.loss_curve_)

                    y_pred_to_fit = method.predict(x_train)
                    fit_score = metric(y_train, y_pred_to_fit)
                    fit_scores_list[j, loop_counter - 1] = fit_score
                    # evaluate to the learning set
                    y_pred_to_train = method.predict(x_train_no_normalization)
                    train_score = metric(y_train_no_normalization, y_pred_to_train)
                    train_scores_list[j, loop_counter-1] = train_score
                    # evaluate to the test set
                    y_pred_to_test = method.predict(x_test)
                    test_score = metric(y_test, y_pred_to_test)
                    print("Prediction",loop_counter - 1, i,"\n", np.array([y_test,y_pred_to_test]))
                    mean = np.mean(y_test)
                    print(np.sum((y_test-y_pred_to_test)**2), np.sum((y_test-mean)**2),test_score)
                    test_scores_list[j, loop_counter-1] = test_score

                    if chart_manager is not None:
                        chart_manager.text_plot(ax[loop_counter - 1, i], [
                            #("%.3f" % fit_score),
                            #("%.3f" % train_score),
                            ("%.3f" % test_score),
                        ], )

                    if verbose is True:
                        print("\t\t", i, str(method), "R^2:", round(train_score,2), round(test_score,2))
                loop_counter += 1

            #calculate mean value
            for j, _ in enumerate(methods):
                train_scores[j, i] = np.mean(train_scores_list[j])
                train_scores_std[j, i] = np.std(train_scores_list[j])
                test_scores[j, i] =  np.mean(test_scores_list[j])
                test_scores_std[j, i] =  np.std(test_scores_list[j])
                fit_scores[j, i] =  np.mean(fit_scores_list[j])
                fit_scores_std[j, i] =  np.std(fit_scores_list[j])

            if verbose is True:
                print("Cross validation", i+1, " Summary")
                for j, method in enumerate(methods):
                    print("\t", str(method), "avg(R^2):", round(train_scores[j, i],2), "+/-", round(train_scores_std[j, i],3),
                          "|", round(test_scores[j, i],2), "+/-", round(test_scores_std[j, i],3), "|", round(fit_scores[j, i],2), "+/-", round(fit_scores_std[j, i],3))


        return train_scores,train_scores_std, test_scores, test_scores_std, fit_scores, fit_scores_std