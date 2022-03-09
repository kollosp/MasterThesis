from FileLoader import *
from Processing import *
from estimators.PyTorchMLP import PyTorchMLP
from estimators.PyTorchSkLearnFit import PytorchRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
import random

# load data
headers, location, data = FileLoader().load_from_csv("./data/07-10Dane.csv")



def MLPCrossValidation(X,y,methods, n_splits=5, n_repeats=1, spaces=[]):
    results = np.zeros([len(spaces), len(methods), n_splits*n_repeats, 2])
    loss = []
    val_loss = []

    print(X.shape)

    #repete for all spaces
    for i, space in enumerate(spaces):
        indexes = random.sample(range(len(X)), int(len(X)*space))
        x_space = X[indexes]
        y_space = y[indexes]

        print(x_space.shape)


        # repeted StratifiedKFold 5x2cv - zamiast tstudenta mozna zastosowac ftest
        skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
        fold_index = 0

        for train_index, test_index in skf.split(x_space, y_space):
            X_train, X_test = x_space[train_index], x_space[test_index]
            y_train, y_test = y_space[train_index], y_space[test_index]

            # 5x1 cross validation
            # print(X_train.shape, X_test.shape)

            for clf_index in range(len(methods)):
                # print(clf_index,methods[clf_index])
                methods[clf_index].fit(X_train, y_train)
                # evaluate interpolation
                y_pred = methods[clf_index].predict(X_test)
                y_pred_train = methods[clf_index].predict(X_train)
                results[i, clf_index, fold_index, 0] = r2_score(y_test, y_pred)
                results[i, clf_index, fold_index, 1] = r2_score(y_train, y_pred_train)

            fold_index += 1

    return results


def learning_curve(cv_results):
    results = np.zeros([len(methods), len(spaces), 2])

    for i, _ in enumerate(cv_results):
        for j, _ in enumerate(cv_results[i]):
            #mean test
            results[j,i, 0] = cv_results[i,j, :, 0].mean()
            #mean train
            results[j,i, 1] = cv_results[i,j, :, 1].mean()

    return results

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    n_splits = 5
    n_repeats = 1
    spaces = np.linspace(0.1, 1, 25)

    print(spaces)
    methods = np.array([
        PytorchRegressor(output_dim=1, input_dim=2, num_epochs=1000, hidden_layer_dims=[10,10], verbose=0),
        PytorchRegressor(output_dim=1, input_dim=2, num_epochs=200, hidden_layer_dims=[10,10], verbose=0),
        PytorchRegressor(output_dim=1, input_dim=2, num_epochs=50, hidden_layer_dims=[10, 10], verbose=0)
    ])

    processed_data_piece = data[1:, 2080]
    #results = Processing().process(np.array([location]), np.array([processed_data_piece]), [mlp], n_splits=5)

    processed_data_piece = np.array([processed_data_piece]).T

    #print(cross_val_score(mlp, location, processed_data_piece, cv=5,
    #        scoring=make_scorer(r2_score)))

    result = MLPCrossValidation(location, processed_data_piece,methods, n_splits=n_splits, n_repeats=n_repeats, spaces=spaces)
    result = learning_curve(result)



    fig, ax = plt.subplots(len(methods))

    for i in range(len(methods)):
        ax[i].plot(spaces, result[i,:,0])
        ax[i].plot(spaces, result[i,:,1])


    plt.show()