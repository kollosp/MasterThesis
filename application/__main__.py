from FileLoader import *
from ChartManager import *
from Processing import *
from Postprocessing import *
from estimators.RandomRegression import RandomRegression
from estimators.GravityRegression import GravityRegression
from estimators.MeanRegression import MeanRegression
from estimators.OnesRegression import OnesRegression
from estimators.ZeroRegression import ZeroRegression
from estimators.LinearInterpolation import LinearInterpolation
from estimators.NNInterpolation import NNInterpolation
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KernelDensity
from estimators.BaggingRegression import BaggingRegression
from sklearn.ensemble import BaggingRegressor
from Normalization import Normalization
from estimators.KMeansInterpolation import KMeansInterpolation
from Filtering import Filtering
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, make_scorer
from estimators.PyTorchSkLearnFit import PytorchRegressor

import matplotlib.pyplot as plt
from scipy.stats import kde

#define possible ranges to be selected
#[west,east,south, north]
poland =        (14,   24.132,48.9,55,"pl_map_2.png",14,   24.132,48.9,55)
lower_silesia = (15.50,18,50.26,51.57,"pl_map_2.png",14,   24.132,48.9,55)
wroclaw =       (16.831,17.4,51,51.2,"wroclaw_map.png",16.831,17.4,51,51.2)
cover_monitors =(14,19.5,49,54,"pl_map_2.png",14,   24.132,48.9,55)

#select the range of area under consideration
selected_range = wroclaw

# load data
#headers, location, data = FileLoader().load_from_csv("./data/07-10Dane.csv")
headers, location, data = FileLoader().load_from_csv("./data/03-03Dane.csv")

west, east, south, north, path, img_west, img_east, img_south, img_north = selected_range
#cm = ChartManager(west=west, east=east, south=south, north=north, path=path, img_west=img_west, img_east=img_east,
#                  img_south=img_south, img_north=img_north)
cm = ChartManager(west=0, east=1, south=0, north=1, path=path, img_west=0, img_east=1,
                  img_south=0, img_north=1)

def stream(estimators, locations_count):
    locations = location[locations_count:]
    samples = data[locations_count+1:, 2000:2300:2]
    reference_locations = location[7:locations_count+7]
    reference_data = data[1:locations_count+1, 2000:2300:2]

    predictions = np.zeros([len(reference_locations), len(estimators), samples.shape[1]])

    #for each dataset
    for i in range(samples.shape[1]):
        #evaluate
        for j in range(len(estimators)):
            # learn estimator
            estimators[j].fit(locations, samples[:, i])
            #predict
            prediction = estimators[j].predict(reference_locations)

            predictions[:, j, i] = prediction

    cm.plot_in_time(x=np.array(range(samples.shape[1])),locations=reference_locations,predictions=predictions,
                    reference_data=reference_data,estimators=estimators)

    cm.show_plots()

    #plot sub step

def run(estimators):
    chart_verbose = False
    ref_location, ref_data = location, data[1:, 2080]
    norm_location, norm_data = np.array(location), np.array(data[1:, 2080])
    # Filtering data
    filtering = None
    filtering = Filtering(west=west, east=east, south=south, north=north, chart_manager=cm)

    # Normalization. In future enable bagging to remove incorrect data affect
    normalization = None
    #normalization = Normalization((15,15),interpolator=BaggingRegression(base_estimator=KMeansInterpolation(), n_estimators=10),chart_manager=None)

    # Processing - std cross validation
    results = Processing.process(location, data[1:, 2080], estimators,
                                 filtering=filtering, normalization=normalization, n_splits=5, chart_manager=cm)

    # Evaluation - compare fitted estimators with row location and data (before normalization)

    # Statistical results analysis (t-student's test)

    #[0] -> because only one dataset
    res_stats = Postprocessing().stats(np.array([results]))[0]

    print("==== Results ====")
    np.set_printoptions(suppress=True, precision=3)
    print(results)
    print("==== Results stats ====")
    print(res_stats)
    print("==== Results analysis ====")
    print(Postprocessing().process(results))

    if filtering is not None:
        filtering.set_chart_manager(cm)
        norm_location, norm_data = filtering.apply(norm_location, norm_data)
        ref_location, ref_data = filtering.apply(ref_location, ref_data)
    if normalization is not None:
        normalization.set_chart_manager(cm)
        norm_location, norm_data = normalization.apply(norm_location, norm_data)
        ref_location = normalization.apply(ref_location)
    for estimator in estimators:
        estimator.fit(norm_location,norm_data)
    #cm.plot(ref_location, solar_intensity=ref_data, estimators=estimators, res_stats=res_stats, n_splits=5, adaptive_colors=True)

    cm.show_figures()
    cm.show_plots()

def learning_curves(estimators, train_sizes= np.linspace(0.4, 1, 10)):

    #index = 2080 + 2*20
    index = 60

    filtered_data = data[1:,index]
    last_data = data[1:,index-2:index]
    filtering = Filtering(west=west, east=east, south=south, north=north, chart_manager=cm)
    filtered_location, filtered_data = filtering.apply(location, filtered_data)

    normalization = None
    normalization = Normalization((20,20),interpolator=BaggingRegression(base_estimator=LinearInterpolation(), n_estimators=25),chart_manager=None)
    #norm_location, norm_data = normalization.apply(location, norm_data)
    n_splits = 5

    fig, ax, _ = cm.create_figure((1, len(estimators)), "Learning curves")

    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_scores, fit_scores_std = Processing().learning_curve(
        filtered_location, filtered_data, estimators,
        y_his=last_data,
        verbose=False,
        normalization=normalization,
        chart_manager=cm,
        x_extension=XExtension(),
        train_sizes=train_sizes, n_splits=n_splits)

    for j, estimator in enumerate(estimators):
        print("plot learning curves for ", str(estimator))
        #train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean, fit_times_std = \
        #    Processing().learning_curve(norm_location, norm_data, estimators[j], train_sizes=train_sizes, n_splits=n_splits)
        print(train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
        if len(estimators) == 1:
            cm.plot_learning_curve(ax, estimator, train_scores_mean[j], test_scores_mean[j], train_sizes, fit_scores[j], fit_scores_std[j],
                                   train_scores_std=train_scores_std[j], test_scores_std=test_scores_std[j])
        else:
            cm.plot_learning_curve(ax[j], estimator, train_scores_mean, test_scores_mean, train_sizes, fit_scores[j], fit_scores_std[j],
                                   train_scores_std=train_scores_std[j], test_scores_std=test_scores_std[j])

    cm.show_figures()


    #cm.plot(location, solar_intensity=processed_data_piece, estimators=estimators, res_stats=res_stats, n_splits=n_splits, adaptive_colors=True)
    #cm.plot_input_data_distribution(np.array([
    #    (data[0, i], data[1:, i]) for i in range(2000,2200) if i%10==0
    #],dtype=object))

#start = 2040, start = 180
def test_then_train(estimators, estimator_count=0, chunks=300, step=1, chunk_size=3, start = 1050, en_weights=False):

    ev_plus = np.zeros((len(estimators), chunks))
    #memory for each prediction
    predictions = np.zeros((len(location), len(estimators), chunks))
    stop = start + chunks
    fitted = False
    for i in range(start,stop,step):
        for j, estimator in enumerate(estimators):

            if fitted:
                # predict future in known points
                pred_x = location
                if chunk_size > 1:
                    pred_x = np.append(location, np.full((len(location), 1), 1), axis=1)
                pred = estimators[j].predict(pred_x)
                evaluation = r2_score(data[1:, i], pred)
                ev_plus[j, int((i - start) / step)] = evaluation
                predictions[:, j, int((i - start) / step)] = pred

            density = None
            if en_weights:
                kde = KernelDensity(bandwidth=0.08 * np.min(
                    [location[:, 0].max() - location[:, 0].min(), location[:, 1].max() - location[:, 1].min()]),
                                    metric="euclidean", kernel="gaussian", algorithm="ball_tree").fit(location)
                density = 1/np.exp(kde.score_samples(location))


            if chunk_size > 1:
                #fit each estimator
                fit_data_x = np.zeros([chunk_size*len(location),3])
                fit_data_y = np.zeros([chunk_size*len(location)])
                for k in range(chunk_size):
                    fit_data_x[k*len(location):(k+1)*len(location), 0:2] = location
                    fit_data_x[k*len(location):(k+1)*len(location), 2] = k-chunk_size+1
                    fit_data_y[k*len(location):(k+1)*len(location)] = data[1:, i + (k-chunk_size+1)*step]

                estimators[j].fit(fit_data_x,fit_data_y, density)
            else:
                estimators[j].fit(location, data[1:, i], density)

        fitted = True

    for i, _ in enumerate(estimators):
        print( "{0}: +1: R^2={1:.3f}({2:.3f})".format(estimators[i], ev_plus[i].mean(), ev_plus[i].std()))

    cm.plot_in_time(x=np.array(range(start,stop,step)),locations=location[4:6],predictions=predictions[4:6,:,:],
                    reference_data=data[4:6, start:stop:step],estimators=estimators)
    cm.pause(0.05)
    cm.show_plots()


def main():



    estimators2 = [
        LinearRegression(),
        MeanRegression(),
        DecisionTreeRegressor(random_state=0),
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        #PytorchRegressor(output_dim=1, input_dim=3, num_epochs=200, hidden_layer_dims=[10, 10], verbose=0),
    ]

    #Obserwacje:
    # modele drzew o glebokosci wiekszej niz 3 sa przetrenowane (pradopodobnie przez rozklad danych wejsciowch)
    # Drzewa niestety nie rozwiazuja problu lepiej niz regresor sredni. Lasy zgodnie z przypuszczeniami generuja mniejsza
    # warinacje od pojedynczych drzew decyzyjnych
    trees = [
        MeanRegression(),
        DecisionTreeRegressor(random_state=1041, max_depth=3),
        DecisionTreeRegressor(random_state=1041, max_depth=2),
        RandomForestRegressor(random_state=1041, max_depth=3,n_estimators=200),
        RandomForestRegressor(random_state=1041, max_depth=3,n_estimators=100),
        RandomForestRegressor(random_state=1041, max_depth=3,n_estimators=1000),
        RandomForestRegressor(random_state=1041, max_depth=2,n_estimators=200),
        RandomForestRegressor(random_state=1041, max_depth=2,n_estimators=100),
        RandomForestRegressor(random_state=1041, max_depth=2,n_estimators=1000),
    ]

    estimators1 = [
        LinearRegression(),
        RandomRegression(),
        NNInterpolation(),
        LinearInterpolation(),
        RandomForestRegressor(max_depth=3),
        MeanRegression(),
        # A Classification and Regression Tree(CART)
        DecisionTreeRegressor(random_state=0, max_depth=3),
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        GravityRegression(func="homographic", gaussian_bandwidth=2),
        #MLPRegressor(hidden_layer_sizes=(10, 10, 10),
        #             max_iter=300000, activation='tanh',
        #             solver='lbfgs', random_state=5),
    ]


    #obserwacje bagging dziala jak powinien zmniejszajac wariancje ale oslabiajac dokladnosc. Gaussian_bandwidth najlepiej ustawic
    #na 2 lub 4. Bez znaczenia jakie jadro zostanie wykorzystane do predykcji.
    gravity = [
        MeanRegression(),
        GravityRegression(func="homographic", gaussian_bandwidth=4),
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        BaggingRegressor(base_estimator=GravityRegression(func="homographic",gaussian_bandwidth=4), n_estimators=20),
        BaggingRegressor(base_estimator=GravityRegression(func="exponential",gaussian_bandwidth=4), n_estimators=20),

        MLPRegressor(hidden_layer_sizes=(10, 10),
                     max_iter=3000, activation='logistic',
                     solver='lbfgs', random_state=5),
    ]

    g = [
        GravityRegression(func="homographic", gaussian_bandwidth=1),
        GravityRegression(func="exponential", gaussian_bandwidth=2),
        DecisionTreeRegressor(random_state=0),
    ]

    max_iter = 4000
    mlp = [
        # MLP
        # for 1 history element
        #MLPRegressor(hidden_layer_sizes=(15,10,2),
        #             max_iter=max_iter,solver='lbfgs', random_state=5,
        #             verbose=False),

        # for 2 history element
        #MLPRegressor(hidden_layer_sizes=(20, 15, 15, 15),
        #             max_iter=max_iter, solver='lbfgs',
        #             activation="relu",
        #             random_state=5,
        #             verbose=False),
        # for 2 history element
        MLPRegressor(hidden_layer_sizes=(50,100,25),
                     max_iter=max_iter,
                     #solver='lbfgs',
                     #activation="relu",
                     random_state=5,
                     verbose=False),
        #GravityRegression(func="homographic", gaussian_bandwidth=2),

        #MLPRegressor(hidden_layer_sizes=(20,20,20),
        #             max_iter=3000, activation='tanh',
        #             solver='lbfgs', random_state=5),
        #MLPRegressor(hidden_layer_sizes=(10,10,10),
        #             max_iter=30000, activation='tanh',
        #             solver='lbfgs', random_state=5),
        #MLPRegressor(hidden_layer_sizes=(60,60),
        #             max_iter=30000, activation='tanh',
        #             solver='lbfgs', random_state=5),
        #RandomForestRegressor(),
        # MLP
        #MLPRegressor(hidden_layer_sizes=(100, 100, 100),
        #             max_iter=1000, activation='tanh',
        #             solver='lbfgs', random_state=5)
        #GravityRegression(func="exponential", gaussian_bandwidth=2),
    ]

    most_interesting = [
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        GravityRegression(func="homographic", gaussian_bandwidth=2),
        NNInterpolation(),
        LinearInterpolation(),
        MLPRegressor(hidden_layer_sizes=(100, 100, 100),
                     max_iter=1000, activation='tanh',
                     solver='lbfgs', random_state=5)
    ]

    learning_curves(mlp)
    #run(estimators1)
    #run(gravity, iterations=1)
    #run(mlp, iterations=1)
    #stream(g, locations_count=4)
    #stream(g, locations_count=4)
    #runMPL(mlp, train_sizes = np.linspace(0.1, 1.0, 25))

    #test_then_train(estimators2)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()




