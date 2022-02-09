from FileLoader import *
from ChartManager import *
from Processing import *
from Postprocessing import *
from Postprocessing import *
from estimators.RandomRegression import RandomRegression
from estimators.GravityRegression import GravityRegression
from estimators.OnesRegression import OnesRegression
from estimators.ZeroRegression import ZeroRegression
from estimators.LinearInterpolation import LinearInterpolation
from estimators.NNInterpolation import NNInterpolation
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
from scipy.stats import kde

#define possible ranges to be selected
#[west,east,south, north]
poland =        (14,   24.132,48.9,55,"pl_map_2.png",14,   24.132,48.9,55)
lower_silesia = (15.50,18,50.26,51.57,"pl_map_2.png",14,   24.132,48.9,55)
wroclaw =       (16.831,17.4,51,51.2,"wroclaw_map.png",16.831,17.4,51,51.2)
cover_monitors =(14,19.5,49,54,"pl_map_2.png",14,   24.132,48.9,55)

#select the range of area under consideration
selected_range = cover_monitors

# load data
headers, location, data = FileLoader().load_from_csv("./data/07-10Dane.csv")

west, east, south, north, path, img_west, img_east, img_south, img_north = selected_range
cm = ChartManager(west=west, east=east, south=south, north=north, path=path, img_west=img_west, img_east=img_east,
                  img_south=img_south, img_north=img_north)


def stream(estimators, locations_count):
    locations = location[locations_count:]
    samples = data[locations_count+1:, 2000:2300:2]
    reference_locations = location[0:locations_count]
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


def run(estimators, iterations=1):

    for i in range(iterations):
        processed_data_piece = data[1:,2080 + i*20]
        n_splits = 5
        results = Processing().process(np.array([location]), np.array([processed_data_piece]), estimators, n_splits=n_splits)

        #[0] -> because only one dataset
        res_stats = Postprocessing().stats(results)[0]

        for estimator in estimators:
            estimator.fit(location,processed_data_piece)

        print("==== Results ====")
        np.set_printoptions(suppress=True, precision=3)
        print(results)
        print("==== Results stats ====")
        print(res_stats)
        print("==== Results analysis ====")
        print(Postprocessing().process(results))


        cm.plot(location, solar_intensity=processed_data_piece, estimators=estimators, res_stats=res_stats, n_splits=n_splits, adaptive_colors=True)
        #cm.plot_input_data_distribution(np.array([
        #    (data[0, i], data[1:, i]) for i in range(2000,2200) if i%10==0
        #],dtype=object))
        cm.pause(0.05)
    cm.show_plots()

def runMPL(estimators, iterations=1):

    for i in range(iterations):
        processed_data_piece = data[1:,2080 + i*20]
        n_splits = 5

        for j in range(len(estimators)):
            train_sizes = np.linspace(0.1, 1.0, 5)
            train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean, fit_times_std = \
                Processing().processMLP(location, processed_data_piece, estimators[j], train_sizes=train_sizes, n_splits=n_splits)
            cm.plot_learning_curve(train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean,fit_times_std, train_sizes)
            cm.pause(0.05)



        #cm.plot(location, solar_intensity=processed_data_piece, estimators=estimators, res_stats=res_stats, n_splits=n_splits, adaptive_colors=True)
        #cm.plot_input_data_distribution(np.array([
        #    (data[0, i], data[1:, i]) for i in range(2000,2200) if i%10==0
        #],dtype=object))

    cm.show_plots()

def main():
    estimators1 = [
        LinearRegression(),
        RandomRegression(),
        NNInterpolation(),
        LinearInterpolation(),
        RandomForestRegressor(),
        # A Classification and Regression Tree(CART)
        DecisionTreeRegressor(random_state=0),
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        GravityRegression(func="homographic", gaussian_bandwidth=2),
        MLPRegressor(hidden_layer_sizes=(10, 10, 10),
                     max_iter=300000, activation='tanh',
                     solver='lbfgs', random_state=5),
    ]

    gravity = [
        GravityRegression(func="homographic", gaussian_bandwidth=1),
        GravityRegression(func="homographic", gaussian_bandwidth=2),
        GravityRegression(func="homographic", gaussian_bandwidth=4),
        GravityRegression(func="exponential", gaussian_bandwidth=1),
        GravityRegression(func="exponential", gaussian_bandwidth=2),
        GravityRegression(func="exponential", gaussian_bandwidth=4),
        GravityRegression(bandwidth=0.6),
        GravityRegression(bandwidth=0.8),
        GravityRegression(bandwidth=1),
    ]

    g = [
        GravityRegression(func="homographic", gaussian_bandwidth=1),
        GravityRegression(func="exponential", gaussian_bandwidth=2),
        DecisionTreeRegressor(random_state=0),
    ]

    mlp = [
        # MLP
        #MLPRegressor(hidden_layer_sizes=(20, 20, 20),
        #             max_iter=1000, activation='tanh',
        #             solver='lbfgs', random_state=5),
        # MLP
        MLPRegressor(hidden_layer_sizes=(10,10),
                     max_iter=3000, activation='logistic',
                     solver='lbfgs', random_state=5),
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

    #run(estimators1, iterations=1)
    #run(gravity, iterations=1)
    #run(mlp, iterations=1)
    #stream(estimators1, locations_count=4)
    stream(g, locations_count=4)
    #runMPL(mlp)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()




