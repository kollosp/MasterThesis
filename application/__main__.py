from FileLoader import *
from ChartManager import *
from Processing import *
from Postprocessing import *
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
from scipy.stats import kde

#[west,east,south, north]
poland =        [14,   24.132,48.9,55,"pl_map_2.png"]
lower_silesia = [15.50,18,50.26,51.57,"pl_map_2.png"]
wroclaw =       [16.60,17.5,50.9,51.3,"pl_map_2.png"]
cover_monitors =[14,19.5,49,54,"pl_map_2.png"]

def main():

    #load data
    headers, location, data = FileLoader().load_from_csv("./data/07-10Dane.csv")

    west, east, south, north, path = tuple(cover_monitors)
    cm = ChartManager(west=west, east=east, south=south, north=north,path=path)
    #cm.plot(x,y, east=19.11, west=14.82, south=50, north=53.40)

    #dlaczego to dziala?
    model = Pipeline([("PolynomialFeatures()", PolynomialFeatures()), ("Ridge", Ridge(alpha=1e-3))])

    #a to juz nie?
    #model = Pipeline([("PolynomialFeatures()",PolynomialFeatures())])

    print(model.predict)
    results = Processing().process(np.array([location]), np.array([data[1:,1800]]), [
        model,
        LinearRegression()
    ],5)

    print("==== Results analysis ====")
    print(Postprocessing().process(results))

    cm.plot(location, solar_intensity=data[1:,1800])

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()




