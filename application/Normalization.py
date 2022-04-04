
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from estimators.LinearInterpolation import LinearInterpolation
from estimators.NNInterpolation import NNInterpolation

class Normalization:

    def __init__(self, grid_size, interpolator, chart_manager=None, n_estimators=10):
        self.grid_size = grid_size
        self.interpolator= interpolator
        self.chart_manager= chart_manager
        self.n_estimators = n_estimators
        pass

    def set_chart_manager(self, chart_manager):
        self.chart_manager = chart_manager

    def get_grid_dimensions(self):
        return self.grid_size

    def get_grid_size(self):
        return self.grid_size[0]*self.grid_size[1]

    # Function make normalization over data set. It also provides bagging if n_estimators is set to the value grater than 1
    def apply(self, locs, data=None):

        locations = locs.copy()

        grid_size = self.grid_size
        interpolator= self.interpolator
        chart_manager= self.chart_manager
        n_estimators = self.n_estimators


        fig, ax = None, None
        if chart_manager is not None:
            fig, ax, _ = chart_manager.create_figure([2,2], str(self))

        #location normalisation to range 0..1
        locations[:, 0] = np.interp(locations[:, 0], (locations[:, 0].min(), locations[:, 0].max()), (0, 1))
        locations[:, 1] = np.interp(locations[:, 1], (locations[:, 1].min(), locations[:, 1].max()), (0, 1))

        if data is None:
            return locations

        #plot density and scatters
        if chart_manager is not None:
            chart_manager.plot_description(ax[0,0], title="Station density", en_coordinates=True)
            chart_manager.density_plot_2d(ax[0,0], locations, (100,100))
            chart_manager.scatter_plot(ax[0,0], locations)

        #interpolation width bagging

        #enumerate through all data pieces
        #interpolate
        #normalized_data = np.zeros(n_estimators, (grid_size[0]+1)*(grid_size[1]+1))
        subspaces = np.random.randint(0, data.shape[0], (n_estimators, int(0.5*data.shape[0])))
        #for i in range(n_estimators):
        interpolator.fit(locations, data)

        #grid creation
        xf = (1/(grid_size[0]-1))
        yf = (1/(grid_size[1]-1))
        mesh_x, mesh_y= np.mgrid[0:1+xf:xf, 0:1+yf:yf]
        grid_points = np.vstack([mesh_x.ravel(), mesh_y.ravel()])
        grid_points_t = grid_points.T

        normalized_data = interpolator.predict(grid_points_t)

        if chart_manager is not None:
            chart_manager.plot_description(ax[0,1], title="Station data", en_coordinates=True)
            chart_manager.scatter_plot(ax[0,1], locations, data)
            chart_manager.plot_description(ax[1,0], title="Interpolation sampling " + str(grid_size[0]) + " x " + str(grid_size[1]), en_coordinates=True)
            chart_manager.mesh_points_plot(ax[1,0], interpolator, grid_size)

            chart_manager.plot_description(ax[1,1], title="Station data " + str(interpolator) + " int.", en_coordinates=True)
            chart_manager.mesh_plot(ax[1,1], interpolator, (100,100))
            chart_manager.scatter_plot(ax[1,1], locations, data)


        return grid_points_t, normalized_data


    def __str__(self):
        return "Normalization"
