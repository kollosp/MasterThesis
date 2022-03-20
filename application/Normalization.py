
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from estimators.LinearInterpolation import LinearInterpolation
from estimators.NNInterpolation import NNInterpolation

class Normalization:


    @staticmethod
    def filtering(locations, data, west=0, east=0, south=0, north=0):
        indexes = [i for i in range(len(locations)) if
                   locations[i, 1] >= west and (locations[i, 1] <= east) and (
                               locations[i, 0] >= south) and locations[i, 0] <= north]

        locations = locations[indexes]
        print(data.shape)
        #for i,_ in enumerate(data):
        data = data[indexes]

        return locations, data

    @staticmethod
    def normalize(locations, data, grid_size, chart_manager=None, interpolator_class = NNInterpolation, max_row=4):

        fig, ax = None, None
        if chart_manager is not None:
            fig, ax = plt.subplots(2,3)

        #location normalisation to range 0..1
        locations[:, 0] = np.interp(locations[:, 0], (locations[:, 0].min(), locations[:, 0].max()), (0, 1))
        locations[:, 1] = np.interp(locations[:, 1], (locations[:, 1].min(), locations[:, 1].max()), (0, 1))

        #density calculation
        xf = (1/(grid_size[0]-1))
        yf = (1/(grid_size[1]-1))
        mesh_x, mesh_y= np.mgrid[0:1+xf:xf, 0:1+yf:yf]
        grid_points = np.vstack([mesh_x.ravel(), mesh_y.ravel()])
        grid_points_t = grid_points.T
        kernel = stats.gaussian_kde(locations.T)
        gaussian_density = np.reshape(kernel(grid_points).T, mesh_x.shape)

        #plot density and scatters
        if chart_manager is not None:

            chart_manager.density_plot_2d(ax[0,0], locations, grid_size)
            chart_manager.scatter_plot(ax[0,0], locations, data)


        #interpolation
        #add weights column to the locations array
        locations = np.hstack([locations, np.array([kernel(locations.T)]).T])

        #enumerate through all data pieces
        #create a new instance of interpolation object to remove previous fitting data
        interpolator = interpolator_class()
        #interpolate
        interpolator.fit(locations[:, 0:2], data)
        normalized_data = interpolator.predict(grid_points_t)


        if chart_manager is not None:
            chart_manager.mesh_plot(ax[0,1], interpolator, grid_size)
            chart_manager.scatter_plot(ax[0,1], locations, data)
            chart_manager.mesh_points_plot(ax[0,2], interpolator, grid_size)
            #ax.pcolormesh(mesh_y, mesh_x, normalized_data.reshape(grid_size),  vmin=0, vmax=1, shading='auto', alpha=0.7)
            #ax.plot(locations[:, 0], locations[:, 1], 'k.', markersize=2)


        if chart_manager is not None:
            plt.show()

        return grid_points_t, normalized_data