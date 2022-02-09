import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import math
import datetime

class ChartManager:
    north = 55
    south = 48.9
    west = 14
    east = 24.132


    sub_north = 55
    sub_south = 48.9
    sub_west = 14
    sub_east = 24.132

    img = None

    x_domain = 0
    y_domain = 0
    extent = []

    nbins = 100



    def __init__(self,north=None, south=None, west = None, east=None, path="./pl_map_2.png", img_north=55, img_south=48.9, img_west=14, img_east=24.132):
        self.img = plt.imread(path)
        self.north = img_north
        self.south = img_south
        self.west = img_west
        self.east = img_east

        if north is None:
            self.sub_north = self.north
        else:
            self.sub_north = north
        if south is None:
            self.sub_south = self.south
        else:
            self.sub_south = south
        if west is None:
            self.sub_west = self.west
        else:
            self.sub_west = west
        if east is None:
            self.sub_east = self.east
        else:
            self.sub_east = east

        self.y_domain = self.north - self.south
        self.x_domain = self.east - self.west


        self.extent = [self.sub_west, self.sub_east, self.sub_south, self.sub_north]

        y_min, y_max = ((self.sub_south - self.south) / self.y_domain), (
                    1 - ((self.north - self.sub_north) / self.y_domain)),
        x_min, x_max = ((self.sub_west - self.west) / self.x_domain), (
                    1 - ((self.east - self.sub_east) / self.x_domain))
        shape = self.img.shape

        self.sub_img = self.img[
              shape[0] - math.floor(shape[0] * y_max): shape[0] - math.floor(shape[0] * y_min),
              math.floor(shape[1] * x_min): math.floor(shape[1] * x_max),
        ]

        print(y_min, y_max,x_min, x_max, self.extent)

    def plot_input_data_distribution(self, data):
        col_count = 5
        fig, ax = plt.subplots(math.ceil(len(data)/col_count),col_count)
        for i in range(len(data)):
            x_plot = np.linspace(0, 1, 100)[:, np.newaxis]
            kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(np.array([data[i][1]]).T)
            log_dens = kde.score_samples(x_plot)
            ax[math.floor(i/col_count),i%col_count].title.set_text(datetime.datetime.fromtimestamp(math.floor(data[i][0]/1000)))
            ax[math.floor(i/col_count),i%col_count].plot(x_plot.flatten(), np.exp(log_dens))


    def plot(self, points, solar_intensity, estimators, n_splits, adaptive_colors=True, res_stats=None):
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

        kde = KernelDensity(bandwidth=0.08*np.min([self.sub_north - self.sub_south, self.sub_east-self.sub_west]),metric="euclidean",  kernel="cosine", algorithm="ball_tree").fit(points)
        #k = kde.gaussian_kde([points[:,0], points[:,1]])
        #addaptive version
        #xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        #pl version
        #xi, yi = np.mgrid[south:north:nbins * 1j, east:west:nbins * 1j]
        xi, yi = np.mgrid[self.sub_south:self.sub_north:self.nbins * 1j, self.sub_east:self.sub_west:self.nbins * 1j]
        yi_max = yi.max()
        xi_max = xi.max()
        yi_min = yi.min()
        xi_min = xi.min()
        xi_delta = xi_max - xi_min
        yi_delta = yi_max - yi_min
        zi = kde.score_samples(np.hstack([np.array([xi.flatten()]).T, np.array([yi.flatten()]).T]))
        #print(np.hstack([xi.flatten(),yi.flatten()]))

        fig, ax = plt.subplots(1+math.ceil(len(estimators)/3),3)

        ax[0,0].title.set_text('Density')

        #extract points which are located inside the rage of coordinates
        indexes = [i for i in range(len(points)) if points[i,1] >= self.sub_west and (points[i,1] <= self.sub_east) and (points[i,0] >= self.sub_south) and points[i,0] <= self.sub_north]
        #for i in range(len(points)):
        #    if (points[i,1] >= self.sub_west) and (points[i,1] <= self.sub_east) and (points[i,0] >= self.sub_south) and (points[i,0] <= self.sub_north):
        #        indexes.append(i)
        points = points[indexes]

        #plot rest charts
        ax[0,0].scatter(points[:,1],points[:,0])

        ax[0,0].imshow(self.sub_img, extent=self.extent)
        # Make the plot
        #ax[0,0].pcolormesh(yi, xi, zi.reshape(xi.shape), shading='auto', alpha=0.6)
        ax[0,0].contourf(yi, xi, zi.reshape(xi.shape), cmap=plt.cm.coolwarm, extend='both', alpha=0.6)

        x_plot = np.linspace(0, 1, 100)[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(np.array([solar_intensity]).T)
        log_dens = kde.score_samples(x_plot)

        ax[0, 1].title.set_text('Input data dis.')
        ax[0, 1].plot(x_plot.flatten(), np.exp(log_dens))

        ax[0,2].title.set_text('Solar intensity')
        ax[0,2].imshow(self.sub_img, extent=self.extent)

        ax[0,2].text(yi_max - 0.20 * yi_delta, xi_max - 0.15 * xi_delta,
                                               ("%.3f" % np.max(solar_intensity)).lstrip("0"))
        ax[0,2].text(yi_max - 0.20 * yi_delta, xi_max - 0.30 * xi_delta,
                                               ("%.3f" % np.min(solar_intensity)).lstrip("0"))
        ax[0,2].text(yi_max - 0.20 * yi_delta, xi_max - 0.45 * xi_delta,
                                               ("%.3f" % np.mean(solar_intensity)).lstrip("0"))
        ax[0,2].text(yi_max - 0.20 * yi_delta, xi_max - 0.60 * xi_delta,
                                               ("%.3f" % np.std(solar_intensity)).lstrip("0"))
        ax[0,2].text(yi_max - 0.20 * yi_delta, xi_max - 0.75 * xi_delta,
                                               ("%d" % points.shape[0]))

        solar_intensity =solar_intensity[indexes]
        #solar_intensity = np.array([solar_intensity]).T
        #solar_intensity = np.hstack([solar_intensity,solar_intensity,solar_intensity])

        ax[0,2].scatter(points[:, 1], points[:, 0],c=solar_intensity)

        #define density of models check points
        nbins = 15
        #xi, yi = np.mgrid[self.sub_south:self.sub_north:nbins * 1j, self.sub_east:self.sub_west:nbins * 1j]
        xi, yi = np.meshgrid(np.linspace(self.sub_south, self.sub_north), np.linspace(self.sub_east, self.sub_west))
        yi_max = yi.max()
        xi_max = xi.max()
        yi_min = yi.min()
        xi_min = xi.min()
        xi_delta = xi_max - xi_min
        yi_delta = yi_max - yi_min
        test_set_size = points.shape[0]/n_splits
        for estimator_index in range(len(estimators)):
            plot_row = math.floor(estimator_index/3+1)
            ax[plot_row,estimator_index%3].title.set_text(estimators[estimator_index])
            #ax[plot_row,estimator_index%3].imshow(self.sub_img, extent=self.extent)

            zi = estimators[estimator_index].predict(
                np.hstack([np.array([xi.flatten()]).T, np.array([yi.flatten()]).T]))
            zi = zi.reshape(xi.shape)

            #truncate if value grater then 1
            #zi = [x if x<1 else 1 for x in zi]
            #switch to grayscale and disable adaptiveness
            #zi = np.array([zi]).T
            #zi = np.hstack([zi, zi, zi])
            #print(yi.shape,xi.shape, )
            #ax[plot_row,estimator_index%3].scatter(yi,xi, c=zi,  vmin=0, vmax=1)
            if adaptive_colors:
                ax[plot_row,estimator_index%3].pcolormesh(yi,xi, zi, shading='auto', alpha=0.7)
            else:
                ax[plot_row,estimator_index%3].pcolormesh(yi,xi, zi,  vmin=0, vmax=1, shading='auto', alpha=0.7)

            ax[plot_row,estimator_index%3].scatter(points[:, 1], points[:, 0],c=solar_intensity)

            ax[plot_row, estimator_index % 3].text(yi_max - 0.20 * yi_delta,xi_max- 0.15 * xi_delta , ("%.3f" %  np.max(zi)).lstrip("0"))
            ax[plot_row, estimator_index % 3].text(yi_max - 0.20 * yi_delta,xi_max- 0.30 * xi_delta , ("%.3f" %  np.min(zi)).lstrip("0"))

            if res_stats is not None:
                ax[plot_row, estimator_index % 3].text(yi_max - 0.30 * yi_delta, xi_max - 0.45 * xi_delta,
                                                       "μ(R^2):" + ("%.3f" % res_stats[estimator_index,0]))
                ax[plot_row, estimator_index % 3].text(yi_max - 0.20 * yi_delta, xi_max - 0.60 * xi_delta,
                                                       "σ:" + ("%.3f" % res_stats[estimator_index,1]))
                ax[plot_row, estimator_index % 3].text(yi_max - 0.30 * yi_delta, xi_max - 0.75 * xi_delta,
                                                      "MEA:" + ("%.3f" % ((1-res_stats[estimator_index,0])/test_set_size)))
                ax[plot_row, estimator_index % 3].text(yi_max - 0.20 * yi_delta, xi_max - 0.90 * xi_delta,
                                                       ("%.1f" % (100*(1-res_stats[estimator_index,0])/test_set_size)) + "%")

        #fill rest of subplots to provide same size
        if len(estimators)%3 > 0:
            plot_row = math.floor(estimator_index / 3 + 1)
            for estimator_index in range(len(estimators)%3, 3):
                ax[plot_row,estimator_index].imshow(self.sub_img, extent=self.extent)

        #plt.subplots_adjust(wspace=0.05,hspace=0.0)
    def plot3d(self, x,y,z):
        # Creating figure
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(x, y, z, color="green")
        plt.title("simple 3D scatter plot")
        extent = [self.west, self.east, self.south, self.north]
        # create a 21 x 21 vertex mesh
        xi, yi = np.mgrid[self.sub_south:self.sub_north:self.nbins * 1j, self.sub_east:self.sub_west:self.nbins * 1j]

        # create vertices for a rotated mesh (3D rotation matrix)
        X = xi
        Y = yi
        Z = np.zeros(X.shape)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=np.flip(self.sub_img,0) ,shade=False)

    def plot_in_time(self, x, reference_data, predictions, locations, estimators=None):
        #predictions[0] - location
        #predictions[1] - estimator
        #predictions[2] - point in time
        fig, ax = plt.subplots(predictions.shape[0])

        print(locations)
        for i in range(locations.shape[0]):
            ax[i].title.set_text("lat: " + ("%.3f" % locations[i,0]) + " long: " + ("%.3f" % locations[i, 1]))

        for i in range(reference_data.shape[0]):
            ax[i].plot(x, reference_data[i],'go-', markersize=2, label='Reference')

        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if estimators is not None:
                    ax[i].plot(x, predictions[i, j], markersize=2, marker='o', label=estimators[j])
                else:
                    ax[i].plot(x, predictions[i, j], markersize=2, marker='o')

        ax[0].legend()


    def plot_learning_curve(self, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean,fit_times_std, train_sizes):

        fig, axes = plt.subplots(3)

        # Plot learning curve
        axes[0].grid()
        '''
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )'''
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )

        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        '''axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )'''
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        '''axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )'''
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

    def show_plots(self):
        plt.show()

    def pause(self, delay):
        plt.pause(delay)