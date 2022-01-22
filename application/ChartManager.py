import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import math

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

    nbins = 500



    def __init__(self,north=None, south=None, west = None, east=None, path="./pl_map_2.png"):
        self.img = plt.imread(path)



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

    def plot(self, points, solar_intensity):
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

        kde = KernelDensity(bandwidth=0.4,metric="euclidean",  kernel="cosine", algorithm="ball_tree").fit(points)
        #k = kde.gaussian_kde([points[:,0], points[:,1]])
        #addaptive version
        #xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        #pl version
        #xi, yi = np.mgrid[south:north:nbins * 1j, east:west:nbins * 1j]
        xi, yi = np.mgrid[self.sub_south:self.sub_north:self.nbins * 1j, self.sub_east:self.sub_west:self.nbins * 1j]

        #print(np.hstack([xi.flatten(),yi.flatten()]))
        print()

        zi = kde.score_samples(np.hstack([np.array([xi.flatten()]).T, np.array([yi.flatten()]).T]))

        fig, ax = plt.subplots(2,3)

        ax[0,0].title.set_text('Density')
        ax[0,0].imshow(self.sub_img, extent=self.extent)
        # Make the plot
        #ax[0,0].pcolormesh(yi, xi, zi.reshape(xi.shape), shading='auto', alpha=0.6)
        ax[0,0].contourf(yi, xi, zi.reshape(xi.shape), cmap=plt.cm.coolwarm, extend='both')

        ax[0,1].title.set_text('Stations')
        ax[0,1].imshow(self.sub_img, extent=self.extent)

        #extract points which are located inside the rage of coordinates
        indexes = []
        for i in range(len(points)):
            if (points[i,1] >= self.sub_west) and (points[i,1] <= self.sub_east) and (points[i,0] >= self.sub_south) and (points[i,0] <= self.sub_north):
                indexes.append(i)
        points = points[indexes]

        #plot rest charts
        ax[0,1].scatter(points[:,1],points[:,0])

        ax[0,2].title.set_text('Solar intensity')
        ax[0,2].imshow(self.sub_img, extent=self.extent)

        solar_intensity = np.array([solar_intensity[indexes]]).T
        solar_intensity = np.hstack([solar_intensity,solar_intensity,solar_intensity])

        ax[0,2].scatter(points[:, 1], points[:, 0],c=solar_intensity)

        ax[1,0].imshow(self.sub_img, extent=self.extent)

        ax[1,1].imshow(self.sub_img, extent=self.extent)
        ax[1,2].imshow(self.sub_img, extent=self.extent)

        plt.subplots_adjust(wspace=0.05,hspace=0.0)
        plt.show()

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

        print(self.sub_img)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=np.flip(self.sub_img,0) ,shade=False)

        # show plot
        plt.show()