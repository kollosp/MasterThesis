#!/usr/bin/python
import sys
from csv import reader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kde


if len(sys.argv) == 1:
    print("Error. No filepath. Please tell me what file do you want to load.")
    exit(1)


# Change color palette
#plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
#plt.show()


# create data
x = []
y = []

# open file in read mode
with open(sys.argv[1], 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    index = 0
    for row in csv_reader:
        if index > 2:
            # row variable is a list that represents a row in csv
            if len(row) >= 3:
                print(row[0], row[1], row[2])
                x.append(row[1])
                y.append(row[2])
        index += 1

x = np.array(x).astype(np.float)
y = np.array(y).astype(np.float)
print(x, y)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins = 300
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
plt.scatter(x,y)
plt.show()
