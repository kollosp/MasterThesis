
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


kde = KernelDensity(bandwidth=0.3,metric="euclidean", kernel="gaussian")

data = np.random.randint(10, size=100).reshape(-1, 1)
print(data)
kde.fit(data)

linear_space = np.linspace(0,10, 1000).reshape(-1, 1)
pred = kde.score_samples(linear_space)
exp_pred = np.exp(pred)



fig, ax = plt.subplots(2,2)
ax[0,0].plot(linear_space.T[0], exp_pred)
ax[1,0].plot(linear_space.T[0], 1 - exp_pred)
ax[0,1].plot(linear_space.T[0], pred)
ax[1,1].plot(linear_space.T[0], 1/pred)

plt.show()