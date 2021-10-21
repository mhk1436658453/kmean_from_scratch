import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs

no_real_cluster = 4


# #############################################################################
# generate cluster
def gen_clusters(no_clusters):
    x_, y, c = make_blobs(n_samples=10000, centers=no_clusters, cluster_std=0.7, return_centers=True)
    return x_, y, c


# #############################################################################
# my k mean algorithm
def my_k_mean(k, x):
    # set seed
    np.random.seed(0)
    pred_label = np.zeros(x.shape[0])
    cluster_center = x[np.random.randint(x.shape[0], size=k)]

    for sample, s in zip(x, range(x.shape[0])):
        dist_arr = np.zeros(k)
        for i, center in zip(range(k), cluster_center):
            dist_arr[i] = euclidean(sample, center)
        pred_label[s] = dist_arr.argmin()
    return pred_label, cluster_center


# #############################################################################
# plot clusters
def plot_clusters(ax, label, center):
    cm = plt.get_cmap('gist_rainbow')

    ax.set_prop_cycle(color=[cm(1. * i / no_real_cluster) for i in range(no_real_cluster)])

    for cluster in range(no_real_cluster):
        cluster_member = (label == cluster)
        ax.scatter(x[cluster_member, 0], x[cluster_member, 1])
    for cluster in range(no_real_cluster):
        ax.scatter(center[cluster, 0], center[cluster, 1], s=100, c='k', marker='X')
    vor = Voronoi(center)
    voronoi_plot_2d(vor, show_points=False, ax=ax)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])


# #############################################################################
# #############################################################################
# generate clusters
x, y_true, c_true = gen_clusters(no_real_cluster)
y_pred, c_pred = my_k_mean(no_real_cluster, x)

# plot real cluster

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1)
plot_clusters(ax1, y_true, c_true)
ax1.set_title('real clusters')

ax2 = fig.add_subplot(1, 2, 2)
plot_clusters(ax2, y_pred, c_pred)
ax2.set_title('k-mean clusters')

plt.show()





