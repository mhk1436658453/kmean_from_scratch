import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs


no_real_cluster = 7
max_iteration = 30

# #############################################################################
# generate cluster
def gen_clusters(no_clusters):
    gen_x, gen_y , gen_c = make_blobs(n_samples=1000, centers=no_clusters, cluster_std=0.7, return_centers=True,
                                      random_state=215)
    return gen_x, gen_y , gen_c


# #############################################################################
# my k mean algorithm
def my_k_mean(k, max_iter):
    # init center
    #np.random.seed(0)
    center_old = x[np.random.randint(x.shape[0], size=k)]

    label_old = kmean_assign(k, center_old)
    label_new, center_new = label_old, center_old

    for update_iter in range(max_iter):
        center_new  = kmean_update(k, label_old)
        label_new   = kmean_assign(k, center_new)
        if (label_new == label_old).all():
            break
        else:
            label_old = label_new
    print(update_iter)
    return label_new, center_new


# #############################################################################
#
def kmean_assign(no_of_cluster, center):
    assigned_label = np.zeros(x.shape[0])

    for sample, s in zip(x, range(x.shape[0])):
        dist_arr = np.zeros(no_of_cluster)
        for i , c in zip(range(no_of_cluster), center):
            dist_arr[i] = euclidean(sample, c)
        assigned_label[s] = dist_arr.argmin()
    return assigned_label


def kmean_update(no_of_cluster, labels):
    new_cluster_center = np.zeros((no_of_cluster, x.shape[1]))
    for cluster_no in range(no_of_cluster):
        cluster_member = (labels == cluster_no)
        new_cluster_center[cluster_no] = x[cluster_member].mean(axis=0)
    return new_cluster_center


# #############################################################################
# plot clusters
def plot_clusters(ax, label, center):
    cm = plt.get_cmap('gist_rainbow')

    ax.set_prop_cycle(color=[cm(1.*i/no_real_cluster) for i in range(no_real_cluster)])

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
y_pred, c_pred = my_k_mean(no_real_cluster, max_iteration)

# plot real cluster

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1)
plot_clusters(ax1, y_true, c_true)
ax1.set_title('real clusters')

ax2 = fig.add_subplot(1, 2, 2)
plot_clusters(ax2, y_pred, c_pred)
ax2.set_title('k-mean clusters')

plt.show()

