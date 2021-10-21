import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs
import matplotlib.animation as animation



# #############################################################################
# generate cluster
def gen_clusters(no_clusters):
    gen_x, gen_y , gen_c = make_blobs(n_samples=50*no_clusters, centers=no_clusters, cluster_std=1.2, return_centers=True,
                                      random_state=216)
    return gen_x, gen_y , gen_c


# #############################################################################
# my k mean algorithm
def my_k_mean(k, max_iter):
    # init center
    np.random.seed(999)
    center_old = x[np.random.randint(x.shape[0], size=k)]

    label_old = kmean_assign(k, center_old)
    label_new, center_new = label_old, center_old

    fig_vis = plt.figure(1)
    ax_vis = fig_vis.add_subplot(1, 1, 1)
    ax_vis.scatter(x[:, 0], x[:, 1])
    plt.pause(5)
    ax_vis.cla()
    plot_clusters(ax_vis, label_new, center_new)
    ax_vis.set_title('iteration: 1')
    plt.pause(0.3)
    for update_iter in range(max_iter):
        center_new  = kmean_update(k, label_old)
        label_new   = kmean_assign(k, center_new)
        if (label_new == label_old).all():
            break
        else:
            label_old = label_new
        ax_vis.cla()
        plot_clusters(ax_vis, label_new, center_new)
        ax_vis.set_title('iteration: {}'.format(update_iter+1))
        plt.pause(0.3)
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
    ax.set_prop_cycle(color=[cm(1.*i/k) for i in range(k)])

    for cluster in range(k):
        cluster_member = (label == cluster)
        ax.scatter(x[cluster_member, 0], x[cluster_member, 1])
    for cluster in range(k):
        ax.scatter(center[cluster, 0], center[cluster, 1], s=100, c='k', marker='X')
    vor = Voronoi(center)
    voronoi_plot_2d(vor, show_points=False, ax=ax)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])


# #############################################################################
# #############################################################################
no_gen_cluster = 4
max_iteration = 30

# generate clusters
x, y_true, c_true = gen_clusters(no_gen_cluster)
k = 4
y_pred, c_pred = my_k_mean(k, max_iteration)

# plot real cluster

fig = plt.figure(2)
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x[:, 0], x[:, 1])
ax1.set_title('data')

ax2 = fig.add_subplot(1, 2, 2)
plot_clusters(ax2, y_pred, c_pred)
ax2.set_title('k-mean clusters')

plt.show()

