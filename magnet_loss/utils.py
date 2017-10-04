#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from itertools import chain


# Model building blocks
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
'''

# Visualization

def moving_average(a, n=3) :
    # Adapted from http://stackoverflow.com/questions/14313510/does-numpy-have-a-function-for-calculating-moving-average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_smooth(history, name):
    plt.plot(history, 'c', moving_average(history, 20), 'b')
    plt.savefig(str(name) + '.eps')

def show_images(H):
    # make a square grid
    num = H.shape[0]
    rows = int(np.ceil(np.sqrt(float(num))))

    fig = plt.figure(1, [10, 10])
    grid = ImageGrid(fig, 111, nrows_ncols=[rows, rows])

    for i in range(num):
        grid[i].axis('off')
        grid[i].imshow(H[i], cmap='Greys')
        
    # Turn any unused axes off
    for j in range(i, len(grid)):
        grid[j].axis('off')


def plot_embedding(X, y, imgs=None, title=None, name=None):
    # Adapted from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # Add image overlays
    if imgs is not None and hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(imgs[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.savefig(str(name) + '.eps')

def zip_chain(a, b):
    return list(chain(*zip(a, b)))


def plot_metric(*args, **kwargs):

    name = args[0]
    plot_data = []
    for i in range(1, len(args), 2):
        metrics = args[i]
        d = [m[name] for m in metrics]
        color = args[i + 1]
        plot_data.extend(zip_chain(d, color * len(d)))

    plt.plot(*plot_data)
    if kwargs['title']:
        plt.title(kwargs['title'])
    plt.show()



# Evaluation

def compute_rand_index(emb, labels):
    """
    https://en.wikipedia.org/wiki/Rand_index
    """
    n = len(emb)
    k = np.unique(labels).size

    m = KMeans(k)
    m.fit(emb)
    emb_labels = m.predict(emb)

    agreements = 0
    for i, j in zip(*np.triu_indices(n, 1)):
        emb_same = emb_labels[i] == emb_labels[j]
        gt_same = labels[i] == labels[j]

        if emb_same == gt_same:
            agreements += 1

    return float(agreements) / (n * (n-1) / 2)


def unsupervised_clustering_accuracy(emb, labels):
    k = np.unique(labels).size
    kmeans = KMeans(n_clusters=k, max_iter=35, n_init=15, n_jobs=-1).fit(emb)
    emb_labels = kmeans.labels_
    G = np.zeros((k,k))
    for i in range(k):
        lbl = labels[emb_labels == i]
        uc = itemfreq(lbl)
        for uu, cc in uc:
            G[i,uu] = -cc
    A = linear_assignment_.linear_assignment(G)
    acc = 0.0
    for (cluster, best) in A:
        acc -= G[cluster,best]
    return acc / float(len(labels))
