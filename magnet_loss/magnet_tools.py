from math import ceil
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

def compute_reps(model, X, chunk_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(len(X)) / chunk_size))
    #print(chunks)
    reps = []
    labels = []

    trainloader = DataLoader(X,
                         batch_size=chunks,
                         shuffle=False,
                         num_workers=4)

    for batch_idx, (img, target) in tqdm(enumerate(trainloader)):
        img = Variable(img).cuda()
        #pdb.set_trace()
        output, train_features = model(img)
        #train_features = model(img)
        embeddings = output.data
        #pdb.set_trace()
        reps.append(embeddings.cpu().numpy())
        #labels.append(target.cpu().numpy())
    return np.vstack(reps)

class ClusterBatchBuilder(object):
    """Sample minibatches for magnet loss."""
    def __init__(self, labels, k, m, d):

        if isinstance(labels, np.ndarray):
            self.num_classes = np.unique(labels).shape[0]
            self.labels = labels
        else:
            self.num_classes = np.unique(labels.numpy()).shape[0]
            self.labels = labels.numpy()

        self.k = k
        self.m = m
        self.d = d

        self.centroids = None

        if isinstance(labels, np.ndarray):
            self.assignments = np.zeros_like(labels, int)
        else:
            self.assignments = np.zeros_like(labels.numpy(), int)
                        
        self.cluster_assignments = {}
        self.cluster_classes = np.repeat(range(self.num_classes), k)
        self.example_losses = None
        self.cluster_losses = None
        self.has_loss = None


    def update_clusters(self, rep_data, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """
        # Lazily allocate array for centroids
        if self.centroids is None:
            self.centroids = np.zeros([self.num_classes * self.k, rep_data.shape[1]])


        for c in range(self.num_classes):

            class_mask = self.labels == c
            #pdb.set_trace()
            class_examples = rep_data[class_mask]
            kmeans = KMeans(n_clusters=self.k, init='k-means++', n_init=1, max_iter=max_iter)
            kmeans.fit(class_examples)

            # Save cluster centroids for finding impostor clusters
            start = self.get_cluster_ind(c, 0)
            stop = self.get_cluster_ind(c, self.k)
            self.centroids[start:stop] = kmeans.cluster_centers_

            # Update assignments with new global cluster indexes
            self.assignments[class_mask] = self.get_cluster_ind(c, kmeans.predict(class_examples))

        # Construct a map from cluster to example indexes for fast batch creation
        for cluster in range(self.k * self.num_classes):
            cluster_mask = self.assignments == cluster
            self.cluster_assignments[cluster] = np.flatnonzero(cluster_mask)


    def update_losses(self, indexes, losses):
        """
        Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses.
        """
        # Lazily allocate structures for losses
        if self.example_losses is None:
            self.example_losses = np.zeros_like(self.labels, float)
            self.cluster_losses = np.zeros([self.k * self.num_classes], float)
            self.has_loss = np.zeros_like(self.labels, bool)

        # Update example losses
        #print(indexes)
        #print(type(indexes))
        #indexes = np.array(indexes)
        #print(indexes)
        #print(type(indexes))
        losses = losses.data.cpu().numpy()
        #print(losses)
        #print(type(losses))
        #sprint(losses.view(12,-1))
        #exit()
        self.example_losses[indexes] = losses
        self.has_loss[indexes] = losses

        # Find affected clusters and update the corresponding cluster losses
        clusters = np.unique(self.assignments[indexes])
        for cluster in clusters:
            cluster_inds = self.assignments == cluster
            cluster_example_losses = self.example_losses[cluster_inds]

            # Take the average closs in the cluster of examples for which we have measured a loss
            self.cluster_losses[cluster] = np.mean(cluster_example_losses[self.has_loss[cluster_inds]])

    def gen_batch(self):
        """
        Sample a batch by first sampling a seed cluster proportionally to
        the mean loss of the clusters, then finding nearest neighbor
        "impostor" clusters, then sampling d examples uniformly from each cluster.
        
        The generated batch will consist of m clusters each with d consecutive
        examples.
        """

        # Sample seed cluster proportionally to cluster losses if available
        if self.cluster_losses is not None:
            p = self.cluster_losses / np.sum(self.cluster_losses)
            seed_cluster = np.random.choice(self.num_classes * self.k, p=p)
        else:
            seed_cluster = np.random.choice(self.num_classes * self.k)

        # Get imposter clusters by ranking centroids by distance
        sq_dists = ((self.centroids[seed_cluster] - self.centroids) ** 2).sum(axis=1)

        # Assure only clusters of different class from seed are chosen
        sq_dists[self.get_class_ind(seed_cluster) == self.cluster_classes] = np.inf

        # Get top impostor clusters and add seed
        clusters = np.argpartition(sq_dists, self.m-1)[:self.m-1]
        clusters = np.concatenate([[seed_cluster], clusters])

        # Sample examples uniformly from cluster
        batch_indexes = np.empty([self.m * self.d], int)
        for i, c in enumerate(clusters):
            x = np.random.choice(self.cluster_assignments[c], self.d, replace=False)
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x

        # Translate class indexes to index for classes within the batch
        class_inds = self.get_class_ind(clusters)
        batch_class_inds = []
        inds_map = {}
        class_count = 0
        for c in class_inds:
            if c not in inds_map:
                inds_map[c] = class_count
                class_count += 1
            batch_class_inds.append(inds_map[c])

        return batch_indexes, np.repeat(batch_class_inds, self.d)

    def get_cluster_ind(self, c, i):
        """
        Given a class index and a cluster index within the class
        return the global cluster index
        """
        return c * self.k + i

    def get_class_ind(self, c):
        """
        Given a cluster index return the class index.
        """
        return c / self.k
