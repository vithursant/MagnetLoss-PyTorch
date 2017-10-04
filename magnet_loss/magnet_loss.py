import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pdb

class MagnetLoss(nn.Module):
	def __init__(self, alpha=1.0):
		super(MagnetLoss, self).__init__()
		self.r = None
		self.classes = None
		self.clusters = None
		self.cluster_classes = None
		self.n_clusters = None
		self.alpha = alpha

	def forward(self, r, classes, m, d, alpha=1.0):
		INT_DTYPE = torch.cuda.IntTensor
		FLOAT_DTYPE = torch.cuda.FloatTensor

		self.r = r
		self.classes = torch.from_numpy(classes).type(torch.cuda.LongTensor)
		self.clusters, _ = torch.sort(torch.arange(0, float(m)).repeat(d))
		self.clusters = self.clusters.type(torch.cuda.IntTensor)
		self.cluster_classes = self.classes[0:m*d:d]
		self.n_clusters = m
		self.alpha = alpha
		#pdb.set_trace()
		'''
		self.clusters = np.repeat(np.arange(m, dtype=np.int32), d)
		self.classes = classes
		self.alpha = alpha
		self.r = r
		self.n_clusters = m
		self.cluster_classes = classes[0:m*d:d]
		'''
		# Take cluster means within the batch
		#cluster_examples = dynamic_partition(self.r, self.clusters, self.n_clusters)
		cluster_examples = dynamic_partition(self.r, self.clusters, self.n_clusters)
		#pdb.set_trace()
		#cluster_examples = torch.FloatTensor(cluster_examples)
		#print(cluster_examples)
		#cluster_means = [np.mean(x, axis=0) for x in cluster_examples]
		#cluster_examples = torch.FloatTensor(cluster_examples)
		#pdb.set_trace()
		cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])
		#pdb.set_trace()
		#print(cluster_means)
		#exit()

		#sample_costs = euclidean_square(np.array(cluster_means, dtype=object), self.r.data.cpu().numpy())
		#pdb.set_trace()
		sample_costs = compute_euclidean_distance(cluster_means, expand_dims(r, 1))
		#pdb.set_trace()
		#pdb.set_trace()
		#sample_costs = torch.FloatTensor(np.array(sample_costs, dtype=np.float32))
		#print(sample_costs)

		# Select distances of examples to their own centroid
		#print(np.arange(n_clusters, dtype=np.int32))
		#self.clusters = self.clusters.astype(np.int32)
		clusters_tensor = self.clusters.type(torch.cuda.FloatTensor)
		#n_clusters_tensor = torch.IntTensor(np.arange(self.n_clusters, dtype=np.int32))
		n_clusters_tensor = torch.arange(0, self.n_clusters).type(torch.cuda.FloatTensor)
		#pdb.set_trace()
		##pdb.set_trace()
		#print(clusters_tensor)
		#print(n_clusters_tensor)
		#intra_cluster_mask = comparison_mask(clusters_tensor, n_clusters_tensor).float()
		intra_cluster_mask = Variable(comparison_mask(clusters_tensor, n_clusters_tensor).type(torch.cuda.FloatTensor))
		#pdb.set_trace()
		#pdb.set_trace()
		#print(intra_cluster_mask)
		intra_cluster_costs = torch.sum(intra_cluster_mask * sample_costs, dim=1)
		#pdb.set_trace()
		#print(intra_cluster_costs)
		#pdb.set_trace()
		# Compute variance of intra-cluster distances
		#N = r.shape[0]
		N = r.size()[0]
		#pdb.set_trace()

		variance = torch.sum(intra_cluster_costs) / float(N - 1)
		#pdb.set_trace()

		var_normalizer = -1 / (2 * variance**2)
		#print(var_normalizer)
		#pdb.set_trace()

		# Compute numerator
		numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)
		#print(numerator)
		#pdb.set_trace()

		#print(classes)
		#self.classes = self.classes.astype(np.int32)
		#classes_tensor = torch.IntTensor(self.classes)
		#print(classes_tensor)

		#print(cluster_classes)
		#self.cluster_classes = self.cluster_classes.astype(np.int32)
		#cluster_classes_tensor = torch.IntTensor(self.cluster_classes)
		#print(cluster_classes_tensor)
		classes_tensor = self.classes.type(torch.cuda.FloatTensor)
		cluster_classes_tensor = self.cluster_classes.type(torch.cuda.FloatTensor)
		#pdb.set_trace()
		# Compute denominator
		diff_class_mask = Variable(comparison_mask(classes_tensor, cluster_classes_tensor).type(torch.cuda.FloatTensor))
		#pdb.set_trace()
		#print(diff_class_mask)
		diff_class_mask = 1 - diff_class_mask # Logical not on ByteTensor
		#pdb.set_trace()

		#denom_sample_costs = torch.exp(torch.stack([var_normalizer * sample_costs]))
		denom_sample_costs = torch.exp(var_normalizer * sample_costs)
		#print(denom_sample_costs)
		#pdb.set_trace()

		#diff_class_mask = diff_class_mask.float()
		denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)
		#pdb.set_trace()
		#print(denominator)

		#epsilon = Variable(torch.FloatTensor(1e-8))
		#epsilon = np.array([1e-8], dtype=np.float32)
		#epsilon = torch.FloatTensor(epsilon)

		#epsilon = Variable(epsilon)
		#print(epsilon.data.size())
		#epsilon = epsilon.expand_as(denominator)
		epsilon = 1e-8
		#pdb.set_trace()

		#relu_compute = numerator / (denominator + epsilon) + epsilon
		#pdb.set_trace()
		#print(type(relu_compute))
		losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
		#pdb.set_trace()
		#pdb.set_trace()
		#input = -torch.log(numerator / (denominator + epsilon) + epsilon.data), requires_grad=True)
		#exit()
		#m = torch.nn.ReLU()
		#pdb.set_trace()
		#input = Variable(-torch.log(relu_compute))
		#pdb.set_trace()
		#losses = m(input)
		#pdb.set_trace()
		#exit()
		#print(losses)
		#exit()
		total_loss = torch.mean(losses)
		#pdb.set_trace()
		#print(total_loss)
		#exit()
		return total_loss, losses		

def expand_dims(var, dim=0):
	""" Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
		var = torch.range(0, 9).view(-1, 2)
		torch.expand_dims(var, 0).size()
		# (1, 5, 2)
	"""
	sizes = list(var.size())
	sizes.insert(dim, 1)
	return var.view(*sizes)

def comparison_mask(a_labels, b_labels):
	return torch.eq(expand_dims(a_labels, 1), 
					expand_dims(b_labels, 0))

def dynamic_partition(X, partitions, n_clusters):

	#pdb.set_trace()
		#cluster_bin[i] = torch.
	#for i in range(n_clusters):
	#	for i in range(partitions):
	#		if partitions == i:
	#			torch.stack(r[])

	#cluster_bin = torch.stack([r[partitions==i] for i in range(n_clusters)])
	cluster_bin = torch.chunk(X, n_clusters)
	#pdb.set_trace()
	return cluster_bin

def compute_euclidean_distance(x, y):
	return torch.sum((x - y)**2, dim=2)

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    out = list(out)
    if num_groups is not None:
        default = (fillvalue, ) * n
        assert isinstance(num_groups, int)
        out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
    if shorten:
        assert fillvalue is None
        out = (tuple(e for e in each if e is not None) for each in out)
    return out

def index(l, i):
	return index(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]