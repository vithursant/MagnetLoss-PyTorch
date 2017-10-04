import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.init as init

import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torchvision.models as models
from models.vgg_cifar import VGG

from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import plot_embedding, plot_smooth

from utils.train_settings import parse_settings
from utils.sampler import SubsetSequentialSampler

from datasets.load_dataset import load_dataset

args = parse_settings()

# Build Network
class LeNet(nn.Module):

	def __init__(self, emb_dim):
		self.emb_dim = emb_dim

		'''
		Define the initialization function of LeNet, this function defines
		the basic structure of the neural network
		'''

		# Inherited the parent class initialization method, that is, first
		# run nn.Module initialization function
		super(LeNet, self).__init__()
		#self.embed = nn.Embedding(64, 2)
		# Define convolution layer: input 1 channel (grayscale) picture,
		# output 6 feature map, convolution kernel 5x5
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # input is 28x28. padding=2 for same padding
		# Define convolution layer: enter 6 feature maps, output 16 feature
		# maps, convolution kernel 5x5
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # feature map size is 14*14 by pooling
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 16 * 5 * 5 nodes connected to 120 nodes
		#self.fc1 = nn.Linear(64*7*7, self.emb_dim)
		self.emb = nn.Linear(64*7*7, self.emb_dim)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 120 nodes connected to 84 nodes
		#self.fc2 = nn.Linear(120, 84)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 84 nodes connected to 10 nodes
		#self.fc3 = nn.Linear(84, 10)

		#self.embedding = nn.Linear(2, self.emb_dim)

		#self.norm_emb =  L2_Normalize(self.emb)

		self.layer1 = None
		self.layer2 = None
		self.features = None
		self.embeddings = None
		self.norm_embeddings = None

	def forward(self, x):
		'''
			Define the forward propagation function and automatically generates
			the backward propagation function (autograd)
		'''

		# Input x -> conv1 -> relu -> 2x2 the largest pool of windows ->
		# update to x
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		self.layer1 = x

		# Input x -> conv2 -> relu -> 2x2 window maximum pooling -> update
		# to x
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		self.layer2 = x

		# The view function changes the tens x into a one-dimensional vector
		# form with the total number of features unchanged, ready for the
		# fully connected layer
		x = x.view(-1, self.num_flat_features(x))
		#x = x.view(-1, self.num_flat_features(x))
		self.features = x
		#self.features = features
		#features.register_hook(print)

		# Input x -> fc1 -> relu, update to x
		#x = F.relu(self.fc1(x))
		x = self.emb(x)
		#fc1 = x
		embeddings = x
		#pdb.set_trace()
		# Input x -> fc2 -> relu, update to x
		#x = F.relu(self.fc2(x))
		#fc2 = x

		# Input x -> fc3 -> relu, update to x
		#x = self.fc3(x)
		#fc3 = x

		#pdb.set_trace()

		#x = self.embedding(x)
		#pdb.set_trace()
		#x.register_hook(print)
		#x = self.norm_emb(x)

		#norm_emb = x
		norm_embeddings = self.l2_normalize(x, 1)
		#pdb.set_trace()
		return embeddings, norm_embeddings

	def num_flat_features(self, x):
		'''
			Calculate the total tensor x feature amount
		'''

		size = x.size()[1:] # All dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s

		return num_features

	def l2_normalize(self, x, dim):

	    if not (isinstance(x, torch.DoubleTensor) or isinstance(x, torch.FloatTensor)):
	        x = x.float()

	    if len(x.size()) == 1:
	        x = x.view(1, -1)

	    norm = torch.sqrt(torch.sum(x * x, dim=dim))
	    norm = norm.view(-1, 1)

	    return torch.div(x, norm)

	def name(self):
		return 'lenet-magnet'

def run_magnet_loss():
	'''
	Test function for the magnet loss
	'''
	m = 8
	d = 8
	k = 3
	alpha = 1.0
	batch_size = m * d


	trainloader, testloader, trainset, testset, n_train = load_dataset(args)

	emb_dim = 2
	n_epochs = 15
	epoch_steps = len(trainloader)
	n_steps = epoch_steps * 15
	cluster_refresh_interval = epoch_steps

	model = LeNet(emb_dim)
	print(model)
	model.cuda()

	#model = EncoderCNN(64)
	#model.cuda()

	#model = VGG(11)
	#model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	#optimizer = optim.Adam(list(model.resnet.fc.parameters()), lr=args.lr)
	minibatch_magnet_loss = MagnetLoss()

	images = getattr(trainset, 'train_data')
	labels = getattr(trainset, 'train_labels')
	#pdb.set_trace()

	# Get initial embedding
	initial_reps = compute_reps(model, trainset, 400)

	if args.cifar10:
		labels = np.array(labels, dtype=np.float32)

	# Create batcher
	batch_builder = ClusterBatchBuilder(labels, k, m, d)
	batch_builder.update_clusters(initial_reps)

	batch_losses = []

	batch_example_inds, batch_class_inds = batch_builder.gen_batch()
	trainloader.sampler.batch_example_inds = batch_example_inds

	_ = model.train()

	for i in tqdm(range(n_steps)):
		for batch_idx, (img, target) in enumerate(trainloader):

			img = Variable(img).cuda()
			#pdb.set_trace()
			target = Variable(target).cuda()

			optimizer.zero_grad()
			output, features = model(img)

			batch_loss, batch_example_losses = minibatch_magnet_loss(output,
													   	  			batch_class_inds, 
													   	  			m, 
													   	  			d, 
													   	  			alpha)
			batch_loss.backward()
			optimizer.step()

		# Update loss index
		batch_builder.update_losses(batch_example_inds,
									batch_example_losses)

		batch_losses.append(batch_loss)

		if not i % 10:
			print (i, batch_loss)

		if not i % cluster_refresh_interval:
			print("Refreshing clusters")
			reps = compute_reps(model, trainset, 400)
			batch_builder.update_clusters(reps)

		if not i % 2000:
			n_plot = 10000
			plot_embedding(compute_reps(model, trainset, 400)[:n_plot], labels[:n_plot], name=i)

		batch_example_inds, batch_class_inds = batch_builder.gen_batch()
		trainloader.sampler.batch_indices = batch_example_inds

	# Plot loss curve
	#pdb.set_trace()
	#plot_smooth(batch_losses, "batch-losses")

if __name__ == '__main__':
	run_magnet_loss()
