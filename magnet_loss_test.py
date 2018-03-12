import matplotlib
matplotlib.use('agg')
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

from models.vgg import VGG
from models.lenet import LeNet

from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import plot_embedding, plot_smooth

from utils.train_settings import parse_settings
from utils.sampler import SubsetSequentialSampler
from utils.average_meter import AverageMeter

from visualizer.visualizer import VisdomLinePlotter

from datasets.load_dataset import load_dataset

args = parse_settings()

def run_magnet_loss():
	'''
	Test function for the magnet loss
	'''
	m = 8
	d = 8
	k = 8
	alpha = 1.0
	batch_size = m * d

	global plotter
	plotter = VisdomLinePlotter(env_name=args.name)

	trainloader, testloader, trainset, testset, n_train = load_dataset(args)

	emb_dim = 2
	n_epochs = 15
	epoch_steps = len(trainloader)
	n_steps = epoch_steps * 15
	cluster_refresh_interval = epoch_steps

	if args.mnist:
		model = torch.nn.DataParallel(LeNet(emb_dim)).cuda()
	if args.cifar10:
		model = torch.nn.DataParallel(VGG(depth=16, num_classes=emb_dim))
	print(model)

	#model = EncoderCNN(64)
	#model.cuda()

	#model = VGG(11)
	#model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	#optimizer = optim.Adam(list(model.resnet.fc.parameters()), lr=args.lr)
	minibatch_magnet_loss = MagnetLoss()

	images = getattr(trainset, 'train_data')
	labels = getattr(trainset, 'train_labels')

	# Get initial embedding
	initial_reps = compute_reps(model, trainset, 400)

	if args.cifar10:
		labels = np.array(labels, dtype=np.float32)

	# Create batcher
	batch_builder = ClusterBatchBuilder(labels, k, m, d)
	batch_builder.update_clusters(initial_reps)

	batch_losses = []

	batch_example_inds, batch_class_inds = batch_builder.gen_batch()
	trainloader.sampler.batch_indices = batch_example_inds
	#pdb.set_trace()

	_ = model.train()

	losses = AverageMeter()

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

		batch_losses.append(batch_loss.data[0])

		if not i % 1000:
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

		losses.update(batch_loss, 1)

		# log avg values to somewhere
		if args.visdom:
			plotter.plot('loss', 'train', i, losses.avg.data[0])

	# Plot loss curve
	plot_smooth(batch_losses, "batch-losses")

if __name__ == '__main__':
	run_magnet_loss()
