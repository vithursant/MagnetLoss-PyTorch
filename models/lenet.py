import torch
import torch.nn as nn
import torch.nn.functional as F

# Build Network
class LeNet(nn.Module):

	def __init__(self):
		'''
			Define the initialization function of LeNet, this function defines
			the basic structure of the neural network
		'''

		# Inherited the parent class initialization method, that is, first
		# run nn.Module initialization function
		super(LeNet, self).__init__()
		# Define convolution layer: input 1 channel (grayscale) picture,
		# output 6 feature map, convolution kernel 5x5
		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
		# Define convolution layer: enter 6 feature maps, output 16 feature
		# maps, convolution kernel 5x5
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 16 * 5 * 5 nodes connected to 120 nodes
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 120 nodes connected to 84 nodes
		self.fc2 = nn.Linear(120, 84)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 84 nodes connected to 10 nodes
		self.fc3 = nn.Linear(84, 10)

		self.layer1 = None
		self.layer2 = None
		self.features = None

	def forward(self, x):
		'''
			Define the forward propagation function and automatically generates
			the backward propagation function (autograd)
		'''

		# Input x -> conv1 -> relu -> 2x2 the largest pool of windows ->
		# update to x
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		layer1 = x

		# Input x -> conv2 -> relu -> 2x2 window maximum pooling -> update
		# to x
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		layer2 = x

		# The view function changes the tens x into a one-dimensional vector
		# form with the total number of features unchanged, ready for the
		# fully connected layer
		x = x.view(-1, self.num_flat_features(x))
		features = x
		#print(features)
		#features.register_hook(print)

		# Input x -> fc1 -> relu, update to x
		x = F.relu(self.fc1(x))
		fc1 = x

		# Input x -> fc2 -> relu, update to x
		x = F.relu(self.fc2(x))
		fc2 = x

		# Input x -> fc3 -> relu, update to x
		x = self.fc3(x)
		fc3 = x

		return F.softmax(x), features

	def num_flat_features(self, x):
		'''
			Calculate the total tensor x feature amount
		'''

		size = x.size()[1:] # All dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s

		return num_features

	def name(self):
		return 'lenet'

class LeNet3D(nn.Module):
	"""For cifar-datasets"""
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		features = out

		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)

		return out, features

	def name(self):
		return 'lenet3D'

class FashionLeNet(nn.Module):
	"""For Mnist-datasets"""
	def __init__(self):
		super(FashionLeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(16*4*4, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		features = out

		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out, features

	def name(self):
		return 'fashionlenet'