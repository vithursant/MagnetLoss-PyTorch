import argparse

def parse_settings():

	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST SPLD')
	print(parser)
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
	                    help='input batch size for training (default: 32)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=50, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                    help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=20, metavar='N',
	                    help='how many batches to wait before logging training status')
	parser.add_argument('--spld', action='store_true', default=False,
						help='enables self-paced learning with diversity')
	parser.add_argument('--dml', action='store_true', default=False,
						help='enables deep metric learning')
	parser.add_argument('--curriculum-epochs', type=int, default=40, metavar='N',
						help='Number of curriculum epochs')
	parser.add_argument('--num-cluster', type=int, default=100, metavar='N',
						help='Number of clusters for clustering')
	parser.add_argument('--epoch-iters', type=int, default=10, metavar='N',
						help='Number of iterations per epoch')
	parser.add_argument('--num-classes', type=int, default=10, metavar='N',
						help='Number of classes in the dataset')
	parser.add_argument('--loss-weight', type=float, default=1.2e+6, metavar='LW',
						help='The loss weight')
	parser.add_argument('--curriculum-rate', type=float, default=0.03, metavar='CR',
						help='The curriculum learning rate')
	parser.add_argument('--decay-after-epochs', type=int, default=10, metavar='N',
						help='Decay after epochs')
	parser.add_argument('--magnet-loss', action='store_true', default=False,
						help='Enables the magnet loss for representation learning')
	parser.add_argument('--mnist', action='store_true', default=False,
						help='Use the mnist dataset')
	parser.add_argument('--cifar10', action='store_true', default=False,
						help='Use the CIFAR-10 dataset')
	parser.add_argument('--fashionmnist', action='store_true', default=False,
						help='Use the Fasnion MNIST dataset')
	parser.add_argument('--name', default='SPLD', type=str,
						help='name of experiment')
	parser.add_argument('--visdom', dest='visdom', action='store_true', default=False,
						help='Use visdom to track and plot')
	parser.add_argument('--mining', type=str, default='Hardest',
						help='Method to use for mining hard examples')
	parser.add_argument('--feature-size', type=int, default=64,
						help='size for embeddings/features to learn')
	parser.add_argument('--lr-freq', default=5, type=int,
						help='learning rate changing frequency (default: 5)')
	parser.add_argument('--tensorboard',
						help='Log progress to TensorBoard', action='store_true')
	parser.add_argument('--fea-freq', default=5, type=int,
						help='Refresh clusters (default: 5)')
	parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
						help='weight decay (default: 5e-4)')
	parser.add_argument('--no-augment', dest='augment', action='store_false',
						help='whether to use standard augmentation (default: True)')
	parser.add_argument('--droprate', default=0, type=float,
						help='dropout probability (default: 0.0)')
	parser.add_argument('--start-epoch', default=0, type=int,
						help='manual epoch number (useful on restarts)')
	parser.add_argument('--print-freq', '-p', default=10, type=int,
						help='print frequency (default: 10)')
	return parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')