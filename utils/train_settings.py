import argparse
import os
import shutil

def parse_settings():

	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Magnet Loss')
	print(parser)
	parser.add_argument('--batch-size', type=int, default=64,
	                    help='input batch size for training (default: 32)')
	parser.add_argument('--epochs', type=int, default=50,
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1e-3,
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--magnet-loss', action='store_true', default=False,
						help='Enables the magnet loss for representation learning')
	parser.add_argument('--mnist', action='store_true', default=False,
						help='Use the mnist dataset')
	parser.add_argument('--cifar10', action='store_true', default=False,
						help='Use the CIFAR-10 dataset')
	parser.add_argument('--fashionmnist', action='store_true', default=False,
						help='Use the Fasnion MNIST dataset')
	parser.add_argument('--visdom', dest='visdom', action='store_true', default=False, help='Use visdom to track and plot')
	parser.add_argument('--name', default='MagnetLoss', type=str,
						help='name of experiment')
	return parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')
