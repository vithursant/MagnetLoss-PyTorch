# Magnet-Loss-PyTorch

PyTorch implementation of the Magnet Loss for Deep Metric Learning, based onthe following paper:

* [Metric Learning with Adaptive Density Discrimination](https://arxiv.org/pdf/1511.05939.pdf) by Oren Rippel, Piotr Dollar, Manohar Paluri, Lubomir Bourdev

## Table of Contents
* [Dataset](#dataset)
* [Installation](#installation)
* [Anaconda](#anaconda)
* [Docker](#docker)
* [Future Work](#future-work)

## Installation

The program requires the following dependencies (easy to install using pip3, Ananconda or Docker):

* python3
* pytorch (tested with 0.2)
* numpy
* matplotlib
* seaborn
* pandas
* tqdm
* pillow
* sklearn
* scipy
* visdom

## Anaconda

#### Anaconda: Installation

To install MagnetLoss in an Anaconda environment:

```sh
conda env create -f environment.yml
```

To activate Anaconda environment:

```sh
source activate magnet-loss
```

### Anaconda: Train

Train ConvNet with Magnet Loss on the local machine using MNIST dataset:

```sh
python magnet_loss_test.py --lr 1e-4 --batch-size 64 --mnist --dml
```

## Docker

### Docker: Installation

**Prerequisites: Docker installed on your machine. If you don't have Docker installed already, then go here to [Docker Setup](https://docs.docker.com/engine/getstarted/step_one/)**

To build Docker image:

```sh
docker build -t magnetloss:latest .
```

### Docker: Train
To deploy and train on Docker container:
```sh
docker run -it magnetloss:latest python --lr 1e-4 --batch-size 64 --mnist --dml
```
