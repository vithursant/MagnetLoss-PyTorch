FROM nvidia/cuda:9.0-cudnn7-devel

MAINTAINER Vithursan Thangarasa

ENV DEBIAN_FRONTEND=noninteractive

# Install some dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libssl-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        python2.7 \
        python2.7-dev \
        python-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python-pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python dependencies
COPY requirements.txt /root/MagnetLoss/requirements.txt
WORKDIR /root/MagnetLoss
RUN pip install --upgrade -r requirements.txt

# Install PyTorch
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
RUN pip install torchvision

COPY ./ /root/MagnetLoss

CMD /bin/bash
