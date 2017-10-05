FROM pytorch/pytorch
MAINTAINER Vithursan Thangarasa

RUN \
	apt-get -qq -y update && apt-get -y install && \
	apt-get -y install ipython ipython-notebook python-tk

RUN \
	pip install -U numpy \
	jupyter \
	matplotlib \
	seaborn \
	pandas \
	tqdm \
	pillow \
	packaging \
	appdirs \
	sklearn \
	scipy

COPY ./ /root/MagnetLoss

WORKDIR /root/MagnetLoss

CMD /bin/bash
