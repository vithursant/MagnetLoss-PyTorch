import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

class BasickBlock(nn.Module):

    def __init__(self, n_in, n_out, stride=1):
        super(BasickBlock, self).__init__()
        self.connection = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_in, n_out, 3, stride, 1, bias=False)),
            ('norm1', nn.BatchNorm2d(n_out)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False)),
            ('norm2', nn.BatchNorm2d(n_out)),
        ]))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, 1, stride, bias=False),
            nn.BatchNorm2d(n_out),
        )
        self.stride = stride

    def forward(self, x):
        mapping = self.connection(x)
        if self.stride != 1:
            x = self.downsample(x)
        return self.relu(mapping + x)


class ResidualBlock(nn.Module):

    def __init__(self, n_in, n_out, n_block, stride=1):
        super(ResidualBlock, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module('block0', BasickBlock(n_in, n_out, stride))
        for i in range(n_block - 1):
            self.blocks.add_module('block{}'.format(i + 1), BasickBlock(n_out, n_out))

    def forward(self, x):
        return self.blocks(x)


class ResNetCifar10(nn.Module):

    def __init__(self, n_block=3):
        super(ResNetCifar10, self).__init__()
        ch = [16, 32, 64]
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)),
            ('norm1', nn.BatchNorm2d(ch[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('resb1', ResidualBlock(ch[0], ch[0], n_block)),
            ('resb2', ResidualBlock(ch[0], ch[1], n_block, 2)),
            ('resb3', ResidualBlock(ch[1], ch[2], n_block, 2)),
            ('avgpl', nn.AvgPool2d(8)),
        ]))
        self.fc = nn.Linear(ch[2], 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.fc(x)
        
        return F.log_softmax(x), features