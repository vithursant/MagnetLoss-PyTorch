import torch.nn as nn
import torch.nn.functional as F

cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, depth, num_classes=10):
        assert depth in cfg, 'Error: model depth invalid or undefined!'
        
        super(VGG, self).__init__()
        self.feature_extractor = self._make_layers(cfg[depth])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.classifier(x)
        return x, features

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x_cfg in config:
            if x_cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x_cfg, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x_cfg),
                           nn.ReLU(inplace=True)]
                in_channels = x_cfg
        return nn.Sequential(*layers)