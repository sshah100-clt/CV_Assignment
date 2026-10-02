"""Model definitions. All models are CNNs."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class TNet(nn.Module):
    """The starter baseline: one 3x3 convolution, max-pooling and a linear classifier."""

    def __init__(self, num_classes, in_channels, img_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        side = (img_size - 2) // 4
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * side * side, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class SqueezeExcite(nn.Module):
    """Channel attention (Hu et al., 2018): each channel is rescaled by a
    weight in (0, 1) computed from the globally pooled features."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class AttentionPool(nn.Module):
    """Soft spatial attention pooling: a 1x1 convolution scores each location,
    a softmax turns the scores into weights, and the output is the weighted
    average of the feature vectors. Average pooling is the case of equal weights."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        weights = self.score(x).view(b, 1, h * w).softmax(-1)
        return (x.view(b, c, h * w) * weights).sum(-1)


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SimpleCNN(nn.Module):
    """VGG-style CNN for training from scratch. Each stage has two 3x3
    convolutions with BatchNorm (optionally squeeze-and-excitation) and 2x2
    max-pooling. Global pooling makes the network independent of input size."""

    def __init__(self, num_classes, in_channels, widths, dropout, attention, pool):
        super().__init__()
        layers, cin = [], in_channels
        for w in widths:
            layers += [conv_bn_relu(cin, w), conv_bn_relu(w, w)]
            if attention == 'se':
                layers.append(SqueezeExcite(w))
            layers.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = AttentionPool(cin) if pool == 'attention' else nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


PLACES365_URL = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'


def resnet18_places365():
    """ResNet-18 trained on Places365 (1.8M scene images, 365 classes) by its authors."""
    model = models.resnet18(num_classes=365)
    path = Path(torch.hub.get_dir()) / 'checkpoints' / 'resnet18_places365.pth.tar'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(PLACES365_URL, str(path))
    state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})  # saved with DataParallel
    return model


def build_model(cfg, num_classes):
    """Returns the model and its classification layer, which the optimizer
    treats separately from the (possibly pretrained) rest of the network."""
    m, d = cfg['model'], cfg['data']
    if m['name'] == 'tnet':
        model = TNet(num_classes, d['channels'], d['img_size'])
        return model, model.classifier
    if m['name'] == 'simple_cnn':
        model = SimpleCNN(num_classes, d['channels'], m['widths'], m['dropout'], m['attention'], m['pool'])
        return model, model.classifier

    if m['pretrained'] == 'imagenet':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif m['pretrained'] == 'places365':
        model = resnet18_places365()
    else:
        model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new layer for our 16 classes
    return model, model.fc
