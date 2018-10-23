import os
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['resnext50', 'resnext101', 'resnext101_64', 'resnext152']

model_urls = {
    'resnext50': 'https://nizhib.ai/models/resnext50-316de15a.pth',
    'resnext101': 'https://nizhib.ai/models/resnext101-a04abaaf.pth'
}


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, baseWidth, cardinality, layers, num_classes):
        super(ResNeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth,
                            self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(x.size())
        x = self.maxpool(x)

        x = self.layer1(x)
        print(x.size())

        x = self.layer2(x)
        print(x.size())

        x = self.layer3(x)
        print(x.size())

        x = self.layer4(x)
        print(x.size())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50(num_classes=1000, pretrained=True, in_channels=3):
    """Constructs a ResNeXt-50 model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        path = os.path.join(os.getenv("HOME"), '' )
        pretrained_parallel_state_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_normal_state_dict = dict()
        for key in pretrained_parallel_state_dict.keys():
            pretrained_normal_state_dict[key.split('module.')[1]] = pretrained_parallel_state_dict[key]
        model.load_state_dict(pretrained_normal_state_dict)
    return model


def resnext101(num_classes=1000, pretrained=True):
    """Constructs a ResNeXt-101 (32x4d) model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101']))
    return model


def resnext101_64(num_classes=1000):
    """Constructs a ResNeXt-101 (64x4d) model."""
    model = ResNeXt(Bottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def resnext152(num_classes=1000):
    """Constructs a ResNeXt-152 (32x4d) model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = resnext50()
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
