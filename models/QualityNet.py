from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['ResNet']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, no_classifier=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.no_classifier = no_classifier

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 64 x 64 x 32
        x = self.layer1(x)
        # 256 x 64 x 32
        x = self.layer2(x)
        # 512 x 32 x 16
        x = self.layer3(x)
        # 1024 x 16 x 8
        out = self.layer4(x)
        # 2048 x 8 x 4
        if self.no_classifier:
            return x, out

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class FeaturePart(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(FeaturePart, self).__init__()
        self.loss = loss
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        if len(x.shape)>2:
            x = F.avg_pool2d(x, x.size()[2:])
            f = x.view(x.size(0), -1)
        else:
            f = x
        # if not self.training:
        #     return f
        y = self.classifier(f)
        return y, f

        # if self.loss == {'xent'}:
        #     return y
        # elif self.loss == {'xent', 'htri'}:
        #     return y, f
        # elif self.loss == {'cent'}:
        #     return y, f
        # elif self.loss == {'ring'}:
        #     return y, f
        # else:
        #     raise KeyError("Unsupported loss: {}".format(self.loss))

class QualityPart(nn.Module):
    def __init__(self, inplanes=1024, planes=2048):
        super(QualityPart, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc = nn.Linear(planes, 1)
        self.bn = nn.BatchNorm1d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        x = self.bn(x)
        return x

class QAN(nn.Module):
    """docstring for QAN"""
    def __init__(self, base_model, feature_model, quality_model, loss={'xent'}):
        super(QAN, self).__init__()
        self.base_model = base_model
        self.feature_model = feature_model
        self.quality_model = quality_model
    def forward(self, x):
        if len(x.shape)>2:
            middle_feature, image_feature = self.base_model(x)
            y,image_feature = self.feature_model(image_feature)
            score = self.quality_model(middle_feature)
            return y, image_feature, score
        else:
            return self.feature_model(x)
        
        
def qualitynet_res50(num_classes, loss={'xent'}, pretrained='pretrained_models/resnet50.pth'):
    base = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = torch.load(pretrained)
        base.load_state_dict(state_dict)
    feature = FeaturePart(num_classes, loss)
    quality = QualityPart()
    model = QAN(base,feature, quality, loss)
    return model