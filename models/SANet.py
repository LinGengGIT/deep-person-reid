from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

__all__ = ['SattentionNet', 'sattention']


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)



class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class SattentionNet(nn.Module):
    def __init__(self, num_features=128, seq_len=15, norm=True, spanum=3, pretrained=1):
        super(SattentionNet, self).__init__()

        self.atn_height = 8
        self.atn_width = 4
        self.pretrained = pretrained
        self.spanum = spanum
        self.seq_len = seq_len
        self.num_features = num_features
        self.norm = norm

        self.conv1 = nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, self.spanum, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.spanum)

        self.softmax = BottleSoftmax()

        self.feat = nn.Linear(2048, self.num_features)

        if not self.pretrained:
            self.reset_params()


    def forward(self, x):
        x = x.view( (-1,2048)+x.size()[-2:] )

        atn = x
        atn = self.conv1(atn)
        atn = self.bn1(atn)
        atn = self.relu1(atn)
        atn = self.conv2(atn)
        atn = self.bn2(atn)
        atn = atn.view(-1,self.spanum, self.atn_height*self.atn_width)
        atn = self.softmax(atn)

        # Diversity Regularization
        reg = atn

        # Multiple Spatial Attention
        atn = atn.view(atn.size(0), self.spanum, 1, self.atn_height, self.atn_width)
        atn = atn.expand(atn.size(0), self.spanum, 2048, self.atn_height, self.atn_width)
        x = x.view(x.size(0), 1, 2048, self.atn_height, self.atn_width)
        x = x.expand(x.size(0), self.spanum, 2048, self.atn_height, self.atn_width)

        x = x * atn
        x = x.view(-1, 2048, self.atn_height, self.atn_width)
        x = F.avg_pool2d(x, x.size()[2:])*x.size(2)*x.size(3)
        x = x.view(-1, 2048)

        x = self.feat(x)
        x = x.view(-1, self.spanum, self.num_features)

        if self.norm:
            x = x / x.norm(2, 2).view(-1,self.spanum,1).expand_as(x)####expand_as
        #x = x.view(-1, self.seq_len, self.spanum, self.num_features)#####!

        return x, reg

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


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
    def __init__(self, num_classes):
        super(FeaturePart, self).__init__()
        self.classifier = nn.Linear(128*3, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # if len(x.shape) > 2:
        #     x = F.avg_pool2d(x, x.size()[2:])
        #     f = x.view(x.size(0), -1)
        #     y = self.classifier(f)
        # else:
        #     f = x
        #     y = self.classifier(f)

        # return y, f
        y = self.classifier(x)
        return y

class QualityPart(nn.Module):
    def __init__(self, planes=128):
        super(QualityPart, self).__init__()
        self.fc = nn.Linear(planes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x * 3 * 128
        x = self.fc(x)
        return x

class ST(nn.Module):
    def __init__(self,base_model,sattention,tattention,feature_model):
        super(ST,self).__init__()
        self.base_model=base_model
        self.sattention=sattention
        self.quality_model=tattention
        self.feature_model=feature_model
    def forward(self, x):
        if len(x.shape) > 2:
            _, image_feature = self.base_model(x)
            image_feature,reg=self.sattention(image_feature)
            score = self.quality_model(image_feature)

            return image_feature,score,reg#30*3*128,30*3,30*3*32
        else:
            return self.feature_model(x)

def stattention(num_classes, loss={'xent'}, seq_len=15, pretrained='pretrained_models/resnet50.pth'):
    base = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = torch.load(pretrained)
        base.load_state_dict(state_dict)
    feature = FeaturePart(num_classes)
    spa=SattentionNet()
    quality=QualityPart()
    model=ST(base,spa,quality,feature)
    return model