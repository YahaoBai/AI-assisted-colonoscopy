import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/network'))
from perception.resnet_backbone import ResNet, BasicBlock


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockResNet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mid_planes=None, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockResNet, self).__init__()
        if not mid_planes:
            mid_planes = planes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, mid_planes, stride)
        self.bn1 = norm_layer(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm_layer(planes),
        )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class UpResNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlockResNet(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BasicBlockResNet(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ColonoscopyNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, use_depth_decoder=True):
        super(ColonoscopyNet, self).__init__()
        self.use_depth_decoder = use_depth_decoder
        
        self.rgbFeatureExtractor = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, input_channels=3)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        
        self.actionGenerator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )
        
        if self.use_depth_decoder:
            self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
            self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
            self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
            self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
            self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x)
        
        action = self.actionGenerator(feature_rgb)
        
        if self.use_depth_decoder:
            output_depth = self.depthDecoder_up1(f5, f4)
            output_depth = self.depthDecoder_up2(output_depth, f3)
            output_depth = self.depthDecoder_up3(output_depth, f2)
            output_depth = self.depthDecoder_up4(output_depth, f1)
            output_depth = self.up(output_depth)
            output_depth = self.depthDecoder_outc(output_depth)
            return action, output_depth
        else:
            return action
