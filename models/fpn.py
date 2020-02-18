import torch
import torch.nn as nn
import torch.nn.functional as F

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class FPN(nn.Module):
    def __init__(self, block, num_classes):
        super(FPN, self).__init__()
        self.expansion = block.expansion
        self.conv4 = nn.Conv2d(512*self.expansion, 64*self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256*self.expansion, 64*self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128*self.expansion, 64*self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(64*self.expansion, 64*self.expansion, kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(64*self.expansion, 64*self.expansion, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64*self.expansion, 64*self.expansion, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(64*self.expansion, 64*self.expansion, kernel_size=3, stride=1, padding=1)

        self.bottleneck3 = ScalaNet(64*self.expansion, 512*self.expansion, 2)
        self.bottleneck2 = ScalaNet(64*block.expansion, 512*block.expansion, 4)
        self.bottleneck1 = ScalaNet(64*block.expansion, 512*block.expansion, 8)

        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, feature_map4, feature_map3, feature_map2, feature_map1):
        p4 = self.conv4(feature_map4)

        p3 = F.interpolate(p4, size=(feature_map3.size(2), feature_map3.size(3)), mode='bilinear') + self.conv3(feature_map3)
        p3 = self.smooth3(p3)
        feat3 = self.bottleneck3(p3).view(-1, 512*self.expansion)
        label3 = self.fc3(feat3)

        p2 = F.interpolate(p3, size=(feature_map2.size(2), feature_map2.size(3)), mode='bilinear') + self.conv2(feature_map2)
        p2 = self.smooth2(p2)
        feat2 = self.bottleneck2(p2).view(-1, 512*self.expansion)
        label2 = self.fc2(feat2)

        p1 = F.interpolate(p2, size=(feature_map1.size(2), feature_map1.size(3)), mode='bilinear') + self.conv1(feature_map1)
        p1 = self.smooth1(p1)
        feat1 = self.bottleneck1(p1).view(-1, 512*self.expansion)
        label1 = self.fc1(feat1)
        
        return [label1, label2, label3]