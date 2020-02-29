'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def BasicBlock_functional_forward(x, weights, stride):
        
    out = F.conv2d(x, weights[0], stride=stride, padding=1)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[1], bias=weights[2],
                               training=True)
    out = F.relu(out, inplace=True)

    out = F.conv2d(out, weights[3], stride=1, padding=1)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[4], bias=weights[5],
                               training=True)
    if len(weights) == 9:
        shortcut_out = F.conv2d(x, weights[6], stride=stride)
        shortcut_out = F.batch_norm(shortcut_out, torch.zeros(shortcut_out.data.size(1)).cuda(), torch.ones(shortcut_out.data.size(1)).cuda(),
                                        weights[7], weights[8],
                                        training=True)
        out += shortcut_out

    else:
        out += x
    out = F.relu(out, inplace=True)
    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def Bottleneck_functional_forward(x, weights, stride):

    out = F.conv2d(x, weights[0])
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[1], bias=weights[2],
                               training=True)
    out = F.relu(out, inplace=True)
        
    out = F.conv2d(x, weights[3], stride=stride, padding=1)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[4], bias=weights[5],
                               training=True)
    out = F.relu(out, inplace=True)

    out = F.conv2d(out, weights[6])
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[7], bias=weights[8],
                               training=True)

    if len(weights) == 12:
        shortcut_out = F.conv2d(x, weights[9], stride=stride)
        shortcut_out = F.batch_norm(shortcut_out, torch.zeros(shortcut_out.data.size(1)).cuda(), torch.ones(shortcut_out.data.size(1)).cuda(),
                                        weights[10], weights[11],
                                        training=True)
        out += shortcut_out

    else:
        out += x
    out = F.relu(out, inplace=True)
    return out


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


def ScalaNet_functional_forward(x, weights, stride):

    out = F.conv2d(x, weights[0], bias=weights[1], stride=1)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[2], bias=weights[3],
                               training=True)
    out = F.relu(out, inplace=True)
        
    out = F.conv2d(out, weights[4], bias=weights[5], stride=stride)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[6], bias=weights[7],
                               training=True)
    out = F.relu(out, inplace=True)

    out = F.conv2d(out, weights[8], bias=weights[9], stride=1)
    out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                               weight=weights[10], bias=weights[11],
                               training=True)
    out = F.relu(out, inplace=True)
    out = F.avg_pool2d(out, kernel_size=4, stride=4)


    return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.block = block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.bottleneck_1 = ScalaNet(64*block.expansion, 512*block.expansion, 8)
        self.bottleneck_2 = ScalaNet(128*block.expansion, 512*block.expansion, 4)
        self.bottleneck_3 = ScalaNet(256*block.expansion, 512*block.expansion, 2)

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, weights=None, train=True, baseline=False):

        if not train:
            return self.predict(x)

        if baseline:
            return self.forward_bs(x, train=train)
        if weights:
            return self.forward_weights(x, weights)

        out = F.relu(self.bn1(self.conv1(x)))

        outfeat1 = self.layer1(out)
        feat1 = self.bottleneck_1(outfeat1).view(out.size(0), -1)
        out1 = self.fc1(feat1)

        outfeat2 = self.layer2(outfeat1)
        feat2 = self.bottleneck_2(outfeat2).view(out.size(0), -1)
        out2 = self.fc2(feat2)

        outfeat3 = self.layer3(outfeat2)
        feat3 = self.bottleneck_3(outfeat3).view(out.size(0), -1)
        out3 = self.fc3(feat3)

        outfeat4 = self.layer4(outfeat3)
        out = F.avg_pool2d(outfeat4, 4).view(out.size(0), -1)
        out = self.linear(out)

        return [out1, out2, out3, out], [outfeat4, outfeat3, outfeat2, outfeat1]

    def forward_weights(self, x, weights):
        out = F.conv2d(x, weights['module.conv1.weight'], stride=1, padding=1)
        out = F.batch_norm(out, torch.zeros(out.data.size(1)).cuda(), torch.ones(out.data.size(1)).cuda(),
                           weights['module.bn1.weight'], weights['module.bn1.bias'],
                           training=True)
        out = F.relu(out, inplace=True)

        for stage_num in range(1, len(self.num_blocks)+1):
            for block_num in range(self.num_blocks[stage_num-1]):
                if self.block == BasicBlock:
                    if block_num == 0 and stage_num != 1:
                        out = BasicBlock_functional_forward(out, 
                            weights=[weights['module.layer{:d}.0.conv1.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.bn1.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.bn1.bias'.format(stage_num)],
                            weights['module.layer{:d}.0.conv2.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.bn2.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.bn2.bias'.format(stage_num)],
                            weights['module.layer{:d}.0.shortcut.0.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.shortcut.1.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.shortcut.1.bias'.format(stage_num)]], stride=2
                            )
                    else:
                        out = BasicBlock_functional_forward(out,
                            weights=[weights['module.layer{:d}.{:d}.conv1.weight'.format(stage_num, block_num)], 
                            weights['module.layer{:d}.{:d}.bn1.weight'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.bn1.bias'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.conv2.weight'.format(stage_num, block_num)], 
                            weights['module.layer{:d}.{:d}.bn2.weight'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.bn2.bias'.format(stage_num, block_num)]], stride=1
                            )
                else:
                    if block_num == 0:        
                        out = Bottleneck_functional_forward(out, 
                            weights=[weights['module.layer{:d}.0.conv1.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.bn1.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.bn1.bias'.format(stage_num)],
                            weights['module.layer{:d}.0.conv2.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.bn2.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.bn2.bias'.format(stage_num)],
                            weights['module.layer{:d}.0.conv3.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.bn3.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.bn3.bias'.format(stage_num)],
                            weights['module.layer{:d}.0.shortcut.0.weight'.format(stage_num)], 
                            weights['module.layer{:d}.0.shortcut.1.weight'.format(stage_num)],
                            weights['module.layer{:d}.0.shortcut.1.bias'.format(stage_num)]], stride=2
                            )
                    else:
                        out = Bottleneck_functional_forward(out,
                            weights=[weights['module.layer{:d}.{:d}.conv1.weight'.format(stage_num, block_num)], 
                            weights['module.layer{:d}.{:d}.bn1.weight'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.bn1.bias'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.conv2.weight'.format(stage_num, block_num)], 
                            weights['module.layer{:d}.{:d}.bn2.weight'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.bn2.bias'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.conv3.weight'.format(stage_num, block_num)], 
                            weights['module.layer{:d}.{:d}.bn3.weight'.format(stage_num, block_num)],
                            weights['module.layer{:d}.{:d}.bn3.bias'.format(stage_num, block_num)]], stride=1
                            )

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.linear(out, weights['module.linear.weight'], weights['module.linear.bias'])

        return out

    def forward_bs(self, x, train=True):

        if not train:
            return self.predict(x)

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        feat1 = self.bottleneck_1(out).view(out.size(0), -1)
        out1 = self.fc1(feat1)

        out = self.layer2(out)
        feat2 = self.bottleneck_2(out).view(out.size(0), -1)
        out2 = self.fc2(feat2)

        out = self.layer3(out)
        feat3 = self.bottleneck_3(out).view(out.size(0), -1)
        out3 = self.fc3(feat3)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4).view(out.size(0), -1)

        feat_teacher = out.detach()
        feat_teacher.requires_grad = False
        feat_loss = ((feat_teacher - feat1) ** 2 + (feat_teacher - feat2) ** 2 + (feat_teacher - feat3) ** 2).sum()


        out = self.linear(out)



        return [out1, out2, out3, out], feat_loss

    def predict(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)


        out = self.layer2(out)


        out = self.layer3(out)


        out = self.layer4(out)

        out = F.avg_pool2d(out, 4).view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
