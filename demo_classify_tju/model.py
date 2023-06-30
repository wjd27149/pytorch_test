import torch
import torch.nn as nn
import torchsummary

# 18层和34 层网络结构
class BasicBlock(nn.Module):
    #expansion 用来表示resnet的实线和虚线，让BasicBlock 满足实线和虚线两种
    expansion = 1
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None, **kwargs):
        super(BasicBlock,self).__init__()
        # 求通道out_channels公式（H-kernelsize + 2*padding）/ stride + 1
        # 因为使用 bn，设置bias对bn起不到作用，所以不设置bias

        self.conv1 = nn.Conv2d(in_channels= in_channel, out_channels= out_channel, 
                               kernel_size= 3, stride= stride, padding= 1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels= out_channel, out_channels= out_channel, 
                               kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    # 正向传播中 输入 为 x
    def forward(self, x):
        # identity 为残差结构上的捷径输出
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

#设置expansion = 4，使其满足resnet 50/101/152
class Bottleneck(nn.Module):
    #expansion 是类属性， 在其他地方要加上self.expansion 来调用
    expansion = 4
    # in_channel: 代表输入的卷积核的个数（256）最后残差结构是用256相加
    # out_channel： 代表第第二层输出的个数（64）
#     def __init__(self, in_channel, out_channel, stride = 1, downsample = None, groups=1, width_per_group=64):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels= in_channel, out_channels= out_channel, 
#                                kernel_size= 1, stride= 1, padding= 1, bias= False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
# #-----------------------------------------------------------------------------------
#         self.conv2 = nn.Conv2d(in_channels= out_channel, out_channels= out_channel,
#                                kernel_size= 3, stride= 1, padding= 1, bias= False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
# #-----------------------------------------------------------------------------------
#         self.conv3 = nn.Conv2d(in_channels= out_channel, out_channels= out_channel * self.expansion, 
#                                kernel_size= 1, stride= 1,padding= 1, bias= False)
#         self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
#         self.relu = nn.ReLU(inplace= True)
#         self.downsample = downsample

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
# block 使用的block种类   blocks_num ： 一个列表，使用的block的数目
    def __init__(self, block, blocks_num, num_classes = 1000, include_top = True, groups = 1, width_per_group = 64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64   # 第二行maxpooling得到的特征深度
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace= True)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            #自适应平均池化下采样 ，输出稳定
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#channel block 主分支上第一个的特征数
    def _make_layer(self, block, channel, block_num, stride = 1):
        # 第一层要有残差结构，单独拎出来
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                # 这一行漏了 bias  逆天
                nn.Conv2d(in_channels= self.in_channel, out_channels= channel * block.expansion,kernel_size= 1, stride= stride, bias= False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        # 后面没有残差结构，全是实线结构，直接循环扔进去
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 非关键字参数
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def Resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def Resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def Resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def Resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def Resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

if __name__ == '__main__':
    model = Resnet50(num_classes= 5)
    x = torch.randn(2, 3, 224, 224).cpu()
    print(model(x).shape)
    torchsummary.summary(model.cpu(), (3, 224, 224))
