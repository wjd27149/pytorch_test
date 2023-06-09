import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weights = False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 1*1*4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p=0.5),#正则化技术，减少神经网络过拟合的问题,一般在relu之后
            nn.Linear(1*1*4096, 1*1*4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p=0.5),
            nn.Linear(1*1*4096, num_classes),# 最后一个全连接输出
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # [N * 3 * 224 * 224]
        x = self.features(x)
        # [N * 512 * 7 * 7]
        x = torch.flatten(x, start_dim= 1)
        x = self.classifier(x)
        return x

def make_layer(cfg: list):
    layers = []
    in_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size= 2, stride= 2)]
        else:
            conv2d = nn.Conv2d(in_channel, v, kernel_size= 3, padding= 1)
            layers += [conv2d, nn.ReLU(inplace= True)]
            in_channel = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name = "vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_layer(cfg))
    return model