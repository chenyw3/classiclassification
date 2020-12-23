import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.optim.lr_scheduler
import os
import copy


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


# class Conv2d(nn.Conv2d):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)
#         # for convolutional layers with a kernel size of 1, just use traditional convolution
#
#         if kernel_size == 1 or True:
#             self.ind = True
#         else:
#             self.ind = False
#             self.oc = out_channels
#             self.ks = kernel_size
#
#             # the target spatial size of the pooling layer
#             ws = kernel_size
#             self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))
#
#             # the dimension of the latent repsentation
#             self.num_lat = int((kernel_size * kernel_size) / 2 + 1)
#
#             # the context encoding module
#             self.ce = nn.Linear(ws * ws, num_lat, False)
#             self.ce_bn = nn.BatchNorm1d(in_channels)
#             self.ci_bn2 = nn.BatchNorm1d(in_channels)
#
#             # activation function is relu
#             self.act = nn.ReLU(inplace=True)
#
#             # the number of groups in the channel interacting module
#             if in_channels // 16:
#                 self.g = 16
#             else:
#                 self.g = in_channels
#             # the channel interacting module
#             self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
#             self.ci_bn = nn.BatchNorm1d(out_channels)
#
#             # the gate decoding module
#             self.gd = nn.Linear(num_lat, kernel_size * kernel_size, False)
#             self.gd2 = nn.Linear(num_lat, kernel_size * kernel_size, False)
#
#             # used to prrepare the input feature map to patches
#             self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
#
#             # sigmoid function
#             self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         # for convolutional layers with a kernel size of 1, just use traditional convolution
#
#         if self.ind:
#             return F.conv2d(x, self.weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#         else:
#             b, c, h, w = x.size()
#             weight = self.weight
#             # allocate glbal information
#             gl = self.avg_pool(x).view(b, c, -1)
#             # context-encoding module
#             out = self.ce(gl)
#             # use different bn for the following two branches
#             ce2 = out
#             out = self.ce_bn(out)
#             out = self.act(out)
#             # gate decoding branch 1
#             out = self.gd(out)
#             # channel interacting module
#             if self.g > 3:
#                 # grouped linear
#                 oc = self.ci(self.act(self.ci_bn2(ce2). \
#                                       view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2, 3).contiguous()
#             else:
#                 # linear layer for resnet.conv1
#                 oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()
#             oc = oc.view(b, self.oc, -1)
#             oc = self.ci_bn(oc)
#             oc = self.act(oc)
#             # gate decoding branch 2
#             oc = self.gd2(oc)
#             # produce gate
#             out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
#             # unfolding input feature map to patches
#             x_un = self.unfold(x)
#             b, _, l = x_un.size()
#             # gating
#             out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
#             # currently only handle square input and output
#
#             return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

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

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}


def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:

        pretrained_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True)
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

model = resnest50(pretrained=True)
model.avgpool = nn.Sequential(nn.Conv2d(2048,512,1),
                            GlobalAvgPool2d())
model.fc = nn.Linear(512,9)

model.conv1[0] = nn.Conv2d(1,32,3,2,1,bias=False)
# model = ResNet(Bottleneck, [3, 4, 6, 3],
#                    radix=2, groups=1, bottleneck_width=64,
#                    deep_stem=True, stem_width=32, avg_down=True,
#                    avd=True, avd_first=False,pretrain=True)
model = model.to(device)
print(model)
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
            # 数据标签转换为Tensor
        return img, label

    def __len__(self):
        return len(self.imgs)


transforms = transform.Compose([
    transform.Grayscale(),
    transform.Resize([320,320]),
    # transforms.RandomResizedCrop([224,224]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    transform.Grayscale(),
    transform.Resize([320,320]),
    transform.ToTensor()
])
train_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/train.txt', transform=transforms)
test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/test.txt', transform=transforms_test)
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))
epochs = 120
lr = 0.00025
batch_size = 2

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.99,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练
total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.85
bets_epoch = 0


milestones = [40,60,80,100]
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8')

for epoch in range(epochs):
    total = 0
    num_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        model.train()

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        num_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 90 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f} Acc:{:.4f} {}/{}"
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(),num_correct/len(train_data),num_correct,len(train_data)))
    print('train-acc:{} correct{} total{}'.format(num_correct/total,num_correct,total))

    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1)

    # 降低学习速度
    if (epoch + 1) % 40 == 0 :  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
# 测试
    model.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        # truth_class = 0
        num_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            num_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        if num_correct / len(test_data) > bets_acc:
            bets_acc = num_correct / len(test_data)
            bets_epoch = epoch + 1
            classc = copy.deepcopy(class_correct)
            classt = copy.deepcopy(class_total)
        print('acc:{:4f}'.format(num_correct/len(test_data)))
print('best_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))
for i in range(9):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * classc[i] / classt[i]))
