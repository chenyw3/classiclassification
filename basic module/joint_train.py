import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.optim.lr_scheduler
import os
import copy
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)
# get list of models
# torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load pretrained models, using ResNeSt-50 as an example
net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True,)
net.avgpool = nn.Sequential(nn.Conv2d(2048,512,1),
                            GlobalAvgPool2d())
net.fc = nn.Linear(512,9)

net.conv1[0] = nn.Conv2d(1,32,3,2,1,bias=False)
def conv3x3(inplanes,outplanes,stride=1,padding=1):
    return nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=stride,padding=padding)


def conv1x1(inplanes,outplaens,stride=1):
    return nn.Conv2d(inplanes,outplaens,kernel_size=1,stride=stride,bias=False)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False,groups=32)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,outplanes,stride=1,downsample=None,norm_layer=None,padding=1,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes,outplanes)
        self.conv2 = conv3x3(outplanes,outplanes,stride,padding)
        self.conv3 = conv1x1(outplanes,outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,se_block,layers,num_classes=9,norm_layer=None ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(se_block, 64, layers[0])
        self.layer2 = self._make_layer(se_block, 128, layers[1], stride=2,)
        self.layer3 = self._make_layer(se_block, 256, layers[2], stride=2,)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                             norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
    best_acc = 0.9
    batch_size = 2
    transforms_test = transform.Compose([
        transform.Grayscale(),
        transform.Resize([320,320]),
        transform.ToTensor()
    ])
    test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/test.txt', transform=transforms_test)
    test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)
    print('num_of_testData:', len(test_data))
    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8')
    net1 = torch.load('C:/Users/cyw/Desktop/nets-73.pth').to(device)
    net2 = torch.load('C:/Users/cyw/Desktop/net-39.pth').to(device)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        # truth_class = 0
        num_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs1 = net1(images)
            outputs2 = net2(images)

            probability1 = nn.functional.softmax(outputs1,dim=1)
            probability2 = nn.functional.softmax(outputs2,dim=1)
            probability = probability1+probability2
            _, predicted = torch.max(probability.data, 1)
            num_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        print('acc:{:4f} {}/{}'.format(num_correct/len(test_data),num_correct,len(test_data)))
    for i in range(9):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))