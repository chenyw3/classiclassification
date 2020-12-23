import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
        return x_out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_triplet_attention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_triplet_attention:
            self.triplet_attention = TripletAttention(planes, 16)
        else:
            self.triplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.triplet_attention is None:
            out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_triplet_attention=False):
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

        if use_triplet_attention:
            self.triplet_attention = TripletAttention(planes*4, 16)
        else:
            self.triplet_attention = None

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

        if not self.triplet_attention is None:
            out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_triplet_attention=att_type=='TripletAttention'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_triplet_attention=att_type=='TripletAttention'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


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
net = ResidualNet(network_type='ImageNet',depth=50,num_classes=9,att_type='TripletAttention')
net.to(device)
print(net)

transforms = transform.Compose([
    # transform.Grayscale(),
    transform.Resize([320,320]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    # transform.Grayscale(),
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
optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.95,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.0
bets_epoch = 0
test_acc = []
train_acc = []
name=[]
milestones = [40,60,80,100]
wrong = open('C:/Users/cyw/Desktop/wrong2.txt', 'r+')
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8')
# 训练
for epoch in range(epochs):
    total = 0
    num_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        net.train()

        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
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
    train_acc.append(num_correct/len(train_data))
    print('epoch:{} train-acc:{} correct{} total{}'.format(epoch+1,num_correct/total,num_correct,total))

    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1)

    # 降低学习速度
    if (epoch + 1) % 20 == 0 :  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
# 测试
    name.clear()

    with torch.no_grad():
        net.eval()
        num_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if predicted[i]!=labels[i]:
                    name.append((str(int(predicted[i])),str(int(labels[i]))))

        if num_correct / len(test_data) > bets_acc:
            wrong.seek(0)
            wrong.truncate(0)
            bets_acc = num_correct / len(test_data)
            bets_epoch = epoch + 1
            classc = copy.deepcopy(class_correct)
            classt = copy.deepcopy(class_total)
            # 计入错判样本
            for x1,x2 in name:
                x = str(x1)+str(x2)+'\n'
                wrong.write(x)
           # torch.save(net,'net-{}'.format(epoch))
    print('acc:{:4f} {}/{}'.format(num_correct/len(test_data),num_correct,len(test_data)))
    test_acc.append(num_correct/len(test_data))
print('best_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))
# 输出各类准确度
try:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * classc[i] / classt[i],classc[i], classt[i]))
except NameError:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * class_correct[i] / class_total[i],class_correct[i], class_total[i]))
wrong.close()
# 准确度可视化
x1 = range(1, epochs+1)
x2 = range(1, epochs+1)
y1 = test_acc
y2 = train_acc
plt.plot(x1, y1, 'o-',label='test_acc')
plt.plot(x2, y2, '.-',label='train_acc')
plt.legend(loc = 'upper right')
plt.xlabel('acc')
plt.ylabel('epoch')
plt.xticks(range(epochs))
plt.show()
