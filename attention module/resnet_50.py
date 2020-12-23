from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transform
from PIL import Image
import torch.nn.functional as F
import math


def conv3x3(inplanes,outplanes,stride=1,padding=1):
    return nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=stride,padding=padding)


def conv1x1(inplanes,outplaens,stride=1):
    return nn.Conv2d(inplanes,outplaens,kernel_size=1,stride=stride,bias=False)


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
    #transform.Grayscale(),
    transform.Resize([320,320]),
    # transforms.RandomResizedCrop([224,224]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    # transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    #transform.Grayscale(),
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 20
lr = 0.00025
batch_size = 16

# transforms = transform.Compose([
#     transform.Pad(4),
#     transform.RandomHorizontalFlip(),
#     transform.RandomResizedCrop(32),
#     transform.ToTensor(),
#     transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
# ])
#
# train_data = torchvision.datasets.CIFAR10(
#     root='./data',
#     transform=transforms,
#     download=False,
#     train=True,
# )
# test_data = torchvision.datasets.CIFAR10(
#     root='./data',
#     transform=transforms,
#     download=False,
#     train=False
# )
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        # nn.init.constant(m[-1].weight,val=0)
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            # 汇集全文的信息 对应的像素点进行匹配，整个图像的像素点全部相加
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class GCBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(GCBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.gc = ContextBlock2d(planes*4,planes,"att",["channel_add"])
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
        out = self.gc(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


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
        self.gct1 = GCT(inplanes)
        self.conv1 = conv1x1(inplanes,outplanes)
        self.gct2 = GCT(outplanes)
        self.conv2 = conv3x3(outplanes,outplanes,stride,padding)
        self.gct3 = GCT(outplanes)
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
    def __init__(self,block,gc_block,layers,num_classes=9,norm_layer=None ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(gc_block, 64, layers[0])
        self.layer2 = self._make_layer(gc_block, 128, layers[1], stride=2,)
        self.layer3 = self._make_layer(gc_block, 256, layers[2], stride=2,)
        self.layer4 = self._make_layer(gc_block, 512, layers[3], stride=2,)
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


model = ResNet(Bottleneck,GCBottleneck, [3, 4, 6, 3]).to(device)
# 计算误差与
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.99,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练
total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.90
bets_epoch = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1020 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f}"
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
    print('epoch{} loss{}'.format(epoch + 1, loss.item()))

    # 降低学习速度
    if (epoch + 1) % 40 == 0:  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# 测试
    model.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        # truth_class = 0
        num_correct = 0
        for i,(images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            num_correct += (predicted == labels).sum().item()
            if num_correct/len(test_data) > bets_acc:
                bets_acc = num_correct/len(test_data)
                bets_epoch = epoch+1
             #   torch.save(model,'nets-{}'.format(epoch))
        print('test_acc:{:4f}'.format(num_correct/len(test_data)))
print('best_test_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))