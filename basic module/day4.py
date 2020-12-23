import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

# 判断是否启用GPU计算
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCHS = 1  # 批处理的次数
batch_size = 100  # 批处理时，每次提取的数据个数
LR = 0.001  # 学习效率

# 图像预处理
transform = transforms.Compose([
    transforms.Pad(4),  # 图片的边界以0填充
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transforms.RandomCrop(32),  # 给定大小的图片的随机剪辑
    transforms.ToTensor(),  # 转化数据类型为tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 训练集、测试集加载
train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True )
test_data = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 残差模块的定义
class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, down_sample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel,
                               out_channels=o_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        # BatchNorm2d(）对小批量3d数据组成的4d输入进行批标准化操作
        # 主要为了防止神经网络退化
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=o_channel,
                               out_channels=o_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 将单元的输入直接与单元输出加在一起
        if self.down_sample:
            residual = self.down_sample(x)  # 下采样
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.layer4 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        # blocks=layers,残差模块的数量
        down_sample = None
        # 判断是否in_channels(输入)与(输出)是否在同一维度
        # 即输入的3d数据的长宽高与输出的数据的长宽高是否一样
        if (stride != 1) or (self.in_channels != out_channels):
            # 如果不一样就转换一下维度
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)  # 添加所有残差块

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


model = ResNet(Residual_Block, [2, 2, 2, 2]).to(device)

# 计算误差与
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练
total_step = len(train_loader)
curr_lr = LR
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        total=0
        correct=0
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(total)
        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f} Acc{:.4f}"
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item(),correct/total))

    # 降低学习速度
    if (epoch + 1) % 10 == 0:  # 每过20个Epoch，学习率就会下降
        curr_lr *= 0.1
        update_lr(optimizer, curr_lr)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('accuracy of the model on the test images: {}%'
          .format(100 * correct / total))
# 测试

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse','8','9')
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))