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
net = net.to(device)
print(net)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


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
    net.eval()
    with torch.no_grad():
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