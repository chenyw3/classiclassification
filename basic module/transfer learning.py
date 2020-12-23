import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torchvision.models import ResNet
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils
from PIL import Image


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


train_dataset = MyDataset(txt='C:/Users/cyw/Desktop/classification_20200916/train.txt', transform=transforms.Compose(
                                                     [
                                                         transforms.RandomResizedCrop(224),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         # transforms.Normalize(
                                                         #     mean=(0.485, 0.456, 0.406),
                                                         #     std=(0.229, 0.224, 0.225))
                                                     ]))

val_dataset = MyDataset(txt='C:/Users/cyw/Desktop/classification_20200916/train.txt', transform=transforms.Compose(
                                                     [
                                                         transforms.RandomResizedCrop(224),
                                                         # transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         # transforms.Normalize(
                                                         #     mean=(0.485, 0.456, 0.406),
                                                         #     std=(0.229, 0.224, 0.225))
                                                     ]))

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)

# 类别名称
# class_names = train_dataset.classes
# print('class_names:{}'.format(class_names))


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# 训练设备  CPU/GPU


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('trian_device:{}'.format(device.type))

# 加载预训练模型
model = models.resnet50(pretrained=True)
print(model)
# 固定模型权重
# for param in model.parameters():
#     param.requires_grad = False
# 最后加一个分类器
model.fc = nn.Linear(2048, 8)
for param in model.fc.parameters():
    param.requires_grad = True
# 检查GPU是否可用
if torch.cuda.is_available():
    model = model.cuda()
# batch大小
batch_size = 128
epoches = 80
lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
total_step = len(train_loader)
best_epoch = 0
best_acc = 0.85
curr_lr = lr
for epoch in range(epoches):
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

        if (i + 1) % 17 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f}"
                  .format(epoch + 1, epoches, i + 1, total_step, loss.item()))
    # if (epoch + 1) % 20 == 0:  # 每过20个Epoch，学习率就会下降
    #     curr_lr /= 3
    #     update_lr(optimizer, curr_lr)

# 测试
    model.eval()
    with torch.no_grad():
        num_correct = 0
        for i,(images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == labels).sum().item()
            if num_correct/len(val_dataset)>best_acc:
                best_acc = num_correct/len(val_dataset)
                best_epoch = epoch+1
        print('acc:{:4f}'.format(num_correct/len(val_dataset)))