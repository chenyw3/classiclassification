import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
from functools import reduce

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
    transform.Resize([224,224]),
    # transforms.RandomResizedCrop([224,224]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    transform.Resize([224,224]),
    transform.ToTensor()
])
train_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201013/train.txt', transform=transforms)
test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201013/test.txt', transform=transforms_test)
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 80
lr = 0.01
batch_size = 16


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
        )

        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
        )

        if in_features == out_features:  # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)

        return self.relu(out + self.shortcut(residual))


class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list=[3, 4, 6, 3], strides_list=[1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage_1 = self._make_layer(64, 128, 128, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(128, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, class_num)

    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers = [SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


def SKNet26(nums_class=1000):
    return SKNet(nums_class, [2, 2, 2, 2])


def SKNet50(nums_class=12):
    return SKNet(nums_class, [3, 4, 6, 3])


def SKNet101(nums_class=1000):
    return SKNet(nums_class, [3, 4, 23, 3])

if __name__ =='__main__':
    model = SKNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


    # 用于更新学习率
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # 训练
    total_step = len(train_loader)
    curr_lr = lr
    bets_acc = 0.85
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
        # if (epoch + 1) % 20 == 0:  # 每过20个Epoch，学习率就会下降
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)

        # 测试
        model.eval()
        with torch.no_grad():
            # correct = 0
            # total = 0
            # truth_class = 0
            num_correct = 0
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                num_correct += (predicted == labels).sum().item()
                # if (i+1) % 15 == 0:
                #     print('accuracy of the class {} is [{}/{}] :{}%'
                #           .format(truth_class, correct, total, 100 * correct / total))
                #     total = 0
                #     correct = 0
                #     truth_class += 1
                if num_correct / len(test_data) > bets_acc:
                    bets_acc = num_correct / len(test_data)
                    bets_epoch = epoch + 1
            print('acc:{:4f}'.format(num_correct / len(test_data)))
    print('best_acc:{}   best_epoch:{}'.format(bets_acc, bets_epoch))