import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from radar_classfication import VGG16


# 模型存储路径
# model_save_path = 'C:/Users/cyw/Desktop/cnn.pth'
# 数据加载

test_data = open('C:/Users/cyw/Desktop/classification/test.txt', 'r')
transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transforms.ToTensor()  # 转化数据类型为tensor
])
imgs = []
for line in test_data:         # 读取测试集并转为列表
    line = line.strip('\n')
    line = line.rstrip('\n')
    words = line.split()
    imgs.append((words[0], int(words[1])))

# class_names = ['0', '1']  # 这个顺序很重要，要和训练时候的类名顺序一致
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 载入模型并且训练
model = VGG16()
model.load_state_dict(torch.load('cnn.pth'))
model.to(device)
model.eval()
img_test_result = open('C:/Users/cyw/Desktop/img_test_result.txt', 'a')

for fn,label in imgs: # 测试模型
    image_PIL = Image.open(fn)
    # 根据路径读取图像文件
    image_tensor = transforms(image_PIL)
    # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor.unsqueeze_(0)
    # 没有这句话会报错
    image_tensor = image_tensor.to(device)
    out = model(image_tensor)
    _, pred = torch.max(out, 1)
    name = fn + ' ' + str(int(pred)) + '\n'
    img_test_result.write(name)  # 写为txt文件
img_test_result.close()
print('finish')





