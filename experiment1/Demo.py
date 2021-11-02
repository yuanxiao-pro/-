#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Demo.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 18:56   xiaoj      1.0         None
'''

# import lib
# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple [pkgname]
import pickle
import torch
from torch import nn
import torch.nn.functional as fun
import torch.optim as optim
from skimage import io, transform
import pylab #显示数据图像
import torch.utils.data as data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 500
EPOCHS = 20
# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cov1 = nn.Conv2d(1, 10, 5)  # 结果是24*24 通道是10
        self.cov2 = nn.Conv2d(10, 20, 5)  # 结果是8*8，通道是20
        self.lt1 = nn.Linear(8 * 8 * 20, 500)
        self.lt2 = nn.Linear(500, 10)  # 最后得到10维向量

    def forward(self, x):  # x 的维度是（500*1*28*28）
        out = fun.relu(self.cov1(x))  # 卷积激活得到24*24*10
        '''nn.Conv2d接受一个4维的张量，每一维分别是sSamples * nChannels * Height * Width（样本数*通道数*高*宽）。
        如果你有单个样本，只需使用 input.unsqueeze(0) 来添加其它的维数'''
        # input,f,s
        out = fun.max_pool2d(out, 2, 2)  # 池化得到12*12*10
        out = fun.relu(self.cov2(out))  # 卷积激活得到8*8*20
        out = out.view(BATCH_SIZE, -1)  # 后面接全连接层,展开成向量
        out = fun.relu(self.lt1(out))
        out = self.lt2(out)
        out = fun.log_softmax(out, 1)  # 再用softmax计算概率分类
        return out


# 读取数据
with open('mnist.pkl', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)

class TrainDataset(data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_train = x
        self.y_train = y
        self.transform = transform

    # 使得len(dataset) 返回dataset大小
    def __len__(self):
        return self.x_train.shape[0]

    # dataset每个单元由一个图像及其对应的标志点组成，可由dataset[idx]访问对应的样本
    def __getitem__(self, idx):
        sample = {'image': self.x_train[idx], 'target': self.y_train[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ValidDataset(data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_valid = x
        self.y_valid = y
        self.transform = transform

    # 使得len(dataset) 返回dataset大小
    def __len__(self):
        return self.x_valid.shape[0]

    # dataset每个单元由一个图像及其对应的标志点组成，可由dataset[idx]访问对应的样本
    def __getitem__(self, idx):
        sample = {'image': self.x_valid[idx], 'target': self.y_valid[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 模式
    for batch_idx, batched_sample in enumerate(train_loader):
        input, target = batched_sample['image'].to(DEVICE), batched_sample['target'].to(
            DEVICE)  # 转变成tensor
        # 梯度置零
        optimizer.zero_grad()
        # 正向传递
        output = model(input.view(BATCH_SIZE, 1, 28, 28))
        # 损失函数
        loss = fun.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 记录checkpoint
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in valid_loader:
            input, target = sample['image'].to(DEVICE), sample['target'].to(DEVICE)  # 转变成tensor
            output = model(input.view(BATCH_SIZE, 1, 28, 28))
            valid_loss += fun.nll_loss(output, target, reduction='sum').item()  # 损失项和
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    valid_loss /= 10000
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, 10000,
        100. * correct / 10000))


train_data = TrainDataset(x_train, y_train, None)
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_data = ValidDataset(x_valid, y_valid, None)
valid_loader = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

# 实例化模型
model = Net().to(DEVICE)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    valid(model, DEVICE, valid_loader)

