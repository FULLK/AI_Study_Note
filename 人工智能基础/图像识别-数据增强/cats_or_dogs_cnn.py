import os.path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets, transforms

#检测cuda是否可用
use_cuda=torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,1)
        self.conv2 = nn.Conv2d(16, 32, 3,1)
        self.conv3 = nn.Conv2d(32, 64, 3,1)
        # linear的输入特征数值不知道是多少，等报错再修改
        self.fc1 = nn.Linear(18496, 512)
        self.fc2 = nn.Linear(512, 2)
    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x, 2)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x, 2)
        x=self.conv3(x)
        x=F.relu(x)
        x=F.max_pool2d(x, 2)
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) # 结果的10 个值是每个类别的“得分”（logits），未经过归一化。
        #这些得分还未归一化为概率分布，通常需要通过 softmax 函数将其转换为概率

        #在训练中，x 会直接传给损失函数（如 CrossEntropyLoss）。
        #损失函数会对 logits 执行 softmax 和标签对比，所以我们不需要手动归一化。
        return x

model = Net().to(device)
#PyTorch 的 torchvision.transforms 模块对图像进行预处理和数据增强。
#针对 训练集 的数据增强和预处理。
transform_for_train = transforms.Compose([
    transforms.Resize((150, 150)),  #改变图片大小
    # 下面是数据增强
    #将图片的尺寸调整为 150×150。
    #适用于所有图片数据，确保输入网络时尺寸统一。
    transforms.RandomRotation(40),   #随机旋转角度
    # 随机旋转图片，角度范围为 [-40°, 40°]。
    # 增强模型对旋转的鲁棒性（例如手写数字倾斜、物体方向变化等）。
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),   # 随机垂直翻转
    transforms.RandomCrop(150),   # 随机裁剪
    #从图片中随机裁剪一个 150×150 的区域。如果图片尺寸本来就是 150×150，那么 transforms.RandomCrop(150) 的操作不会改变图片内容。
    transforms.RandomResizedCrop(150),  #随机裁剪区域并缩放到指定大小（150×150）。
    transforms.ColorJitter(),  # 调整亮度 对比度 饱和度 色调
    #随机调整图像的亮度、对比度、饱和度和色调，模拟光照变化。
    transforms.RandomAffine(
        degrees=30,  # 旋转范围 ±30°
        translate=(0.1, 0.1),  # 平移范围 10% 宽高
        scale=(0.8, 1.2),  # 缩放范围 80%-120%
        shear=20  # 剪切角度范围 ±20°
        #剪切（Shear）不会改变图片的大小（宽和高保持不变），但图片的内容会发生倾斜变形。
    ),
    #仿射变换，包含旋转、平移、缩放、剪切，用于增强图像的变化多样性。
    transforms.ToTensor(),  # 将图片数据的每个数值缩放到0到1之间，图片数值矩阵和图片类型都变成张量
])
#针对 测试集 的简单预处理。
transform_for_test=transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

base_dir='./cats_and_dogs_filtered'
train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir, 'validation')
#加载所有数据为一个数据集
train_datasets=datasets.ImageFolder(train_dir, transform=transform_for_train)
test_datasets=datasets.ImageFolder(validation_dir, transform=transform_for_test)

# classes:类别名称列表，按子目录名的字母顺序排序。
# class_to_idx:类别名称到索引的映射字典。
# imgs:一个列表，包含每张图像的路径和对应类别索引
example_classes=train_datasets.classes
print(example_classes)
index_classes=train_datasets.class_to_idx
print(index_classes)


# 设置数据集的加载器，以某种形式取数据 num_workers用来设置多个子进程数同时加载数据,此时会fork子进程，所以此时需要设置主函数
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False,num_workers=4)

# Softmax 处理+结果取对数计算交叉
loss_fn = nn.CrossEntropyLoss()
#一种调整参数的方式
optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.001)
# 总结一下，在PyTorch中进行二分类，有三种主要的全连接层，激活函数和loss function组合的方法，分别是：
    # torch.nn.Linear+torch.sigmoid+torch.nn.BCELoss，
    # torch.nn.Linear+BCEWithLogitsLoss，和
    # torch.nn.Linear（输出维度为2）+torch.nn.CrossEntropyLoss，
    # 后两个loss function分别集成了Sigmoid和Softmax。

if __name__ == '__main__':
    # 训练20轮，每轮训练所有批次样本并评估
    EPOCHS = 20
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # 获取一个批次的数据图片X和标签y
            x,y=data.to(device),targets.to(device)
            # 开始每一次正向反向传播之前，把optimizer重置一下
            optimizer.zero_grad()
            # 正向传播从 X->predictions
            y_pred=model(x)
            # 从 predictions->Loss值, loss计算的是一个批次所有样本的平均损失   reduction: str = "mean",
            loss = loss_fn(y_pred, y)
            ## 反向传播,求导
            loss.backward()
            # 应用求出来的梯度更新参数
            optimizer.step()
            running_loss += loss.item()
            pre=y_pred.argmax(1,keepdim=True)
            running_corrects+=pre.eq(targets.view_as(pre)).sum().item()
        #打印每一轮训练的平均损失值和准确率
        epoch_loss = running_loss*64/len(train_datasets)
        epoch_acc=running_corrects*100/len(train_datasets)
        print('train Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch_loss, epoch_acc))

        model.eval()
        test_loss = 0
        test_corrects = 0
        with torch.no_grad():
            for data, target in test_loader:
                x,y=data.to(device),target.to(device)
                y_pred=model(x)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                pre=y_pred.argmax(1,keepdim=True)
                test_corrects+=pre.eq(target.view_as(pre)).sum().item()
        # 打印每一轮训练后测试的平均损失值和准确率
        epoch_loss = test_loss*64/len(test_datasets)
        epoch_acc=test_corrects*100/len(test_datasets)
        print('test Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch_loss, epoch_acc))
