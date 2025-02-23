import os.path
import time  # 导入 time 模块

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from mistune.markdown import preprocess

from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.models import alexnet,AlexNet_Weights

img=read_image('./cats_and_dogs_filtered/validation/cats/cat.2000.jpg')

# 模型权重文件
weights=AlexNet_Weights.DEFAULT

#让模型加载参数
model=alexnet(weights=weights)

#看到模型结构
tmp=model.eval()
print(tmp)

# 测试模型能否运行
# 网络模型训练的时候使用了哪些数据转换，那测试的时候图片应经过一样的数据转换，比如归一化
preprocess=weights.transforms()# 拿到对应模型的数据转换的函数逻辑
# 在张量的第 0 维添加一个新的维度。
#如果输入是一张图像，通常是 [channels, height, width]，通过 unsqueeze(0) 变为 [1, channels, height, width]，表示一个批次中只有 1 张图片。
batch = preprocess(img).unsqueeze(0)
# squeeze(0):移除第 0 维的大小为 1 的维度。
#输入批次的形状是 [1, 1000]（一个批次中有 1 张图片，模型输出的结果是 1000 个类别的分数），通过 squeeze(0) 变为 [1000]，表示单张图片的预测结果。
prediction = model(batch).squeeze(0).softmax(0) #dim=0 表示在张量的第 0 维 上计算 softmax。
# 现在prediction里面就是一张图像对应的1000分类的概率分布了
print(prediction)
print(len(prediction))

class_id = prediction.argmax().item()
score = prediction[class_id].item()

print(class_id, score)

# meta是元数据 里面包括了模型模型文件的一些信息
print(weights.meta)
category_name=weights.meta['categories'][class_id]
print(f"{category_name}:{100*score:.1f}%")

class_to_idx={cls:idx for idx,cls in enumerate(weights.meta['categories'])}
print(class_to_idx)

#检测CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#把一些层里面的参数固定住
for k,v in model.named_parameters():
    if not k.startswith('classifier'):
        #除了classifier分类器以外的层都不去调参
        v.requires_grad = False  # v是参数  k是features.0.weight 字符串

# 替换预训练模型中的靠后的一些全连接层和输出层
# 虽然输入和输出节点个数一致，但是因为重新创建了layer层，所以参数会被随机初始化
model.classifier[1]=nn.Linear(9216,4096)            #替换classifier中的第一层
model.classifier[4]=nn.Linear(4096,4096)
model.classifier[6]=nn.Linear(4096,2)  # 二分类

model.to(device)

transforms_for_train=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
])

transforms_for_validation=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
basedir="./cats_and_dogs_filtered/"
train_dir=os.path.join(basedir,"train")
validation_dir=os.path.join(basedir,"validation")
train_datasets=datasets.ImageFolder(train_dir,transform=transforms_for_train)
test_datasets=datasets.ImageFolder(validation_dir,transform=transforms_for_validation)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_datasets, batch_size=20, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 20

for epoch in range(EPOCHS):
    print('EPOCH', epoch)
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        input,targets=inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs=outputs.argmax(1,keepdim=True)
        running_correct+=outputs.eq(targets.view_as(outputs)).sum().item()
    epoch_loss = running_loss*20/ len(train_datasets)
    epoch_acc = running_correct*100/len(train_datasets)
    print("Training loss:{:.4f} Acc:{: .2f} %".format(epoch_loss, epoch_acc))
    end_time=time.time()  #计算一轮所需要的时间
    print("Epoch time:", end_time-start_time)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            x,y=data.to(device),target.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            test_loss += loss.item()
            pred = output.argmax(1,keepdim=True)
            correct += pred.eq(y.view_as(outputs)).sum().item()
        epoch_loss = test_loss*20/ len(test_datasets)
        epoch_acc = correct*100/len(test_datasets)
        print("Test Loss:{:.4f} Acc:{: .2f} %".format(epoch_loss, epoch_acc))

