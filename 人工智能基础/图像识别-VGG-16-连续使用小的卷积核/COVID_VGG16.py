import os
import time

import torch
import torchvision
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from  torchvision import datasets, transforms
# 检测CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建预训练模型
model=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
print(model)

#将所有参数都先固定
for param in model.parameters():
    print(param)
    param.requires_grad=False

# 做二分类任务，替换分类器部分，这里的参数不会固定
model.classifier = nn.Sequential(
    #  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))知道输出size，然后根据最后一层卷积知道通道是512的
    nn.Linear(512*7*7, 256), # 256就是当前这一层全连接对应的神经元的个数
    nn.ReLU(inplace=True), #inplace=True 允许在原地（in-place）进行操作。这意味着输入的张量会直接在原地被修改，而不需要额外分配内存来存储结果。
    nn.Linear(256, 2),  #这里2是二分类
)
print(model)

#需要训练的参数放到列表里去
params_to_update=[]
for name, param in model.named_parameters():
    #param可以看到有requires_grad=True这个选项，代表会去调参
    #print(name, param)  #各个部分的每个层的参数  但非线性和池化没有，只包括有参数的层会打印
    if param.requires_grad==True:
        params_to_update.append(param)
        print('\t', name)


model=model.to(device)

#数据转换的函数，可以数据增强
data_transform={
    'data':transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test':transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ])
}

# 读取数据
basedir="COVID_VGG16"
train_dir=os.path.join(basedir,"train")
val_dir=os.path.join(basedir,"test")

train_dataset = datasets.ImageFolder(train_dir, data_transform['data'])
val_dataset = datasets.ImageFolder(val_dir, data_transform['test'])

#类别和类别对应的索引
example_classes=train_dataset.classes
index_classes=train_dataset.class_to_idx
print(example_classes)
print(index_classes)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 20
for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch+1, EPOCHS))
    model.train()

    running_loss = 0.0
    running_corrects = 0.0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        X,Y=inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        y_pred=model(X)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        optimizer.step()
        # 统计
        running_loss += loss.item()
        preds=y_pred.argmax(1,keepdim=True)
        running_corrects += preds.eq(Y.view_as(preds)).sum()

    epoch_loss = running_loss * 10/ len(train_dataset)
    epoch_acc = running_corrects *100/ len(train_dataset)
    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    end_time=time.time()
    print('train Time taken: {:.4f} seconds'.format(end_time-start_time))
    #测试
    model.eval()
    test_loss = 0.0
    test_corrects = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            X,Y=inputs.to(device),targets.to(device)
            y_pred=model(X)
            loss = loss_fn(y_pred, Y)
            preds=y_pred.argmax(1,keepdim=True)
            test_corrects += preds.eq(Y.view_as(preds)).sum()
            test_loss += loss.item()
        epoch_loss = test_loss * 10/ len(val_dataset)
        epoch_acc = test_corrects *100/ len(val_dataset)
        print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))









