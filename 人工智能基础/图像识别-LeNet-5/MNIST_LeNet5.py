import numpy as np
import torch
import torch.nn as nn
from jinja2.compiler import F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 三层卷积+两层全连接
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=F.sigmoid(x)
        x=F.max_pool2d(x, 2, 2)
        x=self.conv2(x)
        x=F.sigmoid(x)
        x=F.max_pool2d(x, 2, 2)
        x=self.conv3(x)
        # 展平为二维数组,    行是x.shape[0]就是样本行数，列是自动生成，就是剩下的都展平
        x=x.view(x.shape[0], -1)
        x=self.fc1(x)
        x=F.sigmoid(x)
        x=self.fc2(x)
        return x


net_module=LeNet()

# 两层卷积+三层全连接
net_sequential=nn.Sequential(
    nn.Conv2d(1, 6, 5, 1),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(6, 16, 5, 1),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(256, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
# 在pytorch里面默认的数据的格式是 (batch size, channel, h, w)  dtype=torch.float32：指定张量的数据类型为 32 位浮点数。
X=torch.rand(size=(1,1, 28, 28),dtype=torch.float32)
for layer in net_sequential.children():
    X=layer(X)
    #__class__ 是一个特殊的属性，它指向创建该对象的类
    # __name__ 是类的名称属性，它会返回类的名称
    print(layer.__class__.__name__, 'output shape: \t',X.shape)

transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset=datasets.MNIST(root='./mnist', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./mnist', train=False, transform=transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer=optim.Adam(net_sequential.parameters(), lr=0.001)


if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')


def train(epoch_id):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs,targets = data.to(device),target.to(device)
        #把梯度置为0
        optimizer.zero_grad()
        outputs = net_sequential(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('[%d, %5d] loss: %.3f' %(epoch_id, batch_idx, running_loss))
            running_loss=0.0


def test(epoch_id):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            inputs,targets = data.to(device),target.to(device)
            outputs = net_sequential(inputs)
            loss = loss_fn(outputs, targets)
            #dim=1 每行找到最大那一列  返回沿着指定的维度 dim=1 找到最大值及其对应的索引位置
            _,predicted=torch.max(outputs.data, 1)
            total += target.size(0)
            correct+=(predicted==targets).sum().item()
        print("Accuracy of the network on the  test images: %d %%" % (100 * correct / total))

if __name__=='__main__':
    for id in range(20):
        train(epoch_id=id)
        if id%2==0:
            test(epoch_id=id)


