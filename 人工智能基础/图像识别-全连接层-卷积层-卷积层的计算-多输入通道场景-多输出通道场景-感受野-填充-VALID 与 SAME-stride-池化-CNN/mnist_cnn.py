import torch
#导入 PyTorch 的神经网络模块 torch.nn
import torch.nn as nn
#导入 PyTorch 的神经网络功能模块 torch.nn.functional
import torch.nn.functional as F
#导入 PyTorch 的优化器模块 torch.optim
import torch.optim as optim

#torchvision.datasets提供常用的 图像数据集加载器
#torchvision.transforms 提供 数据预处理工具
from torchvision import datasets, transforms

#查看torch版本
print(torch.__version__)

#检测CUDA是否可用
use_cuda = torch.cuda.is_available()
print(use_cuda)

#设置device变量
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#设置对数据进行连续处理的逻辑
transform = transforms.Compose([
    #让数据转成tensor张量
    transforms.ToTensor(),
    #标准归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

#读取数据
dataset1 = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataset2 = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#设置数据加载器，顺带设置批次大小和是否打乱数据
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset1, batch_size=1000)
#DataLoader 类是 PyTorch 中用于将数据集和采样器（Sampler）结合，并提供一个可迭代的数据加载器。它支持对数据集进行批量加载，并可选择进行多进程加载、自动批处理（collation）、内存固定等操作。
#求解数据集的均值和方差，方便对数据进行标准归一化
total_sum = 0.0        # 累计所有数据的总和
total_square_sum = 0.0 # 累计所有数据的平方和
total_count = 0        # 累计所有数据的总数
for batch_idx, data  in enumerate(train_loader,start=0):
    inputs, labels = data
    print(f"Batch {batch_idx}, Data Shape: {inputs.shape}")
    #第一维：-1 表示自动推断维度，通常是批量大小。
    #第二维：28*28 表示将每张图片展平为一个 784 维的向量（适用于 28x28 的图像数据）。
    x=inputs.view(-1,28*28)
    print(f"Batch {batch_idx}, Data Shape: {x.shape}")
    #.item()：将结果从 PyTorch 张量转换为 Python 的浮点数标量
    # 只能用于包含单一元素的张量。如果张量包含多个元素，使用 .item() 会抛出错误。

    # 累计数据总和和平方和
    total_sum += x.sum()
    total_square_sum += (x ** 2).sum()
    total_count += x.numel()  # numel 返回张量的元素个数

# 计算全体数据的平均值和标准差
x_mean = total_sum / total_count
x_std = (total_square_sum / total_count - x_mean**2)**0.5

print(f"Overall Mean: {x_mean.item()}")
print(f"Overall Std: {x_std.item()}")

 #通过自定义类来构建模型
# 这是一个继承自 torch.nn.Module 的自定义类，用于定义一个神经网络模型。
# nn.Module 是 PyTorch 中所有神经网络的基类，自定义模型必须继承它。
class Net(nn.Module):
    def __init__(self):
        #调用父类 nn.Module 的初始化方法,确保父类的所有功能得以继承。
        #super(Net, self)：
        # 获取 Net 类的父类，即 nn.Module。
        # super() 函数返回的是父类对象，使我们可以调用父类的方法
        super(Net, self).__init__()
        #添加一个 Dropout 层，用于在训练过程中随机丢弃 20% 的神经元，防止过拟合。
        self.dropout = nn.Dropout(0.2)
        #定义一个全连接层（fully connected layer），输入维度是 1600，输出维度是 128。
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

        self.conv1 = nn.Conv2d(1, 64, 3,1)
        self.conv2 = nn.Conv2d(64, 64, 3,1)

    def forward(self, x):#每次一个批次前向传播 #128*1*28*28
        #batchsize 运算机制https: // blog.csdn.net / weixin_44986037 / article / details / 144216069
        x=self.conv1(x)  #128*64*26*26
        x=F.relu(x)
        x=F.max_pool2d(x,2,2) # 128*64*13*13
        x=self.conv2(x) #128*64*11*11
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)# 128*64*5*5 11/2向下取整
        #将输入张量展平（flatten），从形状 [batch_size, channels, height, width] 转换为 [batch_size, features]。
        #参数 1 表示从第 1 维开始展平，保留第 0 维（批次大小）。
        # 4D转换成2D的  64*5*5->1600
        x=torch.flatten(x, 1)
        #将展平后的输入通过第一个全连接层 fc1。
        x=self.fc1(x)
        #应用激活函数 ReLU（Rectified Linear Unit），将线性输出非线性化
        x = F.relu(x)
        #在激活后加入 Dropout，随机丢弃一些神经元，减少过拟合。
        x = self.dropout(x)
        #将数据通过第二个全连接层 fc2，生成 10 个类别的输出。
        x=self.fc2(x)
        #应用 log-softmax 激活函数，对输出进行归一化处理，得到对数概率
        #dim=1 表示在类别维度上计算 softmax，即对每一行（每个样本的类别分数）进行归一化。
        output = F.log_softmax(x, dim=1)
        #返回模型的输出，通常是类别的对数概率
        return output
#创建一个 Net 类的实例（定义的神经网络模型）
#将模型移动到指定的设备（device），通常是 CPU 或 GPU。
#例如，如果 device = torch.device('cuda')，则模型会被移动到 GPU 上。
model = Net().to(device)

#定义训练模型的逻辑
def train_step(data, target,model,optimizer):
    optimizer.zero_grad() #优化器梯度置为0
    output=model(data)
    #nll代表负negative log likely hood 负对数似然,得到平均损失
    loss = F.nll_loss(output, target)
    #反向传播的本质是求梯度
    loss.backward()
    #应用梯度去调参
    optimizer.step()
    return loss.item()

#定义测试模型的逻辑
def test_step(data, target,model,test_loss,correct):
    output=model(data)
    #累积的批次损失
    test_loss+=F.nll_loss(output, target,reduction='sum').item()
    # 获得对数概率最大值对应的索引号，即类别
    #参数 dim=1 指定沿着类别的维度（每个样本的预测结果）找最大值
    # keepdim=True 保持输出的维度与输入相同，这有助于后续的操作。
    pred=output.argmax(dim=1,keepdim=True)
    #比较预测类别（pred）与实际类别（target）
    #target.view_as(pred) 确保 target 的形状与 pred 相匹配
    #pred.eq() 检查预测是否与目标相等
    # .sum().item() 计算此批次中正确预测的总数
    correct+=pred.eq(target.view_as(pred)).sum().item()
    return test_loss,correct

#创建训练调参使用的优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 设置整个数据集将被网络处理的次数，称为周期（epochs）。
EPOCHS = 5
for epoch in range(EPOCHS):
    # 将模型设置为训练模式，启用特定于训练的功能，如丢弃（dropout）和批量归一化（batch normalization）
    model.train()
    #遍历来自 train_loader 的每个数据批次 enumerate会返回批次
    for batch_idx, (data, target) in enumerate(train_loader,start=1):
        #将数据和目标移动到计算设备（CPU或GPU）上。
        data,target=data.to(device),target.to(device)
        loss = train_step(data, target,model,optimizer)
        if batch_idx % 10 == 0:
            print('train epoch：{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100.0*batch_idx/len(train_loader), loss))

    model.eval()
    test_loss = 0
    correct = 0
    #是 PyTorch 中用于禁止梯度计算的上下文管理器。在该上下文中的所有操作都会关闭梯度计算，这通常用于推理阶段（而不是训练阶段），从而减少内存消耗和加速计算。
    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.to(device),target.to(device)
            test_loss,correct=test_step(data, target,model,test_loss,correct)

        test_loss /= len(test_loader.dataset)
        print('\ntest set : average loss :{:.4f},accuracy:{}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.0*correct/len(test_loader.dataset)))

