@[toc]
# 感知机
感知机是神经网络的起点
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c8d976252ad41799b77a7a19fd799e1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d7ebfe135eb24013a927b0fec7727df2.png)
## 工作流程

在感知机模型中，step function起到了一个 分类决策 的作用。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff1875ee040b43a4b3a57a2f8c527a1b.png)
学习调整权重
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce5be4cbec524dd09e128f01ea214478.png)
# 神经网络

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8b9c6a19b74a4349a18e2f2cb065d988.png)
## 区别
无需自己选择特征，传统感知机人工手动提取选择特征，深度学习特征提取的过程是自动完成的。只需要输入各种数据即可
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d3220d758c24bde9f4cf6e57cdda2c8.png)
提取特征可以通过最后输出层的前一层隐藏层的输出特征可以看出来，与一开始的输入层的特征不一样
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aac596de11bf4964ad81a98884124fd5.png)
可以用作多分类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c3204388025450d990d7029e5d00b20.png)
或者目标检测
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba9f60f57179428e8a0410ae3832fab8.png)
## 各种各样的神经网络
DNN 深度神经网络（ANN 人工神经网络 /MLP 多层感知机）、CNN、RNN
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6c77e26642d4406b1407f2e75e73cf7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9be6661ee7cf44ebbef17dc51b4994a7.png)
# 激活函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c5691a85a80a48fdbfe4a6bf9d063603.png)
激活函数是一种非线性函数，它可以把输入的数据映射到一个新的输出值，这个输出值可以在下一层被用作输入。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d654d2714a440c0b946f53717bff1b4.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3207f4c404745979351fc3db5628f5c.png)
## 激活函数类型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4740643b92e24af28c9f14bb58f121ba.png)
### Sigmoid 函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a8bf8e86dcf544f4a1bdc1486405a3ea.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/641f8abbda9845ae8aff21cf6bd412eb.png)
### ReLU函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e4f1666b07147a8a04611cd39ec0c3c.png)
### Leaky ReLU 函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f23d77f4f1d14d0ebfcb58943120ff4e.png)
### Tanh 函数
虽然两者都是非线性的，但 Tanh 函数在输入接近零时提供了更强烈的非线性映射，这是因为其曲线在这个区域内更为陡峭。这使得它能够在某些任务中捕捉到更复杂的模式。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e2826f17c864d8bbc3419a50c5e5c5b.png)
# 正向传播
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e86b76efabd4bcba13e1cf3e7ec1024.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd807cd2ce15435c8eae6d54fdac8828.png)
# 反向传播
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5500de2c605c4135a670e0948f0b292d.png)
负责多少就是对损失值影响多少，就是梯度
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/671815a174c3484abae7ea0d3b7c7529.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6574a2a85c5240cf9eb806550bd7f623.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/541d1e0ad2cc404e9d6eb82648a11a0f.png)
# 梯度消失(gradient vanish)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78d9533727884ce7a9df4c78a0f456eb.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2dc75a25a5f0418eb2e2bb2b1a979223.png)
## 如何解决
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2dffa7427b845f69880c2310e5db71b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d21bb5cdb3f34386b41defee30e529b2.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3140ac4f07ee477bb1fb5d6487b7105d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16255a9503404271acfea4f07fc72312.png)
Sigmoid函数导函数区间在0到0.25之间，Tanh函数导函数区间在0到1之间，它们输出的数值一直很小。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d37d1aacd0443cd91a4cfe44a7073ba.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/388507845845494e88d86e97dbc27932.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/75d42a8faada4021a61460b0bc17eca9.png)
# Dropout
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cbbe11c7cbbc47949d0f5c704d25715e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f653b8951064574a4a4bcf81f39326f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce43806c7a6f4ff68aef26405168f157.png)
消除特征间的依赖关系：不依赖其他特征通过一部分学习的特征就能判断
## 使用
深度学习框架中会封装有 Dropout 层，这样需要将哪一层输出随机的丢弃一些，那么就在哪一层后面跟 Dropout 层即可。

自行实现如下
```python
p = 0.5  # 激活的概率。p越高，Dropout 越少。
def train_step(X):
    """ X 是输入数据 """
    
    # 第 1 层的前向传播
    H1 = np.maximum(0, np.dot(W1, X) + b1)  # ReLU 激活
    U1 = np.random.rand(*H1.shape) < p  # 随机生成 Dropout mask
    H1 *= U1  # 应用 Dropout，丢弃一些神经元
    
    # 第 2 层的前向传播
    H2 = np.maximum(0, np.dot(W2, H1) + b2)  # ReLU 激活
    U2 = np.random.rand(*H2.shape) < p  # 随机生成 Dropout mask
    H2 *= U2  # 应用 Dropout，丢弃一些神经元
    
    # 输出层
    out = np.dot(W3, H2) + b3  # 不对输出层应用 Dropout
    
    return out

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/00c947e7bb57466fa364755732702bf6.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/399b277de8124344a1256c5ef9c15a7b.png)
# PyTorch实战神经网络算法(手写MNIST数字识别)
[https://pytorch.org/](https://pytorch.org/)

```python
inputs.view(-1, 28*28)
```

## view
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/75336e745a004081a7c1d8472086deb1.png)
## softmax和log-softmax
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de03c3e6031a47baa4b2e8877f35fbe6.png)
## cross-entropy loss
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a8562d720f54277b47a70b14e0e14e7.png)
## nll_loss
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e9d5f71cd0e460cbb313144169667fe.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4bb832fc0c05449e8350df400a6dc23d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a80be5df5ec64dd287b40c15bdeab393.png)
## batch size底层运算机制
[https://blog.csdn.net/weixin_44986037/article/details/144216069](https://blog.csdn.net/weixin_44986037/article/details/144216069)