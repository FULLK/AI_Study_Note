@[toc]
# 语言模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15c09bb72e8e4e28a87dcc153cc7c88d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16d2c4f01f484cbda6a33a91b4cb8831.png)
#   统计语言模型
   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4937dfe1c4a54dc982e3f103fe76d604.png)
   n-gram模型的缺点是当某个词组在训练集中没有出现过时，模型会给出零概率，这个问题通常通过“平滑”技术来解决。平滑方法通过对低频词组分配一个较小的非零概率，避免了零概率的问题。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1109a8584d394f628cf9a4d6921e9ebe.png)

- 最大熵原理的核心思想是：在所有符合已知条件的概率分布中，选择那个让系统不确定性（熵）最大的分布。也就是说，我们希望在已知一些信息的基础上，选择一个最符合这些信息但又尽量保持最大不确定性的概率分布。

  例如，假设你在预测某个词是动词还是名词。根据训练数据，你知道某些词更有可能是动词，而某些词更有可能是名词。最大熵模型会通过最大化熵来选择一个概率分布，使得在已知数据约束的情况下，模型不会对任何一个标签（动词或名词）做过强的假设，而是选择一个最“中立”的分布，直到数据给出更多信息来确定正确的标签。

- 模型参数的确定：最大熵模型的“参数”指的是模型中需要学习的东西，通常是概率分布中的一些权重或系数。通过最大化熵的目标，模型会学习如何设置这些参数，使得它在符合训练数据的条件下，能够产生最大熵的概率分布。

- 简单的例子：
假设你正在做文本分类，目标是判断一个词语是动词还是名词。假设我们有以下信息：
“吃”通常是动词
“苹果”通常是名词
当你用最大熵模型来训练时，模型的目标就是学习如何根据这些已知的训练数据，最大化熵，也就是让每个词的标签（动词或名词）尽量不确定，直到数据提供了足够的信息来确定正确的标签。例如，如果给定上下文信息（如“我吃苹果”），模型会逐渐调整参数，减少不确定性，从而作出正确的预测。


# 神经网络语言模型(NNLM)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/749203923c02438091cc20fa0f2566a3.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b3534549c934899844f8d7b6643faf8.png)
神经网络语言模型（NNLM）是一种基于神经网络的语言模型，具有比传统的n-gram语言模型更好的性能。在近年来，随着深度学习技术的发展，神经网络语言模型也得到了广泛的研究和应用。常见的神经网络语言模型有以下几种：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2915d1b38c46b88260abe8229d7387.png)
前馈神经网络是一种最基本的人工神经网络类型，它由输入层、隐藏层和输出层组成。其特点是信息在网络中单向传播：从输入层传递到隐藏层，再从隐藏层传递到输出层，没有循环结构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d94ee21b18f46e5a15f7b7508430321.png)
综上所述，预训练语言模型的发展经历了从Word2vec到ELMo再到Transformer，并出现了一系列基于Transformer的预训练语言模型，如GPT系列，这些模型都具有着在不同领域中应用广泛以及效果卓越的特点。

# 词向量技术
两个节点与另外两个节点平行，说明这两个节点可能与另外两个节点关系相近
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d5f5b5bdc344228b9de6164d7d03690.png)
 独热编码：就是所有词一个表，是这个词，那么对应位置就是1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c2e67c0953b941dbab267f4690d4264d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e4d1189b0d74507b58433503ead44a6.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c6fd736e4cb4be49ccb4dac2b7cb85f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32b3692905c04314b4e981e34d104501.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2e0a2bb6170494ebdc2fa47e55269ab.png)

# 词向量如何计算？
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd390e1e7f4d47f8b3fdf4b65d30b6f4.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e6b1cac3bfe044b2a06990d694f2935b.png)
从整体上来说，词向量模型背后对应着一个神经网络 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/38d04c2c1cfb445c87bf2b061d188ccc.png)
隐藏层的权重就是E矩阵，向量维度个数就是隐藏层的神经元个数。但最后输出层是onehat编码的个数，代表每个词的可能性


具体训练神经网络方式上又有两种：
CBOW：根据上下文预测出来当前这个词是什么 
Skip-gram：根据当前这个词来预测上下文
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d6005c70716c43f696283ef16cf49070.png)
训练方式，都是通过窗口大小为5进行滑动得到数据进行训练

CBOW 的核心思想是根据上下文词来预测中心词。在图中，中心词是句子中的目标词，而上下文词是目标词周围的词。Skip-gram 与 CBOW 正好相反，它是通过目标词来预测上下文词。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06a1cd7630d34f45a40d374104d1cfb8.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f4194e593f246edbe28988a02b89762.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4db85650875143d59cd9c5faa2670549.png)
# 循环神经网络
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1379c59ade342698dd9538277241b00.png)
为什么需要RNN？

DNN都只能单独的去处理一个个的输入，前一个输入和后一个输入是完全没有关系的。但是，某些任务需要能够更好的处理序列的信息，即前面的输入和后面的输入是有关系的。

比如，当我们在理解一句话意思时，孤立的理解这句话的每个词是不够的，我们需要处理这些词连接起来的整个序列； 当我们处理视频的时候，我们也不能只单独的去分析每一帧，而要分析这些帧连接起来的整个序列。

以 "我 吃 苹果" 三个词的句子为例子，这里苹果之所以我们会理解是个水果，就因为前面是"吃"，因为一般大概率人不会吃苹果手机。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/815bbc377a0b41a78b22efec37bee190.png)
U：输入到隐藏层的权重矩阵。
V：隐藏层到输出层的权重矩阵。
W：隐藏层到隐藏层的权重矩阵，表示不同时间步之间的状态转移。

如果按时间轴展开就是右侧图；这里需要注意的是U、V、W是不同时刻共享的；t  t-1 t+1分别是不同时刻的输入，前一个时刻的隐藏层输出会通过w矩阵给下一个时刻的隐藏层。


W矩阵是上一个时刻的隐藏层到下一个时刻的隐藏层的参数矩阵。是一个全连接，也就是t-1时刻的隐藏层节点到t时刻的隐藏层节点之间是全连接。参数矩阵自然n*n的矩阵
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/87b2cc5c0bcf4be585cdb81fc190a939.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a93f91a9319a45a5a7be6670400493dd.png)
具体如下，相当于一个隐藏节点接受两种输入，两种参数矩阵
隐藏状态的计算公式：h_t = tanh(U * X_t + W * h_(t-1))
- U 和 W 是 RNN 的权重矩阵，分别对应输入和隐藏状态的加权矩阵。
- X_t 是当前时刻的输入。
- h_(t-1) 是上一时刻的隐藏状态。
- 通过激活函数 tanh 处理这个加权和，从而得到新的隐藏状态 h_t。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e5d869181fcf4837b9ad9633a8f25cd5.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/56beea6adb4f463b95d97f457e15cdb2.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6318364cc489406cab371c9a81318c58.png)
# 长短时记忆单元
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a8aec5dc98534235b0ff3a0d376dde97.png)
 循环神经网络存在短期记忆问题。 如果一个序列足够长，它们将很难将信息从较早的时间步传递到较晚的时间步。 因此，如果你尝试处理一段文本来进行预测，Basic RNN 可能会从一开始就遗漏重要信息。

在反向传播过程中，递归神经网络会遇到梯度消失问题。 梯度是用于更新神经网络权重的值。 梯度消失问题是指梯度随着时间向后传播而收缩。 如果梯度值变得非常小，它不会对学习有太大贡献。

所以在递归神经网络中，获得小梯度更新的层停止学习。 这些通常是较早的层。 因此，由于这些层不学习，RNN 可以忘记它在较长序列中看到的内容，从而仅拥有短期记忆。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3492742ec946410da63a82562fce9011.png)
         
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/17d15d004a7c4753a8b7778fbf374dcf.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/def55184aa90405eb293ba0af8011d4f.png)

## 遗忘门
![-](https://i-blog.csdnimg.cn/direct/5c3f4100d57c489ea4be3cf51239a518.png)

## 输入门

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69f3949eb22941409f1dedbf3ac377c9.png)


## 单元状态更新
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8a594b2d64e4f2989276f212ff1531f.png)


## 输出门
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5f25c1c5f3d4df78cec8ec26139a3b2.png)
## 总体

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0ad042a939764e32959ec2a9d7a10c98.png)


#  Seq2Seq网络模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1c7fb3d55064ea5a8551b2e9429101e.png)
##  one to one

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3bd6a15e2e847b980ddf4706ca64f75.png)
- 输入
固定大小的图片（例如 64×64 像素）。
每张图片输入到模型时会被预处理（如归一化）。

- 输出
一个固定大小的输出（如一个包含两个元素的向量）。
例如 [0.9,0.1] 表示输入图像有 90% 的概率是“猫”，10% 的概率是“狗”。

## one to many
                  
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b6aae79c7dba46d8a670314a076dee2b.png)


- 固定大小的输入（图片）：
输入为一张 224×224×3 的彩色图片。
通过预训练的 CNN（卷积神经网络，如 ResNet）提取图片的特征向量（feature vector）。
- 输出不固定（语句）


##  many to one

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32fea1df7fba480897cc3020a835f9e7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43220ef70077483e98936c87fabba613.png)

- 输入：
输入是一个序列，长度可以变化。例如：
多封邮件（如垃圾邮件检测）。
一篇文章或多段文本（如文本分类或情感分析）。


- 输出：
输出是一个固定大小的结果。例如：
单一分类标签（如“垃圾邮件”或“正常邮件”）。
情感分类（如“正面情感”或“负面情感”）。
文本分类（如科技类 体育类 娱乐类）


## many to many
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5815377a4bd44601a4dc7d5dcc842ff2.png)

- 输入序列：

输入是一个不定长度的序列，例如：
视频的每一帧图像。
句子中的每一个单词。
- 输出序列：

输出是一个与输入长度对应的序列，每一步输出一个结果，例如：
视频帧的标签（动作或物体识别）。
对句子中每个单词的标注（命名实体识别）。
- 同步特性：

输出的每个标签与输入的每个元素（如视频帧或单词）一一对应。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d06070884ada4ff6a74cbb0f53b2ac03.png)

##  Seq2Seq（sequence to sequence）
Seq2Seq（sequence to sequence），对应的是上图中输入输出序列大小不一样的many to many

提出这种模型的作者 (Cho et al. (2014) and Sutskever et al.(2014)) 称它为 encoder-decoder 或者 sequence-to-sequence架构：
         
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1d771332028344c6a414b6f54b00366d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dad6bc172a194b91855c9171927c01c0.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a377c7a2956a4384a1f5dba387084dfd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51efa86f7a19465f9e8554d8b4aee6b8.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ceb3bb25b6ec45a9988e7cc29760d8dc.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ffa62455a892428684915952bb3db492.png)


#  Attention注意力机制

在人工智能领域，注意力机制也被广泛地应用。它可以帮助计算机模型关注到与当前任务有关的特征，而忽略掉与任务无关的特
征，从而提高模型的准确率。

通过给予不同的权重，使得模型能够更有针对性地学习和处理数据，从而提高模型的准确率。

注意力机制就是通过一系列数学计算，给模型分配一些权重，使模型能够更加关注与当前任务相关的信息。这些权重越大，表示该位置对解决问题的贡献也越大，反之则不那么重要。


以机器翻译为例



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3ce24172d37412ea8632a5c89c54b2c.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a0ead5a59a6488fb253e32d7bb65e47.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bdbfc1209e654ef7a0256828916fd336.png)

注意力输出（上下文向量）主要包含来自那些被赋予较高注意力的编码器隐藏状态的信息。

有时我们会将前一个时间步的注意力输出作为输入，连同通常的解码器输入一起输入到解码器中。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a1b8ff53614148b3a9c743123125384f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b95670d467542978c0855871b5a1484.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c6ba00e74eaa4244a1b6127ccc11c458.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7499996df08f441b8ded46cc3a4a8170.png)
注意：解码器的输入是当前时间步的输入单词（通常是目标语言中的单词索引）和 前一个时间步的隐藏状态

# 注意
## detach

```python
     for di in range(target_length):
            # 一个时刻一个时刻的完成decoder的forward正向传播
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 如果不使用teacher forcing，我们就是需要自己拿到当前这一时刻的预测，当成下一个时刻的输入
            #topk(k) 是 PyTorch 中的一个函数，用于选取前 k 个最大值及其对应索引。
            #topv：选中的最大值（即当前时间步概率最高的单词的对数概率） 形状为 [1, 1]。
            #topi：选中的最大值的索引（即当前时间步预测的单词在词表中的索引） 形状为 [1, 1]。
            topv, topi = decoder_output.topk(1)
            #squeeze()：将 topi 的形状从 [1, 1] 变为标量值（1 维）。

            decoder_input = topi.squeeze().detach() #梯度只在当前时间步内传播，不会回传到更早的时间步。
            loss += criterion(decoder_output, target_tensor[di])
            # 如果预测出来的词是<EOS> 那么就没必要下一时刻传递进去了
            if decoder_input.item() == EOS_token:
                break
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4067d4e8b624692ac1f6a04c09b15b3.png)

```python
decoder_input = topi.squeeze().detach()

```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f7eb568d020d47f68d6de61e11ab0233.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ced5d4857c140adbd1791e944a1d9b9.png)
