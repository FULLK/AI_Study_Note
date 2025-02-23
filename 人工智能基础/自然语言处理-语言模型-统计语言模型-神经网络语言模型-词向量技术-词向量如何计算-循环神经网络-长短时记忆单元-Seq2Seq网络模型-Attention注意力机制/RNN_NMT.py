# Seq2Seq机器翻译：法文->英文
import random
import time

import torch
import unicodedata
import re
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 字典，需要加入特殊的字符 <SOS> <EOS>
SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}  #每个词对应的索引
        self.word2count = {}  #每个词出现的次数
        self.index2word = {0:"SOS",1:"EOS"}  #每个索引对应的词
        self.n_words = 2      #共出现的词的个数

    def addSentence(self,sentence):
        for word in sentence.split(' '):  #分词
            self.addWord(word)  #对每个词加入字典


    def addWord(self,word):
        #如果词是第一次出现，就加入字典
        if word not in self.word2count:  #会被检查是否是 字典的键
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.n_words += 1
        #如果不是第一次出现，就将其出现的次数加1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#lang1: 源语言（如英语）。
#lang2: 目标语言（如法语）。
#reverse: 布尔值，表示是否反转语言对的顺序。若 reverse=True，则源语言和目标语言互换。
def readlangs(lang1,lang2,reverse=False):
    #read(): 读取整个文件内容。
    #strip(): 去掉文件内容两端的空白字符。
    #split('\n'): 将文件按行拆分，返回一个包含每行文本的列表。
    lines=open("./data/%s_%s.txt"%(lang1,lang2),"r",encoding="utf-8").read().strip().split("\n")
    #每行是由源语言句子和目标语言句子通过制表符（\t）分隔的。split('\t') 会将每行的内容分割为两个部分：源语言句子和目标语言句子。
    #\t 是一个控制字符，用于表示水平制表符（Tab）
    pairs=[[normalizeString(s) for s in l.split("\t") ] for l in lines]
    #如果 reverse 参数为 True，则会反转每对句子的顺序。即原来的源语言句子会变成目标语言句子，目标语言句子会变成源语言句子。
    if reverse:
        #reversed(p) 是 Python 的内建函数，用于返回一个反转的迭代器。对于列表 p，它会将列表的顺序反转。
        #reversed(p) 返回一个迭代器，而 list() 会将这个迭代器转化为一个列表。
        pairs=[list(reversed(p)) for p in pairs]
        input_lang=Lang(lang2) #法语输入
        output_lang=Lang(lang1) #英语输出
    else:
        input_lang=Lang(lang1)
        output_lang=Lang(lang2)
    return pairs, input_lang, output_lang


MAX_LENGTH=10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
#对每对英法语句进行过滤
#过滤长度<10并且英语语句以eng_prefixes开头的

def filterpair(pair,reverse=False):
    #当 startswith 参数是一个元组时，如果字符串以元组中的任何一个字符串开头，则返回 True，否则返回 False。
    return len(pair[0].split(" "))<MAX_LENGTH and len(pair[1].split(" "))<MAX_LENGTH and pair[1 if reverse else 0].startswith(eng_prefixes)

def filterpairs(pairs,reverse=False):
    return [pair for pair in pairs if filterpair(pair,reverse)]

def preparedata(lang1,lang2,reverse=False):
    pairs, input_lang, output_lang = readlangs(lang1,lang2,reverse)
    print("pairs:",len(pairs))
    filtered_pairs = filterpairs(pairs,reverse)
    print("filtered_pairs:",len(filtered_pairs))
    #对每对语句进行处理，将里面的词分别加入各自的字典
    for pair in filtered_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("input_lang nwords :",input_lang.n_words)
    print("output_lang nwords :",output_lang.n_words)
    return input_lang, output_lang,filtered_pairs

input_lang, output_lang,pairs=preparedata("eng","fra",reverse=True)
#random.choice 方法：这是 Python random 模块中的一个方法，用于从一个非空序列（如列表、元组或字符串）中随机选择一个元素。
#它的输入是一个可迭代的序列，输出是从该序列中随机选中的一个元素。
print(random.choice(pairs))

# 编码器神经网络部分
class EncoderRNN(nn.Module):
    #input_size代表输入层神经元个数
    #hidden_size代表隐藏层神经元个数
    def __init__(self, input_size, hidden_size):
        #super(当前类名, 当前实例).__方法名__()
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 词嵌入层,输入层传进来的每个时刻的单词，首先都会经过词嵌入变成向量
        #nn.Embedding 是一个查找表，它的作用是：输入一个索引（如单词的整数编码）。输出该索引对应的嵌入向量。
        #嵌入层内部维护了一个形状为 (output_size, hidden_size) 的矩阵，通常称为嵌入矩阵或词向量矩阵：
        #行数为 output_size（词汇表大小），每一行对应词汇表中的一个单词。
        #列数为 hidden_size，表示每个单词的稠密向量表示的维度。
        self.embedding = nn.Embedding(input_size, hidden_size)
        #GRU（门控循环单元，Gated Recurrent Unit） 层，GRU 是一种改进的循环神经网络（RNN）
        #GRU 是一种循环神经网络（RNN）的变体 Cell，和 LSTM（长短时记忆网络） Cell类似，用于处理序列数据。
        #nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
        #input_size:输入特征的维度，即每个输入向量的大小。
        #hidden_size:隐藏状态的特征维度。 前一个时刻的Cell的输出
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self,input,hidden):
        # view(1, 1, -1) 相当于把数据reshape成了3维数组
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    def initHidden(self):
        #前三个参数代表三个维度的值
        return torch.zeros(1, 1, self.hidden_size,device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        #Linear全连接连接的就是GRU隐藏层到输出层，所以W矩阵形状就是
        # (hidden_size)*(output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output=F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output=self.softmax(output)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,device=device)


#带有注意力机制的decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length=max_length
        # 我们input_size和output_size大小是一样的
        # 说白了都是词典里面词的数量，one-hot编码中向量的长度
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    #对于有注意力机制的Decoder来说，每一时刻它都需要Encoder的所有时刻的输出
    def forward(self,input,hidden,encoder_outputs):
        # 首先把输入做词嵌入 和encoder一侧是一样的
        embeded = self.embedding(input).view(1, 1, -1)
        embeded=self.dropout(embeded)
        # 开始实现Attention Layer
        # 大家需要注意，这个代码的AttentionLayer和理论讲的稍有不同
        # 但是也可以 这就说明Attention有各种变化的可能性
        scores=self.attn(torch.cat((embeded[0],hidden[0]),1))# embeded, hidden 都是来自于decoder的
        attn_weights = F.softmax(scores,dim=1)
        # 继续计算 attn_applied，其实它就是理论部分说的context vector
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.unsqueeze(0))
        # 计算出了context vector之后，再次回到 decoder这一端
        output=self.attn_combine(torch.cat((embeded[0],attn_applied[0]),1)).unsqueeze(0)
        output=F.relu(output)
        #正常的GRU单元的隐藏层
        output,hidden=self.gru(output,hidden)
        output=self.out(output[0])
        output=F.log_softmax(output,dim=1)
        return output, hidden,attn_weights
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,device=device)

#训练需要数据，我们的数据此时是pairs，但是它的类型并不是pytorch里面的tensor
#这里需要把每一句话的一堆词变成一堆索引号
def IndexFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def TensorFromSentence (lang,sentence):
   Indexs=IndexFromSentence(lang,sentence)
   Indexs.append(EOS_token)
   #将原张量的形状调整为两维：最后一维（第二维）固定为 1  前面所有维度自动计算并展平为一个维度。
   return torch.tensor(Indexs,dtype=torch.long,device=device).view(-1,1)


# 将pair转换为pytorch能接受的张量
def TersorFromPair(pair):
    input_tensor=TensorFromSentence(input_lang,pair[0])
    target_tensor=TensorFromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)

#一次迭代所需要训练逻辑，就是一个pair，输入为英文，输出为法文
def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
    # 初始化全为0的encoder第一个时刻隐藏层的输入
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 法语的句子长度，和对应英文的句子长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化每一个时刻encoder隐藏层向上的输出
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    #首先encoder这一侧的正向传播
    for ei in range(input_length):
        # ei 代表着就是encoder这一侧每一个时刻的输入的索引
        # input_tensor[ei] 代表着就是encoder这一侧每一个时刻的输入
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # 当上面的for循环执行完成，encoder一侧的正向传播就完成了
    # 接着就该开始decoder了, decoder第一个时刻的输入就是 < SOS >
    decoder_input = torch.tensor([[SOS_token]], device=device)
    #把encoder最后一个时刻的隐藏层的向右输出交给decoder当成第一个时刻隐藏层的输入
    decoder_hidden = encoder_hidden
    # decoder这一端每一个时刻的输入分两种情况，是否去使用teacher forcing
    # 如果是 teacher forcing 就是每一个时刻的输入是事先训练集当中的对应的上一时刻的target
    # 如果不使用 teacher forcing 就是每一个时刻的输入是上一个时刻的输出
    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            # 一个时刻一个时刻的完成decoder的forward正向传播
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #把每一个时刻的decoder端的loss累加一下
            loss += criterion(decoder_output, target_tensor[di])
            ## 因为是tearcher forcing所以每一个时刻的输入就来自于target
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            # 一个时刻一个时刻的完成decoder的forward正向传播
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 如果不使用teacher forcing，我们就是需要自己拿到当前这一时刻的预测，当成下一个时刻的输入
            #topk(k) 是 PyTorch 中的一个函数，用于选取前 k 个最大值及其对应索引。
            #topv：选中的最大值（即当前时间步概率最高的单词的对数概率） 形状为 [1, 1]。
            #topi：选中的最大值的索引（即当前时间步预测的单词在词表中的索引） 形状为 [1, 1]。
            topv, topi = decoder_output.topk(1)
            #squeeze()：将 topi 的形状从 [1, 1] 变为标量值（1 维）。
            #在非 Teacher Forcing 模式下，解码器会使用自己前一时间步的预测作为当前时间步的输入。
            #如果没有调用 detach()，当前时间步的输入（decoder_input）会与之前时间步的计算图关联，导致梯度在时间步之间无限制传播。
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            # 如果预测出来的词是<EOS> 那么就没必要下一时刻传递进去了
            if decoder_input.item() == EOS_token:
                break
    loss.backward()#得到梯度
    #分别调参
    encoder_optimizer.step()
    decoder_optimizer.step()
    return  loss.item() / target_length


import time
import math

#将输入的秒数 s 转换为 "分钟 秒" 的格式，并返回一个格式化的字符串。
def asMinutes(s):
    #计算分钟数（整除 60）
    m = math.floor(s / 60)
    #从总秒数中减去已转换为分钟的部分，得到剩余的秒数。
    s -= m * 60
    #返回一个字符串，格式为 X minutes Y seconds，即分钟数和秒数。
    return '%dm %ds' % (m, s)

#算从 since 到当前时间所经过的时间，并估算剩余时间。
#since：开始时间的时间戳percent：任务完成的进度（百分比，0 到 1）。
def timeSince(since, percent):
    #获取当前时间（时间戳）。
    now = time.time()
    #计算已经过去的时间。
    s = now - since
    #估算完成整个任务所需的总时间。
    es = s / (percent)
    #计算剩余时间（es 总时间 - 已经过去的时间 s）。
    rs = es - s
    #使用 asMinutes 格式化时间并返回已用时间和剩余时间。
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    #创建一个新的图形窗口。
    plt.figure()
    #创建一个包含单个子图的图形。
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #设置 y 轴刻度的间隔为 0.2。
    loc = ticker.MultipleLocator(base=0.2)
    #将 y 轴的刻度设置为 loc（即每隔 0.2 显示一个刻度）。
    ax.yaxis.set_major_locator(loc)
    #绘制传入的 points 数据，并显示图形。
    plt.plot(points)

def trainIters(encoder,decoder,n_iters,print_every=1000,plot_every=100,learning_rate=0.01):
    start=time.time()
    plot_losses=[]
    print_loss_total=0
    plot_loss_total=0
    encoder_optimizer=torch.optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer=torch.optim.SGD(decoder.parameters(),lr=learning_rate)
    training_pairs=[TersorFromPair(random.choice(pairs)) for i in range(n_iters)]
    #nn.NLLLoss() 是 PyTorch 中用于多分类任务的损失函数，
    # NLL 代表 Negative Log Likelihood（负对数似然），通常用于处理分类问题，特别是在输出是对数概率分布时。
    criterion=nn.NLLLoss()
    for iter in range(1,n_iters+1):
        train_pair=training_pairs[iter-1]
        input_tensor=train_pair[0]
        target_tensor=train_pair[1]
        loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total+=loss
        plot_loss_total+=loss
        if iter % print_every == 0:
            print_loss_avg=print_loss_total/print_every
            print_loss_total=0
            print('%s (%d %d%%) %.4f'%(timeSince(start,iter/n_iters),iter,iter/n_iters*100,print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg=plot_loss_total/plot_every
            plot_loss_total=0
            plot_losses.append(plot_loss_avg)

    showPlot(plot_losses)

#评估
def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor=TensorFromSentence(input_lang,sentence)
        input_length=input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention
            topv, topi = decoder_output.topk(1)
            decoder_input=topi.squeeze()
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()] )
        return decoded_words, decoder_attentions[:di+1]

def evaluateRandomly(encoder,decoder,n=10):
    for i in range(n):
        pair=random.choice(pairs)
        print(">",pair[0])
        print("=",pair[1])
        output_words, decoder_attentions = evaluate(encoder,decoder,pair[0])
        output_sentence=" ".join(output_words)
        print("<",output_sentence)

#调用封装函数，完成训练和评估流程
hidden_size = 256
#encoder是要接收要去翻译的法语
encoder1=EncoderRNN(input_size=input_lang.n_words, hidden_size=hidden_size).to(device)
#decoder是拿着encoder传递过来的信息，要去把法语翻译成英语
attn_decoder1=AttnDecoderRNN(hidden_size=hidden_size, output_size=output_lang.n_words, dropout_p=0.1).to(device)
#通过多次迭代进行训练
trainIters(encoder=encoder1,decoder=attn_decoder1,n_iters=5000,print_every=5000)
#评估
evaluateRandomly(encoder=encoder1,decoder=attn_decoder1)
#可以传入自己定义的一句话让它翻译
output_words,attentions=evaluate(encoder=encoder1,decoder=attn_decoder1,sentence="je suis trop froid .")
#plt.matshow() 是 Matplotlib 中用于显示矩阵的函数。它将矩阵中的每个值映射到颜色上，显示一个热力图。
plt.matshow(attentions.numpy())


#该函数用于将注意力权重可视化为热力图，并显示输入和输出句子的对应关系。
# 显示输入句子和输出句子的每个单词之间的注意力权重。
# 使用热力图表示注意力权重，颜色强度代表了模型在生成每个词时对输入的关注程度。
def showAttention(input_sentence, output_words, attentions):
    # 创建一个新的图形窗口
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 绘制热力图
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # 准备输入标签
    input_words = input_sentence.split(' ')

    # 设置 x 轴
    ax.set_xticks(range(len(input_words) + 2))  # +2 是为了包含空字符串和 <EOS>
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=90)

    # 设置 y 轴
    ax.set_yticks(range(len(output_words)))
    ax.set_yticklabels(output_words)

    # 设置刻度间隔
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")

