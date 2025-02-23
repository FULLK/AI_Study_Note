@[toc]

# 梯度下降

当前参数-损失函数关于参数的导数=新参数，新参数会往损失函数减少的方向变化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7051630515f0458898dd324960224f16.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/316a8143190548a28590e23c635feaf6.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/041fcf7a537d447b85968443abd18f32.png)

# 小批量梯度下降
 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f2dd0b5ba1914ffbb812f19bc8f19989.png)

小批量梯度下降：每次选择部分数据计算损失率进行梯度下降
随机梯度下降：随机选择数据计算损失率进行梯度下降
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1ad10d54d86e448aa83934df32fc38ba.png)

## 多轮训练 
每轮分别使用各个批次的数据进行梯度下降，然后反复多轮
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3fa262259f7741abb8e14e3819d63046.png)

# 数据归一化


## 归一化原因
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ef0600bafc6945aea468d9ece8aa20fd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a89ee80c4af4434854b109b4ef0fdd1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e9ef229386e34b6aa3d53d22d470b0e7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28d4521f02104f7fb626a5d90a151256.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4c3a747aabf494eb06032b67cf03356.png)
## 损失函数等高线

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c06ae8773dd4138aab37eebf171725d.png)
## 归一化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c4be517c6ba4d24857a087ad465e8a5.png)
#  实战
