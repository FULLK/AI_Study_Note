@[toc]
# 逻辑回归
logistic regression 并不是回归任务的算法，而是属于分类任务算法

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cb2c448641eb43778370fe83f13da7ac.png)
## 逻辑回归算法表达式
一个 型曲线（Sigmoid函数），将线性回归的值压缩到了0到1之间，这样就具备了概率含义；大于0.5为正例，反之是负例
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b82bdf2a84bd432b9290d187138f498d.png)

## 模型训练
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2391e51bc57f490688326ce8604b166f.png)

## 逻辑回归做多分类

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bfe06925df034365bb637365e1b76a5a.png)


# softmax回归
多分类问题的机器学习算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b9b310815632423b9a9114ae8b825848.png)
## 模型实例

假设你是一名动物园管理员，你想识别你们园区的动物。你拍了一些照片，其中包含狗、猫和鸟三种不同的动物。你需要一个算法来识别每张照片中的动物是哪一种。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/87ddea454c504616aec0d1f2b664a82a.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c92b7a3018dd442f90c3d395115bdd42.png)
## 训练模型

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/75b3f81ee60542d9ab7c92da79b714c9.png)

# 注意
多标签和多分类的区别
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77273873401947b0aee9f585f8cc4968.png)

- Softmax回归适合多分类模型，因为互斥就是各类别概率之和必须为1； 1
- 逻辑回归多分类转多个二分类适合，因为各个二分类模型是相互独立的；

# 代码

```python
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.
```




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ce2e000377d40e5b90a16e9c7bb2dcb.png)
