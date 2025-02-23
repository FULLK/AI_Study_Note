@[toc]

# 代码
[https://github.com/FULLK/AI_Study/tree/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92](https://github.com/FULLK/AI_Study/tree/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92)
# 什么是回归任务
根据历史数据训练模型，未来预测具体数值的任务就是回归任务
多元线性回归是众多回归算法中最基础的一个

# 什么是多元
多元：多个维度  影响预测目标的多个因素 房屋的售价取决多个因素
三维平面的线性：就是一个平面
更高维度的线性：统称为超平面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b68bc3f53bbf40018e3432c4ddd1d719.png)

# 什么是回归
回归：就是拟合数据点
拟合可以是线性也可以是非线性
三维的线性回归：用平面去拟合数据点
三维的非线性回归：曲面拟合数据点

# 什么是多元线性回归
多元线性回归：多元空间使用线形体（直线 平面 超平面）去拟合数据点


# 表达式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/415d1aafa08645a7829d1c9a6eb81770.png)

# 何时使用多元线性回归
何时使用：猜测是否是的，未知情况尝试是否拟合

# 注意
算上Y，其实是N+1维度，N个X，1个Y
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4161bce2201d40b99fd9e28771b9ce00.png)

Y由一群X决定，具体反映就是线形体上对应X的位置的Y值
# 损失函数
实际Y和预测Y的均方差（Mean Squared Error）：回归任务常见的损失函数，越小拟合效果越好
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32d74e6ecd1d4601a25dd37648ec37c4.png)
