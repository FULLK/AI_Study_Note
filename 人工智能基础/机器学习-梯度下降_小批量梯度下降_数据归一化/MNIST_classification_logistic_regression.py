from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler  # 预处理中的标准化
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np  # 一种数据格式操作库
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 加载MNIST手写数字图片数据集
mnist=fetch_openml('mnist_784')
#X 和 Y的数据集形状
print(mnist.data.shape)
print(mnist.target.shape)
# 类型
print(type(mnist.data))
print(type(mnist.target))
# X
print(mnist.data)
# Y
print(mnist.target)
print()

# 查看第一个
imag0=np.array(mnist.data)[0]
imag0_reshape=imag0.reshape(28,28)
print(imag0_reshape)
plt.imshow(imag0_reshape, cmap='gray')
#用于在当前图形中显示图像。参数 cmap='gray' 设置颜色图为灰色调
plt.show()
#plt.show() 函数打开一个窗口并显示绘制的所有图形

label0=np.array(mnist.target)[0]
print(label0)
#标准归一化  因为各个数据特征范围要从样本里面来看，因为标签只有1到10，有些特征会一直为0
# 计算并运用
X=StandardScaler().fit_transform(mnist.data)
# 不归一化相比归一化的会出现还没有收敛的警告，因为跨度大收敛难

# 划分数据集和训练集
X_train, X_test, y_train, y_test=train_test_split(X,mnist.target,test_size=0.2,random_state=42)

# 创建逻辑回归模型
model=LogisticRegression(max_iter=1000)

# 训练模型
model.fit(X_train,y_train)

# 测试
y_pred=model.predict(X_test)

# 计算结果的准确率
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: %.2f" % accuracy)

# 预测结果
print("predict imag0: ",model.predict([imag0]))