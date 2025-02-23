from sklearn import datasets #datasets 模块内含多个函数和方法，可以用来加载一些经典的数据集
from sklearn.model_selection import train_test_split  # 模型数据工具
from sklearn.linear_model import LinearRegression  # 模型
from sklearn.metrics import mean_squared_error # metrics（度量）指的是用来评估模型性能的一系列标准或方法

diabetes=datasets.load_diabetes()  # 加载糖尿病数据集

# 将数据集拆分为数据集和标签级别
X=diabetes.data
Y=diabetes.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2)

# 创建一个线性回归对象
lr=LinearRegression()

# 使用训练集训练模型
lr.fit(X_train,y_train)

# 对测试集进行预测
y_pred_test=lr.predict(X_test)
y_pred_train=lr.predict(X_train)

# 打印模型的均方误差
print("均方误差：%.2f" % mean_squared_error(y_test,y_pred_test))
print("均方误差：%.2f" % mean_squared_error(y_train,y_pred_train))
