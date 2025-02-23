from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris() # 加载数据集
X = iris.data
Y = iris.target

print(X)
# 标签0 1 2三种类别
print(Y)

# 切分数据集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
# random_state保证切分得到的数据集相同还是不同，没写random_state就是它的值就是随机，所以每次切分得到数据集不同
# 数据集不同自然导致训练效果不同

# 创建逻辑回归对象
# 逻辑回归里面封装了softmax回归，如何知道是把多分类转成多个二分类还是使用softmax回归
lc=LogisticRegression(max_iter=10)
# max_iter是迭代调整参数的次数，如下提示就是迭代次数不够到拟合
# C:\Users\35925\anaconda3\envs\AI_Study\lib\site-packages\sklearn\linear_model\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(

# LogisticRegression() 默认是multi_class='auto',如果数据是二分类就选择ovr ,其他就选择multinomial
# LogisticRegression(multi_class='ovr') 多个二分类  但多标签选择多个二分类
# LogisticRegression(multi_class='multinomial') 是softmax回归

# 训练模型  这里逻辑回归会根据数据进行二分类还是多分类
lc.fit(X_train, y_train)

# 对测试集训练模型
y_pred = lc.predict(X_test)

# 打印模型的准确率
print("准确率: %.2f" % accuracy_score(y_test, y_pred))

