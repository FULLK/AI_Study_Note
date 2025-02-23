from sklearn import datasets
from sklearn.pipeline import make_pipeline
#make_pipeline 是 scikit-learn 中的一个便捷函数，用于快速构建一个机器学习的流水线（pipeline）将数据处理模型训练等多个步骤串联起来
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer 是 scikit-learn 中用于文本特征提取的一个类，它将文本数据转换为可以被机器学习算法使用的数值型特征矩阵。
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# CountVectorizer到底做了啥
corpus = [
 'This is a sample text.',
 'This text is another example text.',
 'This is just another text.'
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# fit 阶段：CountVectorizer 分析 corpus 中的所有文档，构建一个全局词汇表，并确定每个词的索引。
# transform 阶段：根据构建好的词汇表，将每个文档转换为一个词频向量，形成一个稀疏矩阵 X。

print(X.toarray())
print(vectorizer.get_feature_names_out())
# [[0 0 1 0 1 1 1]
#  [1 1 1 0 0 2 1]
#  [1 0 1 1 0 1 1]]
#这个矩阵表示的是每个文档中各个词汇出现的次数，其中每一行代表一个文档，每一列代表一个词汇。
#['another' 'example' 'is' 'just' 'sample' 'text' 'this']



# 加载20newsgroups数据集合
newsgroups_train=datasets.fetch_20newsgroups(subset='train')
newsgroups_test=datasets.fetch_20newsgroups(subset='test')
#subset: 选择要加载的数据子集，可以是 'train'（训练集）、'test'（测试集）或 'all'（全部）。默认情况下，它会加载训练集。
#data_home:参数用于指定一个自定义的路径，以便下载和缓存 scikit-learn 的数据集。如果你不提供具体的路径（即设置为 None），那么 scikit-learn 会默认将所有数据集存储到用户主目录下的 scikit_learn_data 文件夹中


# 创建一个pipeline 用于文件特征提取,再使用逻辑回归
# 逻辑回归是拿到数值化的数据做模型
# 文本需要量化成数值才能被逻辑回归处理，CountVectorizer()量化文本变成数值
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000)) # 增加迭代次数
#LogisticRegression的fit_intercept : bool, default=True 是 LogisticRegression（以及其他线性模型）中的一个参数，用于指定是否在决策函数中添加一个常数项（也称为偏置项或截距项）或者说W0,W1后才是参数权重
#
# 使用训练集训练模型,本质先是CountVectorizer特征提取数值化，然后结果给LogisticRegression再fit
pipeline.fit(newsgroups_train.data, newsgroups_train.target)

# 在测试集上预测 本质先是CountVectorizer特征提取数值化，然后结果给LogisticRegression预测
y_pred=pipeline.predict(newsgroups_test.data)

# 准确率

print("准确率: %.2f" % accuracy_score(newsgroups_test.target, y_pred))

