import numpy as np
from sklearn.cluster import KMeans
# 初始化数据点
X=np.array( [[1,2],[1,4],[1,0],
             [4,2],[4,4],[4,0]] )

# 初始化模型
# n_clusters聚类中心点个数
#random_state: int，随机状态实例或None， default=None
# 确定聚类中心初始化的随机数生成
# 使用int使随机性具有确定性。
kmeans=KMeans(n_clusters=2,random_state=0)

# 训练模型
kmeans.fit(X)

#输出X的聚类结果
print(kmeans.labels_)

# 输出聚类中心点
print(kmeans.cluster_centers_)


# 纯底层实现
print("纯底层实现，不依赖机器学习库")
import matplotlib.pyplot as plt

def Kmeans(X,k,max_iter=100):
    center_index=np.random.choice(X.shape[0],k,replace=False)# 从样本中挑选k个作为中心，不能有重复的
    # center_index  [0 5]
    print("center_index ",center_index)

    X_center=X[center_index]
    #中心点
    print("X_center ",X_center)
    #标签
    labels=np.zeros( (X.shape[0]) )
    print("labels ",labels)

    print("X_center[:,np.newaxis]",X_center[:,np.newaxis])
        # 分配样本到最近的中心点
    for i in range(max_iter):
        # 计算到每个点到两个中心的距离
        distances=np.sqrt(np.sum( (X-X_center[:,np.newaxis])**2,axis=2))
        print("distances ",distances)

        # 计算此时的标签结果 离那个标签最近
        newlabels=np.argmin(distances,axis=0)
        print("newlabels ",newlabels)

        #更新中心点  相同标签的点的平均值作为新的聚类中心点
        for i in range(k):
            X_center[i]=X[newlabels==i].mean(axis=0)

        print("new X_center ",X_center)

        #如果聚类的结果没有变化，则提前结束迭代
        if(newlabels == labels).all():
            break
        else:
            labels=newlabels

    return labels,X_center


XX=np.vstack((np.random.randn(10,2)*0x75+np.array([1,0]),
              np.random.randn(10,2)*0x25+np.array([-0.5,0.5]),
              np.random.randn(10,2)*0x5+np.array([-0.5,-0.5])
              ))
print("XX ",XX)

labels,X_center=Kmeans(XX,3)

plt.scatter(XX[:,0],XX[:,1],c=labels)
plt.scatter(X_center[:,0],X_center[:,1],marker='x',s=200,linewidths=3,c='r')
plt.show()