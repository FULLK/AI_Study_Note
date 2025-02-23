from scipy.io.wavfile import read
from  speaker_features import extract_features
import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

source='./speaker-identification/development_set'
train_file='./speaker-identification/development_set_enroll.txt'
dest='./speaker_models/'
file_paths=open(train_file,'r')
features=np.asarray(())
count=1

for path in file_paths:
    path=path.strip()
    print(source+path)
    # 读取声音文件
    full_path = os.path.join(source, path)  # 使用os.path.join正确拼接路径
    print(full_path)  # 打印路径调试
    sr, audio = read(full_path)  # 读取音频文件
    #提取40个维度特征
    vector=extract_features(audio,sr)

    #合并提取的每帧的特征
    if features.size==0:
        features=vector
    else:
        features=np.vstack((features,vector))
    #提取了5个文件，对应一个人
    if count==5:
        gmm=GaussianMixture(n_components=16,max_iter=200,covariance_type='diag',n_init=3)
        # 16 个高斯分布来拟合输入数据
        # 模型优化（EM 算法） 迭代 200 次
        # 协方差矩阵的类型，本例中为 'diag'，表示协方差矩阵是对角矩阵,协方差矩阵定义了高斯分布的形状
        #  n_init 控制模型会随机初始化参数 多次，并分别训练多次,最后选择对数似然值（log-likelihood）最大的那次作为最终结果。
        gmm.fit(features)

        #保存每个人对应的GMM模型，名字是说话人的
        picklefile=path.split('-')[0]+'.gmm'
        pickle.dump(gmm,open(dest+picklefile,'wb'))
        # 清空
        features=np.asarray(())
        count=0
    # 一个一个文件
    count+=1