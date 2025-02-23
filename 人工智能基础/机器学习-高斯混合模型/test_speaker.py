import time

from scipy.io.wavfile import read
from  speaker_features import extract_features
import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

source='./speaker-identification/development_set'
test_file= './speaker-identification/development_set_test.txt'
modelpath='./speaker_models/'
file_paths=open(test_file, 'r')
features=np.asarray(())
count=1

# 得到每一个模型文件的路径
# 找到所有以.gmm结尾的文件
gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# 加载每个模型文件
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

# 提取每个说话人的姓名
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

for path in file_paths:
    path=path.strip()
    print(path)
    # 读取声音文件
    full_path = os.path.join(source, path)  # 使用os.path.join正确拼接路径
    print(full_path)  # 打印路径调试
    sr, audio = read(full_path)  # 读取音频文件
    #提取40个维度特征
    vector=extract_features(audio,sr)

    #存储每个模型的结果
    log_likelihood=np.zeros(len(models))
    # 使用每个高斯混合模型来进行评分
    for i in range(len(models)):
        gmm=models[i]
        scores=np.array(gmm.score(vector))
        log_likelihood[i]=scores.sum()
        #GMM假设数据是由多个高斯分布混合生成的。每个GMM对应一个高斯混合成分
        # scores.sum() 反映了当前样本在所有高斯混合成分下的整体匹配程度。
    winner=np.argmax(log_likelihood)
    print("\tdetected as -",speakers[winner])
    time.sleep(1.0)

