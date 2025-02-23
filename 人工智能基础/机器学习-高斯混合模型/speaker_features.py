import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing



def calculate_delta(array):
    rows,cols=array.shape
    deltas=np.zeros((rows,cols))
    N=2
    for i in range(rows):
        index=[]
        j=1
        while j<=N:
            if i-j<0:
                first=0
            else:
                first=i-j
            if i+j>rows-1:
                second=rows-1
            else:
                second=i+j
            index.append((second,first))
            j+=1
        deltas[i]=(array[index[0][0]]-array[index[0][1]]+(2*(array[index[0][1]]-array[index[1][1]])))/10
    return deltas

def extract_features(audio,rate):
    mfcc_feat = mfcc.mfcc(audio,rate,0.025,0.01,20,appendEnergy=True)
    #返回一个二维数组，每行是一个帧，每列是一个特征数
    mfcc_feat = preprocessing.scale(mfcc_feat)
    #mfcc_feat 中的每一列都会被独立地标准化，使每一列的均值为 0，方差为 1
    delta=calculate_delta(mfcc_feat)
    combined=np.hstack((delta,mfcc_feat))
    return combined