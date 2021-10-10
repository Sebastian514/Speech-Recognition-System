
from dtw import dtw
import librosa
from librosa.feature.spectral import mfcc
from matplotlib.pyplot import get
import numpy
from scipy.ndimage.measurements import label
from scipy.signal.windows.windows import triang
from sklearn.model_selection import train_test_split as ts
from numpy.linalg import norm
import os
k = []
label = []
# def get_y_test():
Names=[]
dic = {}
totaldis = {str(i): 0 for i in range(10)}
dirlist = os.listdir("/home/lichengjun/lza/dsp实验/Speech-Recognition-System/recordings")
test_dir = dirlist[:300:1]
test_label = [i[0] for i in dirlist[:300:1]]
test_mfcc =[]
train_dir = dirlist[300::1]
train_label = [i[0] for i in dirlist[300::1]]
train_mfcc = []
for i in test_dir:
    tmp, sr = librosa.load("/home/lichengjun/lza/dsp实验/Speech-Recognition-System/recordings/"+i,sr=None)
    mfccs = librosa.feature.mfcc(y = tmp, sr = sr, n_mfcc= 12, n_fft = 512)
    test_mfcc.append(mfccs)
for i in train_dir:
    tmp, sr = librosa.load("/home/lichengjun/lza/dsp实验/Speech-Recognition-System/recordings/"+i,sr=None)
    mfccs = librosa.feature.mfcc(y = tmp, sr = sr, n_mfcc= 12, n_fft = 512)
    train_mfcc.append(mfccs)
##Compute_dis 
##转置
for i in range(len(train_mfcc)):
    train_mfcc[i] = list(map(list, zip(*train_mfcc[i])))
for j in range(len(test_mfcc)):
    test_mfcc[j] = list(map(list, zip(*test_mfcc[j])))
train_mfcc = numpy.array(train_mfcc)
test_mfcc = numpy.array(test_mfcc)
print(train_mfcc)
print(test_mfcc)
# train_mfcc = numpy.array(train_mfcc)
test_predict = []
dirnum = {}
for i in range(300):
    dic = {}
    dirnum = {}
    print(i)
    for j in range(2700):
        dist, cost, acc_cost, path = dtw(train_mfcc[j], test_mfcc[i], dist=lambda x, y: norm(x - y, ord=1))
        dic[train_label[j]] = dic.get(train_label[j], 0) + dist
        dirnum[train_label[j]] = dirnum.get(train_label[j], 0) + 1
    dic = {k:l/dirnum[k] for k, l in dic.items()}
    print(dic)
    test_predict.append(min(dic, key = dic.get))    
print("accuracy:",[test_predict[i] == test_label[i] for i in range(300)].count(1)/300)

