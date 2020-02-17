# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:02:19 2020

@author: mayn
"""
import numpy as np
import sys
from sklearn.kernel_ridge import KernelRidge

np.set_printoptions(threshold=sys.maxsize) 

#训练数据读取
trainData = np.load('train_data.npy')
yData = np.load('y_train_data.npy')

#训练数据处理
trainData[:,2:10] = trainData[:,2:10]*trainData[:,10].reshape(-1,1)
trainData[:,11:22] = trainData[:,11:22]*trainData[:,10].reshape(-1,1)
trainData[:,22:30] = trainData[:,22:30]*trainData[:,30].reshape(-1,1)
trainData[:,31:42] = trainData[:,31:42]*trainData[:,30].reshape(-1,1)
yData=np.exp(yData)/1000*3.6
trainData[:,225]=np.exp(trainData[:,225])/1000*3.6
         
import matplotlib.pyplot as plt
for sel in range(225):
    trainData=trainData[np.argsort(trainData[:,sel])]
    plt.title('fea:'+str(sel))
    plt.plot(trainData[:,sel],yData)
    plt.show()
    
trainData=trainData[np.argsort(trainData[:,217])]
anaD=trainData[:,217].tolist()
anaY=yData.tolist()
