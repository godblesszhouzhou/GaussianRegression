# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:02:19 2020

@author: mayn
"""
import numpy as np
import sys
from sklearn.svm import SVR

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

'''
from sklearn import linear_model
#模型训练
reg = SVR(kernel='linear')
reg.fit(trainData,yData)

#测试数据读取
testData = np.load('train_data.npy')
ytestData = np.load('y_train_data.npy')

#测试数据处理
testData[:,2:10] = testData[:,2:10]*testData[:,10].reshape(-1,1)
testData[:,11:22] = testData[:,11:22]*testData[:,10].reshape(-1,1)
testData[:,22:30] = testData[:,22:30]*testData[:,30].reshape(-1,1)
testData[:,31:42] = testData[:,31:42]*testData[:,30].reshape(-1,1)
testData[:,225]=np.exp(testData[:,225])/1000*3.6
ypred=np.log(reg.predict(testData)/3.6*1000)
estTestData=np.log(testData[:,225].reshape(-1,1)/3.6*1000)

#结果比较
print(np.mean(np.abs(estTestData-ytestData)))
print(np.mean(np.abs(ypred-ytestData)))
'''