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
trainData = np.load('train_data.npy')[0:150000]
yData = np.load('y_train_data.npy')[0:150000]

#训练数据处理
trainData[:,2:10] = trainData[:,2:10]*trainData[:,10].reshape(-1,1)
trainData[:,11:22] = trainData[:,11:22]*trainData[:,10].reshape(-1,1)
trainData[:,22:30] = trainData[:,22:30]*trainData[:,30].reshape(-1,1)
trainData[:,31:42] = trainData[:,31:42]*trainData[:,30].reshape(-1,1)
yData=np.exp(yData)/1000*3.6
trainData[:,225]=np.exp(trainData[:,225])/1000*3.6

#mm转km
trainData[:,0]=trainData[:,0]/1000000
#trainData[:,87]=trainData[:,0]/1000000
for i in range(112,222,2):
         trainData[:,i]=trainData[:,i]/1000000
trainData[:,223]=trainData[:,223]/1000000


for i in range(226):
    tmp=trainData[:,i].reshape(-1,1)
    trainData=np.concatenate((trainData,tmp*tmp),axis=1)

#添加新数据


from sklearn import linear_model
from sklearn.linear_model import Ridge
#模型训练
#reg = linear_model.LinearRegression()
reg=Ridge(0.1)
reg.fit(trainData,yData)

#测试数据读取
testData = np.load('train_data.npy')[190000:200000]
ytestData = np.load('y_train_data.npy')[190000:200000]

#测试数据处理
testData[:,2:10] = testData[:,2:10]*testData[:,10].reshape(-1,1)
testData[:,11:22] = testData[:,11:22]*testData[:,10].reshape(-1,1)
testData[:,22:30] = testData[:,22:30]*testData[:,30].reshape(-1,1)
testData[:,31:42] = testData[:,31:42]*testData[:,30].reshape(-1,1)
testData[:,225]=np.exp(testData[:,225])/1000*3.6
        
#mm转km
testData[:,0]=testData[:,0]/1000000
#trainData[:,87]=trainData[:,0]/1000000
for i in range(112,222,2):
         testData[:,i]=testData[:,i]/1000000
testData[:,223]=testData[:,223]/1000000 
        
for i in range(226):
    tmp=testData[:,i].reshape(-1,1)
    testData=np.concatenate((testData,tmp*tmp),axis=1)

ypred=np.log(reg.predict(testData)/3.6*1000)
estTestData=np.log(testData[:,225].reshape(-1,1)/3.6*1000)


#结果比较
print(np.mean(np.abs(estTestData-ytestData)))
print(np.mean(np.abs(ypred-ytestData)))
