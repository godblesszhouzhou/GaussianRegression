# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:07:49 2020

@author: mayn
"""
import numpy as np
import sys
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
         

trainData=trainData[np.argsort(trainData[:,217])]
xx=trainData[:,223].reshape(-1,1)
xx=xx/1000000;
xxx=xx.tolist()
#X=xx
X=np.concatenate((xx,xx*xx,xx*xx*xx),axis=1)

from sklearn import linear_model
#模型训练
reg = linear_model.LinearRegression()
reg.fit(X,yData)
yPre=reg.predict(X)
print(np.mean((yPre-yData)*(yPre-yData)))


#测试
'''
testorig=np.linspace(1.0, 75.0, num=75).reshape(-1,1);

xtest=np.concatenate((testorig,testorig*testorig),axis=1)
import matplotlib.pyplot as plt
testPre=reg.predict(xtest)
plt.plot(testorig,testPre)
plt.show()
'''