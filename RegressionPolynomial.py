# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:31:47 2020

@author: mayn
"""

import numpy as np
import sys
from sklearn.linear_model import Ridge

trainData = np.load('train_data_51076.npy')
yData = np.load('y_train_data_51076.npy')
print('Load finish!')
reg=Ridge(0.01)
reg.fit(trainData,yData)
print('Train finish!')
#测试数据读取
testData = np.load('test_data.npy')
ytestData = np.load('y_test_data.npy')

res=reg.predict(testData)/3.6*1000

ypred=np.log(res)
estTestData=np.log(testData[:,225].reshape(-1,1)/3.6*1000)


#结果比较
print(np.mean(np.abs(estTestData-ytestData)))
print(np.mean(np.abs(ypred-ytestData)))             