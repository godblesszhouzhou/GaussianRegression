# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:49:29 2019

@author: joezhoushen
"""
import re
import numpy as np
import math
filename='case_shoutu_20200105_20200112_cleanSpeedCase.txt.trainSvm'

fileLen=200000

divLen=int(fileLen/10000)
data=np.zeros((fileLen,51302),dtype='float')
ydata=np.zeros((fileLen,1),dtype='float')
j=0

with open(filename,'r') as file_to_read:
    while True:
      lines=file_to_read.readline()
      if  not lines:
          break
      if j>=fileLen:
          break
#      if j<1000000:
#          continue;
      if j%divLen==0:
          print(str((j/divLen))+'%')
      d=re.split(r'[\s,\n,:]',lines)
      for i in range(225):
          dt=float(d[2*(i+1)])
          if math.isnan(dt):
              print('Error',j,i)
              dt=0
          if math.isinf(dt):
              print('Error',j,i)
              dt=1
          data[j,i]=dt
      data[j,225]=float(d[226*2])
      ydata[j,0]=float(d[0])
      ########################################
      #训练数据处理
      data[j,2:10] = data[j,2:10]*data[j,10]
      data[j,11:22] = data[j,11:22]*data[j,10]
      data[j,22:30] = data[j,22:30]*data[j,30]
      data[j,31:42] = data[j,31:42]*data[j,30]
      ydata[j,0]=np.exp(ydata[j,0])/1000*3.6
      data[j,225]=np.exp(data[j,225])/1000*3.6
      #mm转km
      data[j,0]=data[j,0]/1000000
      #trainData[:,87]=trainData[:,0]/1000000
      for i in range(112,222,2):
          data[j,i]=data[j,i]/1000000
      data[j,223]=data[j,223]/1000000
      
      #加平方多项式项
      for i in range(226):
          for k in range(226):
              #tmp1=data[:,i]
              #tmp2=data[:,k]
              #trainData=np.concatenate((trainData,tmp1*tmp2),axis=1)
              #trainData = np.column_stack((trainData,tmp1*tmp2))
              data[j,226+i*226+k]=data[j,i]*data[j,k]
      #########################################
      j=j+1
print('100.0%')
np.save('train_data_51076.npy',data)
np.save('y_train_data_51076.npy',ydata)
