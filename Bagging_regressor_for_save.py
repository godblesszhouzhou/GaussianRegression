# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:23:27 2019

@author: joezhoushen
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor 
import numpy as np

X_train=np.load("train_data1_div1.npy")[:,0:-1]
print(X_train.shape)

X_test=np.load("test_data_div1.npy")[:,0:-1]
print(X_test.shape)

y_train=np.load("y_train_data1_div1.npy")
print(y_train.shape)
y_train=y_train.reshape(y_train.shape[0],)

y_test=np.load("y_test_data_div1.npy")
print(y_test.shape)
y_test=y_test.reshape(y_test.shape[0],)

#释放X和y占用的
from sklearn.externals import joblib
bag_clf= BaggingRegressor(DecisionTreeRegressor(max_depth=8,min_samples_split=20,min_samples_leaf =20),n_estimators=2,bootstrap=False,max_samples=0.75,n_jobs =12,verbose=10)
bag_clf.fit(X_train,y_train)
joblib.dump(bag_clf, "train_model_12_bagging.m")

y_train_pred=bag_clf.predict(X_train)
y_train=y_train.reshape(y_train.shape[0],)
print(np.mean(abs(y_train-y_train_pred)/y_train))
y_test_pred=bag_clf.predict(X_test)
y_test=y_test.reshape(y_test.shape[0],)
print(np.mean(abs(y_test-y_test_pred)/y_test))
np.save("y_test_pred.npy",y_test_pred)
np.save("y_train_pred.npy",y_train_pred)

test_len=X_test[:,0]
test_t=test_len/np.exp(y_test)
pre_time=test_len/np.exp(y_test_pred)
print(np.mean(abs(test_t-pre_time)/test_t))
