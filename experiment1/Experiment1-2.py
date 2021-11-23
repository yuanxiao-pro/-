#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiment1-2.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 18:40   xiaoj      1.0         基于pickle和批处理提高模型计算速度
'''

# import lib
import pickle
import numpy as np
from experiment1.DataSetBean import DataSetBean
import tensorflow as tf
import utils.DataUtils as dataUtils
import datetime
'''
使用pickle的方式加载数据集
@:param 
@:returns 
'''
def load(path):
      bean = None
      with open(path, 'rb') as f:
            # ((x_train, y_train), (x_test, y_test)) = pickle.load(f, encoding="latin-1") # 加载数据
            paramList = pickle.load(f, encoding="latin-1") # 加载数据
            bean = paramList
      return bean


'''
定义均方误差函数
'''
def mean_squared_error(y,t):
      return 0.5 * np.sum((y-t)**2)

def sigmoid(x):
      indices_pos = np.nonzero(x >= 0)
      indices_neg = np.nonzero(x < 0)
      y = np.zeros_like(x)
      y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
      y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
      return y

def predict(network, x):
      W1, W2, W3 = network['W1'], network['W2'], network['W3']
      b1, b2, b3 = network['b1'], network['b2'], network['b3']
      a1 = np.dot(x,W1)+b1
      z1 =sigmoid(a1)
      a2 = np.dot(z1, W2) + b2
      z2 = sigmoid(a2)
      a3 = np.dot(z2, W3) + b3
      y = sigmoid(a3)
      return y

'''
使用pickle的方式加载数据集
@:param 
@:returns 
'''
def load(path):
      bean = None
      with open(path, 'rb') as f:
            # ((x_train, y_train), (x_test, y_test)) = pickle.load(f, encoding="latin-1") # 加载数据
            paramList = pickle.load(f, encoding="latin-1") # 加载数据
            bean = paramList
      return bean


'''
加载数据,x是图像,y是标签
@:param mnist数据集
@:returns (x_train, y_train), (x_test, y_test) ( 训练图像, 训练标签)， ( 测试图像，测试标签) DataSetBean对象
'''
def get_data(mnist):
      x_train_1=[]
      y_train_1=[]
      x_test_1=[]
      y_test_1=[]
      # ( 训练图像, 训练标签)， ( 测试图像，测试标签)
      (x_train, y_train), (x_test, y_test) = mnist.load_data() # 加载数据

      x_train_1 = [0 for i in range(len(x_train))] #range(0,len(x_train))
      y_train_1 = []
      x_test_1 = []
      y_test_1 = []

      # x_train.flatten(True)
      print("x_train.shape-1",x_train.shape)
      for i in range(len(x_train)):
            x_train_1[i]=x_train[i].reshape(784,)
      print("x_train.shape-2",x_train.shape)

      x_train, x_test = x_train / 255.0, x_test / 255.0 #样本转为浮点数

      bean = DataSetBean(x_train, y_train, x_test, y_test)
      return bean

'''
使用批处理
'''
def batch():
      batch_size = 100
      accuracy_cnt = 0
      network=load("../data/sample_weight.pkl")
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = dataUtils.load_data_gz('../data/')
      for i in range(0, len(x_test_gz), batch_size):
            x_batch = x_test_gz[i:i + batch_size]
            y_batch = predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            print(p)
            accuracy_cnt += np.sum(p == y_test_gz[i:i + batch_size])
      return "Accuracy:" + str(float(accuracy_cnt) / len(x_test_gz))

'''
不使用批处理
'''
def nonBatch():
      accuracy_cnt = 0
      network = load("../data/sample_weight.pkl")
      # bean = get_data(mnist)
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = dataUtils.load_data_gz('../data/')
      for i in range(0, len(x_test_gz)):
            x = x_test_gz[i]
            # print("x_batch.shape",x_batch.shape)
            # np.asfortranarray(x_batch)
            A = np.mat(x)
            # print(A.shape)
            y = predict(network, A)

            p = np.argmax(y)
            if p == y_test_gz[i]:
                  accuracy_cnt += 1
      return "Accuracy:" + str(float(accuracy_cnt) / len(x_test_gz))
if __name__ == '__main__':
      mnist = tf.keras.datasets.mnist
      # 获取时间
      print(datetime.datetime.now())
      batchResult = batch()
      print(datetime.datetime.now())
      print(batchResult)

      print("----------------------------------------------")

      print(datetime.datetime.now())
      nonBatchResult = nonBatch()
      print(datetime.datetime.now())
      print(nonBatchResult)

      print("success!")