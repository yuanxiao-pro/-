#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework5-2.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/31 18:52   xiaoj      1.0         None
'''

# import lib
import numpy as np
import tensorflow as tf
from Homework5.DataSetBean import DataSetBean
import pickle
import os
import gzip
import utils.DataUtils as dataUtils
def sigmoid(x):
      # y = 1 / (1 + np.exp(-x))
      # return y
      indices_pos = np.nonzero(x >= 0)
      indices_neg = np.nonzero(x < 0)

      y = np.zeros_like(x)
      y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
      y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))

      return y

def predict(network, x):
      W1, W2, W3 = network['W1'], network['W2'], network['W3']
      b1, b2, b3 = network['b1'], network['b2'], network['b3']
      print(W1.shape,W2.shape,W3.shape)
      # print("before",W1.shape)
      # np.transpose([W1])
      # print("after",W1.shape)
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

def batch(mnist):
      batch_size = 100
      accuracy_cnt = 0
      network=load("../data/sample_weight.pkl")
      # bean = get_data(mnist)
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = dataUtils.load_data_gz('../data/')
      # x = bean.x_test
      # t = bean.y_test
      # print(len(x))
      for i in range(0, len(x_test_gz), batch_size):
            x_batch = x_test_gz[i:i + batch_size]
            print("x_batch.shape",x_batch.shape)

            y_batch = predict(network, x_batch)

            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == y_test_gz[i:i + batch_size])
            return "Accuracy:" + str(float(accuracy_cnt) / len(x_test_gz))

if __name__ == '__main__':
      mnist = tf.keras.datasets.mnist
      result = batch(mnist)
      print(result)