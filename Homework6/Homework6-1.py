#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework6-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/3 11:22   xiaoj      1.0
从mnist训练集中按现有顺序分出六个万份数据，编写程序，统计每万份数据中与测试集相同位置上具有相同数字的个数。
'''

# import lib
import pickle
import numpy as np
from experiment1.DataSetBean import DataSetBean
import tensorflow as tf
import utils.DataUtils as dataUtils
import datetime



'''
加载mnist数据集
'''
def loadMnist(path):
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = dataUtils.load_data_gz(path)
      return x_train_gz, y_train_gz, x_test_gz, y_test_gz

'''
加载pickle数据集
'''
def loadPickle(path):
      bean = None
      with open(path, 'rb') as f:
            # ((x_train, y_train), (x_test, y_test)) = pickle.load(f, encoding="latin-1") # 加载数据
            paramList = pickle.load(f, encoding="latin-1")  # 加载数据
            bean = paramList
      return bean

def sigmoid(x):
      indices_pos = np.nonzero(x >= 0)
      indices_neg = np.nonzero(x < 0)
      y = np.zeros_like(x)
      y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
      y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
      return y


'''
使用批处理
'''
def batch():
      batch_size = 100
      accuracy_cnt = 0
      network=loadPickle("../data/sample_weight.pkl")
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = loadMnist('../data/')
      resultSet = []
      for j in range(1,6):

            for i in range(0, len(x_test_gz[(j-1)*10000:j*10000]), batch_size):
                  x_batch = x_test_gz[i:i + batch_size]
                  y_batch = predict(network, x_batch)
                  p = np.argmax(y_batch, axis=1) # p是预测出来是数字
                  # print(p)
                  accuracy_cnt += np.sum(p == y_test_gz[i:i + batch_size])
            resultSet.append(accuracy_cnt)
      # return "Accuracy:" + str(float(accuracy_cnt) / len(x_test_gz))
      return resultSet

'''
做分类
'''
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

if __name__ == '__main__':
      resultSet=batch()
      print(resultSet)