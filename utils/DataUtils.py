#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DataUtils.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 8:53   xiaoj      1.0         None
'''

# 方法2：gz格式数据集的加载
import os
import gzip
import numpy as np

# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def load_data_gz(data_folder):
      files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

      paths = []
      for fname in files:
            paths.append(os.path.join(data_folder, fname))

      # 读取每个文件夹的数据
      with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

      with gzip.open(paths[1], 'rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 784)

      with gzip.open(paths[2], 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

      with gzip.open(paths[3], 'rb') as imgpath:
            x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 784)

      return x_train, y_train, x_test, y_test

if __name__ == '__main__':
      # 调用load_data_gz函数加载数据集
      data_folder = '../data/'
      x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)

      # 查看数据集的形状
      print('x_train_gz:{}'.format(x_train_gz.shape))
      print('y_train_gz:{}'.format(y_train_gz.shape))
      print('x_test_gz:{}'.format(x_test_gz.shape))
      print('y_test_gz:{}'.format(y_test_gz.shape))
