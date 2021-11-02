#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiment1-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 8:44   xiaoj      1.0         None
'''

# import lib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

from experiment1.DataSetBean import DataSetBean

'''
加载数据,x是图像,y是标签
@:param mnist数据集
@:returns (x_train, y_train), (x_test, y_test) ( 训练图像, 训练标签)， ( 测试图像，测试标签) DataSetBean对象
'''
def load(mnist):
      # ( 训练图像, 训练标签)， ( 测试图像，测试标签)
      (x_train, y_train), (x_test, y_test) = mnist.load_data() # 加载数据
      x_train, x_test = x_train / 255.0, x_test / 255.0 #样本转为浮点数
      bean = DataSetBean(x_train, y_train, x_test, y_test)
      return bean
'''
训练模型
@:param DataSetBean对象
@:return model 训练好的模型对象
'''
def train(bean):
      # tf.keras.models.Sequential是tf的顺序模型
      model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'), # 激活函数使用ReLU
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax') # 激活函数使用softmax
      ])

      model.compile(optimizer='adam',
                    # loss='sparse_categorical_crossentropy', # 交叉熵作为损失函数
                    loss="mse", # 均方差作为损失函数
                    metrics=['accuracy'])
      # 将数据与模型拟合
      model.fit(bean.x_train, bean.y_train, epochs=5)
      return model

'''
验证模型
@:param model 训练好的模型对象, bean 数据集对象
print: loss accuracy
'''
def valid(model,bean):
      model.evaluate(bean.x_test, bean.y_test, verbose=2)

'''
定义均方误差函数
'''
def mean_squared_error(y,t):
      return 0.5 * np.sum((y-t)**2)


if __name__ == '__main__':
      mnist = tf.keras.datasets.mnist

      bean = load(mnist)
      model=train(bean)
      valid(model,bean)
      print("success!")