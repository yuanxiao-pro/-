#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework4-2.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/27 8:52   xiaoj      1.0         绘制三种激活函数的图像
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt
import math as math
'''
绘制三种激活函数的图像
'''


def drawActivationFunction(vector):
      return


'''
绘制sigmoid图像
'''
def drawSigmoidImg(x):
      y = 1 / (1 + np.exp(-x))
      plt.yticks(np.linspace(0, 1, 5, endpoint=True))
      plt.plot(x, y, color='coral', linestyle="-", label="sigmoid", alpha=0.5)
      plt.title("sigmoid")
      ax = plt.gca()  # 获取到当前坐标轴信息
      ax.xaxis.set_ticks_position('bottom')  # 将X坐标轴移到上面
      plt.show()
      return y

'''
绘制soft_max函数图像
'''
def drawSoftMax(vector):
      c = np.max(vector)
      exp_a = np.exp(vector - c)
      sum_exp_a = np.sum(exp_a)
      y = exp_a / sum_exp_a

      plt.yticks(np.linspace(0, 1, 5, endpoint=True))
      plt.plot(x, y, color='coral', linestyle="-", label="soft_max", alpha=0.5)
      plt.title("soft_max")
      ax = plt.gca()  # 获取到当前坐标轴信息
      ax.xaxis.set_ticks_position('bottom')  # 将X坐标轴移到上面
      plt.show()

'''
ReLU函数图像
'''
def drawReLUImg(vector):
      y = np.maximum(0,vector)
      plt.yticks(np.linspace(0, 100, 5, endpoint=True))
      plt.plot(x, y, color='coral', linestyle="-", label="soft_max", alpha=0.5)
      plt.title("soft_max")
      ax = plt.gca()  # 获取到当前坐标轴信息
      ax.xaxis.set_ticks_position('bottom')  # 将X坐标轴移到上面
      plt.show()


if __name__ == '__main__':
      x = np.linspace(-10,10,num=20)
      drawSigmoidImg(x)
      drawSoftMax(x)
      drawReLUImg(x)
      print()
