#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiment3.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/4 8:12   xiaoj      1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 汉显

'''
梯度下降函数实现
'''
def numerical_gradient(func, x):
      h = 1e-4
      grad = np.zeros_like(x)  # 生成和x形状一样的全0矩阵
      for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = func(x)
            x[idx] = tmp_val - h
            fxh2 = func(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
      return grad


def function_2(x):
      return x[0] ** 2 + x[1] ** 2


def gradient_descent(func, init_x, lr=0.01, step_num=100):
      x = init_x
      x_history = []
      for i in range(step_num):
            x_history.append(x.copy())
            grad = numerical_gradient(func, x)
            x -= lr * grad
      return x,np.array(x_history)


if __name__ == '__main__':
      init_x=np.array([-3.0,4.0])
      x,x_history = gradient_descent(function_2,init_x,lr=0.1,step_num=100)
      print(x_history)

      fig1 = plt.figure()
      ax = Axes3D(fig1)
      ax.scatter(x_history[:, 0], x_history[:, 1], np.array(x_history[:, 0] ** 2 + x_history[:, 1 ** 2]), c='r')
      ax.set_xlabel('x0')
      ax.set_ylabel('01')
      ax.set_title('函数图形')

      plt.xlim([-3.5, 3.5])
      plt.ylim([-4.5, 4.5])
      plt.xlabel('x0')
      plt.ylabel('x1')
      plt.show()
      plt.savefig("result.png")
      print("Finished!")
