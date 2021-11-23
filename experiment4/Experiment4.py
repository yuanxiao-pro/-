#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiment4.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/4 8:50   xiaoj      1.0         None
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
# def numerical_gradient(func, x):
#       h = 1e-4
#       grad = np.zeros_like(x)  # 生成和x形状一样的全0矩阵
#       for idx in range(x.size):
#             tmp_val = x[idx]
#             x[idx] = tmp_val + h
#             fxh1 = func(x)
#             x[idx] = tmp_val - h
#             fxh2 = func(x)
#             grad[idx] = (fxh1 - fxh2) / (2 * h)
#             x[idx] = tmp_val
#
#       return grad


def _numerical_gradient_no_batch(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_va1 = x[idx]
        x[idx] = float(tmp_va1) + h
        fxh1 =f(x)

        x[idx] = tmp_va1 -h
        fxh2 = f(x)
        grad[idx] = (fxh1- fxh2)/(2*h)

        x[idx] = tmp_va1
    print("grad:",grad)
    return grad

def numerical_gradient(f,X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f,X)
    else:
        grad = np.zeros_like(X)
        for idx,x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f,x)
        return grad

def function_2(x):
      if x.ndim == 1:
            return np.sum(x ** 3)
      else:
            return np.sum(x ** 3, axis=1)


def gradient_descent(func, init_x, lr=0.01, step_num=100):
      x = init_x
      x_history = []
      for i in range(step_num):
            x_history.append(x.copy())
            grad = numerical_gradient(func, x)
            x -= lr * grad
      return x,np.array(x_history)

def draw3D():
      x1 = np.arange(-10, 10, 1)  # 为了绘制函数的原图像
      x2 = np.arange(-10, 10, 1)
      x1, x2 = np.meshgrid(x1, x2)
      z = x1 ** 3 + x2 ** 3
      fig = plt.figure()
      ax = Axes3D(fig)
      ax.plot_surface(x1, x2, z)  # 绘制3D坐标系中的函数图像
      # ax.scatter(w1, w2, targetFunction([w1, w2]), s=50, c='red')  # 绘制已经找到的极值点
      ax.legend()  # 使坐标系为网格状
      plt.show()  # 显示


def drawVerImg(X,Y,grad):
      plt.figure()
      plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
      plt.xlim([-2, 2])
      plt.ylim([-2, 2])
      plt.xlabel('x0')
      plt.ylabel('x1')
      plt.grid()
      plt.legend()
      plt.draw()
      plt.show()
      plt.savefig("quiver.png")
      pass


if __name__ == '__main__':
      init_x=np.array([-2.0,2.0])
      x,x_history = gradient_descent(function_2,init_x,lr=0.1,step_num=100)
      # 绘制函数图像
      draw3D()

      # x0 = np.arange(-2, 2.5, 0.25)
      # x1 = np.arange(-2, 2.5, 0.25)
      # X, Y = np.meshgrid(x0, x1)
      # X = X.flatten()
      # Y = Y.flatten()
      #
      # grad=numerical_gradient(function_2,np.array([X,Y]))
      #
      # 绘制梯度矢量图
      # drawVerImg(X,Y,grad)

      print("Finished!")

