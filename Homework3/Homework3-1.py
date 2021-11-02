#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework3-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/4 10:26   xiaoj      1.0         使用课上讲的单层感知机组成多层感知机实现 异或门 函数定义
'''

# import lib
import numpy as np


def XOR(x1, x2):
      s1 = NAND(x1, x2) # 第一层
      s2 = OR(x1, x2)
      y = AND(s1, s2)
      return y


def AND(x1, x2):
      x = np.array([x1, x2])
      w = np.array([0.5, 0.5])
      b = -0.7  # set offset
      tmp = np.sum(w * x) + b
      if tmp <= 0:
            return 0
      else:
            return 1

# 与非门
def NAND(x1, x2):
      x = np.array([x1, x2])
      w = np.array([-0.5, -0.5])
      b = 0.7
      tmp = np.sum(w * x) + b
      if tmp <= 0:
            return 0
      else:
            return 1


def OR(x1, x2):
      x = np.array([x1, x2])
      w = np.array([0.5, 0.5])
      b = -0.2  # set offset
      tmp = np.sum(w * x) + b
      if tmp <= 0:
            return 0
      else:
            return 1

if __name__ == '__main__':
      print(XOR(0, 0))
      print(XOR(1, 0))
      print(XOR(0, 1))
      print(XOR(1, 1))