#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework2-2.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/21 16:39   xiaoj      1.0         None
'''

# import lib
import numpy as np

def AND(x1, x2):

    x = np.array([x1, x2])

    w = np.array([0.5, 0.5])

    b = -0.7

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1


def NAND(x1, x2):
      x = np.array([x1, x2])
      w = np.array([0.5, 0.5])
      b = -0.7
      tmp = np.sum(w * x) + b

      if tmp <= 0:
            return 1
      else:
            return 0

def OR(x1, x2):
      if(x1 ==1 or x2 == 1 or (x1 ==1 and x2 == 1)):
            return 1
      return 0

if __name__ == '__main__':

    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:

        y = AND(xs[0], xs[1])

        # print(str(xs) + " -> " + str(y))

    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
          y = OR(xs[0], xs[1])

          print(str(xs) + " -> " + str(y))
