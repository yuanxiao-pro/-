#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LinearModelSolution.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/29 15:44   xiaoj      1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
'''
约束条件定义 w取值范围
'''
def constraint1(w1, w2, w3):
      if w1 + w2 + w3 == 1 and w1 >= 0 and w2 >= 0 and w3 >= 0:
            return True
      return False


'''
约束条件定义 w和Epsilon取值范围
'''
def constraint2(w1, w2, w3, epsilon):
      if w1 == 0 or w2 == 0 or w3 == 0 or epsilon == 0:
            return False
      f1 = (abs(float(w3 / w1) - 8) <= epsilon)
      f2 = (abs(float(w3 / w2) - 2) <= epsilon)
      f3 = (abs(float(w2 / w1) - 5) <= epsilon)

      if f1 and f2 and f3:
            return True
      return False


'''
线性公式求解
'''
def formula(w1, w2, w3, epsilon):
      # eArray = [[]]
      c1 = constraint1(w1, w2, w3)
      c2 = constraint2(w1, w2, w3, epsilon)
      if c1 and c2:
            print(w1, w2, w3, epsilon)
            return w1, w2, w3, epsilon
      return 0,0,0,0


if __name__ == '__main__':
      w1=0.1
      w2=0.1
      w3=0.1
      epsilon = 0.1
      stop = 1
      step = 0.01
      epsilonStop = 8
      epsilonStep = 0.01

      eArray = []
      param = []
      print("w1,w2,w3,epsilon")
      for w1 in np.arange(0,stop,step):
            for w2 in np.arange(0, stop, step):
                  for w3 in np.arange(0, stop, step):
                        for epsilon in np.arange(0, epsilonStop, epsilonStep):
                              r1, r2, r3, epl=formula(w1,w2,w3,epsilon)
                              if  r1 != 0 and r2 != 0 and r3 != 0 and epl != 0:
                                    eArray.append(epl)
                                    param.append((r1, r2, r3))
      print(np.argmin(eArray))
      print(np.min(eArray))
      print(param[np.argmin(eArray)])
      print("Finish")
