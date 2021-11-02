#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework4-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/27 8:43   xiaoj      1.0         给出softmax定义（防溢出），并给出输入输出实例证明有效性。
'''

# import lib
import numpy as np

'''
@:param vector 输入一个向量
@:return soft_max值 
'''
def soft_max(vector):
      c = np.max(vector)
      exp_a = np.exp(vector - c)
      sum_exp_a = np.sum(exp_a)
      return exp_a / sum_exp_a


if __name__ == '__main__':
      vec = np.array([0,1000,-10])
      print(soft_max(vec))
