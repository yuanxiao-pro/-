#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework5-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/27 10:38   xiaoj      1.0        使用0~9生成10个随机数，并将之转换为one-hot编码.
'''

# import lib
import random
import numpy as np

'''
@:param 0~9的10个随机数
@:return one_hot编码
'''
def getOneHotCode(vector):
      # 设置类别的数量
      num_classes = 10
      print(np.eye(num_classes)[vector])
      return np.eye(num_classes)[vector]



if __name__ == '__main__':
      # 生成0~9随机数
      vec = np.random.randint(0, 9, 10);
      getOneHotCode(vec)
      print()
