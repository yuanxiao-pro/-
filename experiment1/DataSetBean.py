#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DataSetBean.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 12:46   xiaoj      1.0         数据集对象
'''

# import lib
class DataSetBean:
      x_train = None
      y_train = None
      x_test = None
      y_test = None

      def __init__(self,x_train, y_train, x_test, y_test):
            self.x_test = x_test
            self.x_train = x_train
            self.y_test = y_test
            self.y_train = y_train
