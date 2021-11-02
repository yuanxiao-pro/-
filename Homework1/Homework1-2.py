#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Homework1-1.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/21 15:50   xiaoj      1.0         None
'''

# import lib
import math

import numpy as np

from functools import reduce

from operator import mul

def re():
      n, m = map(int, input().split())
      print(math.factorial(n) // (math.factorial(m) * math.factorial(n - m)))



if __name__ == '__main__':
      re()