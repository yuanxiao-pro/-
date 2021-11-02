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
import matplotlib.pyplot as plt
import numpy as np

def drawSin():
      # line
      x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
      # 定义余弦函数正弦函数
      c, s = np.cos(x), np.sin(x)
      plt.figure(1)
      # 画图，以x为横坐标，以c为纵坐标
      plt.plot(x, c, color='coral', linestyle="--", label="COS", alpha=0.5)
      # plt.scatter(x, c, color='coral')
      plt.plot(x, s, color='blue',linestyle="-", label="SIN")
      # 增加标题
      plt.title("COS & SIN")
      ax = plt.gca()

      ax.xaxis.set_ticks_position("bottom")
      ax.yaxis.set_ticks_position("left")

      plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
      for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(16)
      label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))
      # 图例显示
      plt.legend(loc="upper left")
      # 显示网格
      plt.grid()
      plt.show()

if __name__ == '__main__':
    drawSin()