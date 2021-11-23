#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   QuiverTest.py    
@Contact :   1665219552@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/11 8:37   xiaoj      1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return np.sum(x**3)#return x[0]**2 + x[1]**2

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
    print("grad:"+str(grad))
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
        return np.sum(x **3)
    else:
        return np.sum(x**3,axis=1)


def tangent_line(f,x):
    d = numerical_gradient(f,x)
    print(d)
    y = f(x) - d * x
    return lambda t : d * t + y

print(_numerical_gradient_no_batch(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([[3.0,4.0],[0.0,2.0],[3.0,0.0]])))
if __name__ =='__main__':
    x0= np.arange(-2,2.5,0.25)
    x1=np.arange(-2,2.5,0.25)
    X,Y= np.meshgrid(x0,x1)
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2,np.array([X,Y]))

    plt.figure()
    plt.quiver(X,Y,-grad[0],-grad[1],angles="xy",color="#666666")
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()