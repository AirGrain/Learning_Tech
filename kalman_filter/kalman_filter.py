# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:38:34 2019

@author: Throp
"""

import numpy as np
import matplotlib.pyplot as plt

Z = np.arange(1,101,1)  # 观测值
noise = np.random.randn(1,100) # 均值为0，方差为1的高斯噪声（线性kalman滤波的基本假设）
Z = Z+noise

X = np.mat([[0],[0]]) #状态
P = np.mat([[1,0],[0,1]]) # 状态协方差矩阵
F = np.mat([[1,1],[0,1]]) # 状态转移矩阵
Q = np.mat([[0.0001,0],[0,0.0001]]) # 状态转移协方差矩阵
H = np.mat([1,0]) # 观测矩阵
R = 1 # 观测噪声方差

for i in range(0,100):
    X_ = np.dot(F,X)
    P_ = np.dot(F.dot(P),F.T) + Q
    K  = np.dot(P_.dot(H.T),(np.dot(H.dot(P_),H.T)+R).I)
    X  = X_ + np.dot(K,(Z[0][i]-H.dot(X_)))
    P  = np.dot(np.eye(2)-np.dot(K,H),P_)
    
    plt.plot(X[0][0],X[1][0],'b*') # 横轴表示位置，纵轴表示速度
    


