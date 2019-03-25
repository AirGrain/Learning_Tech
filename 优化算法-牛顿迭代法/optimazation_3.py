from phi import phi as func
import numpy as np

e = 0.0005
n = 30
# 初始模型
m1 = -1
m2 = 1

gk = np.ones((2,1),float)
Gk = np.zeros((2,2),float)

gk = np.mat(gk)
Gk = np.mat(Gk)

m1stp = 0.5 # m1差分计算导数的步长
m2stp = 0.5 # m2差分计算导数的步长

k = 1
while ((k < n) or np.sqrt(np.power(gk[0,0],2)+np.power(gk[1,0],2))) > e:
    #向前差分计算一阶导
    gk[0,0] = 1/m1stp*(func(m1+m1stp,m2)-func(m1,m2))
    gk[1,0] = 1/m2stp*(func(m1,m2+m2stp)-func(m1,m2))
    
    #向前差分计算海森矩阵,注意：以函数为二阶导连续为前提
    Gk[0,0] = 1/m1stp*(func(m1+m1stp,m2)-2*func(m1,m2)+func(m1-m1stp,m2))
    Gk[0,1] = 1/(m1stp*m2stp)*(func(m1+m1stp,m2+m2stp)-func(m1,m2+m2stp)-func(m1+m1stp,m2)+func(m1,m2))
    Gk[1,0] = Gk[0,1]
    Gk[1,1] = 1/m2stp*(func(m1,m2+m2stp)-2*func(m1,m2)+func(m1,m2-m2stp))

    dk = Gk.I*gk

    #修正模型
    m1 = m1-dk[0,0]
    m2 = m2-dk[1,0]

    k = k+1
    print(m1,m2)
