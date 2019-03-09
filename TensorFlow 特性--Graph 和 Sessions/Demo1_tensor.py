import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# 使用Numpy生成假数据(phony data),总共100个点.

x_data = np.float32(np.random.rand(2, 100))  # 随机输入
print(x_data)
y_data = np.dot([0.100, 0.200], x_data) + 0.300  #输出的y为[[]]的list
print(y_data)

# 构造一个线性模型, 下面开始反演
# 实际问题中如果没有确切的物理关系,很难知道是否是线性模型, 也很难知道解在哪个范围

b = tf.Variable(tf.zeros([1]))
print(b)
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
print(W)
y = tf.matmul(W, x_data) + b   # y is synthetic data

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)   #learning_rate,就是步长
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图(graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 51):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b))
W = sess.run(W)
b = sess.run(b)

sess.close()

y_pred = np.dot(W, x_data) + b

print('----------------')
print(y_pred[0])
# 得到最佳拟合结果 W: [[0.100 0.200]]\, b: [0.300]

# 画图展示
plt.figure()
plt.title('Comparation of y_syn with y_data')
plt.plot(y_data,color='green',label='data')
plt.plot(y_pred[0],color='red',label='synthetic')
plt.legend()
plt.xlabel('sequence')
plt.ylabel('Value of variable Y')
plt.savefig('Comparation.jpg', dpi=150)
plt.show()



