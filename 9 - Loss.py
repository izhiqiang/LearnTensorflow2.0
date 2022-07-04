import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

SEED = 23455
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

# 均方误差损失函数
print("----------------------------------------均方误差损失函数----------------------------------------")
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_mean(tf.square(y_ - y))

    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps, w1 is " % epoch)
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
print("----------------------------------------均方误差损失函数----------------------------------------")

# 自定义损失函数
print("----------------------------------------自定义损失函数----------------------------------------")
COST = 99
PROFIT = 1
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps, w1 is " % epoch)
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
print("----------------------------------------自定义损失函数----------------------------------------")

# 交叉熵
print("----------------------------------------交叉熵----------------------------------------")
loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)
print("----------------------------------------交叉熵----------------------------------------")

