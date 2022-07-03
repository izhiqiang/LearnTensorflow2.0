import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

# 数据集读入
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print("x_data from datasets: \n", x_data)
print("y_data from datasets: \n", y_data)

# 数据集显示
# x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# pd.set_option('display.unicode.east_asian_width', True)
# print("x_data add index: \n", x_data)
#
# x_data['类别'] = y_data
# print("x_data add a colum: \n", x_data)

# 数据集乱序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 从数据集分离出训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换 x 的数据类型，否则后面矩阵相乘时会因为数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 组成[输入特征， 标签]对， 每次喂入一小撮(batch)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络总所有可训练参数
w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为 0.1
train_loss_results = []  # 将每轮的 loss 记录在此列表中，为后面画 loss 曲线提供数据
test_acc = []  # 将每轮的 acc 记录在此列表中，为后面画 acc 曲线提供数据
epoch = 500  # 循环 500 轮
loss_all = 0  # 每轮 4 个 step，loss_all 记录四个 step 生成的四个 loss 的和

# 嵌套循环迭代，with结果更新参数，显示当前loss
for epoch in range(epoch):
    # 训练部分
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # 前向传播过程计算y
            y = tf.matmul(x_train, w) + b
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            # 计算总loss
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        # 计算 loss 对各个参数的梯度
        grads = tape.gradient(loss, [w, b])  # 采用均方误差损失函数
        # 实现梯度更新 w = w - lr * w_grad    b = b - lr * b_grad
        w.assign_sub(lr * grads[0])  # 参数 w 自更新
        b.assign_sub(lr * grads[1])  # 参数 b 自更新
    # 打印 epoch 的 loss 信息
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0

    # 测试部分
    # 计算当前参数前向传播后的准确率，显示当前 acc
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w) + b  # y 为预测结果
        y = tf.nn.softmax(y)  # 使 y 符合概率分布
        pred = tf.argmax(y, axis=1)  # 返回 y 中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)  # 调整数据类型与标签一致
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # 若分类正确，则 correct = 1，反之为 0
        correct = tf.reduce_sum(correct)  # 将每个 batch 中的 correct 数加起来
        total_correct += int(correct)  # 将所有 batch 中的 correct 数加起来
        total_number += x_test.shape[0]
        acc = total_correct / total_number
        test_acc.append(acc)
        print("Test_acc:", acc)
        print("-----------------------------")

# acc / loss 可视化
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()