import tensorflow as tf
import numpy as np

# 下面是一个“标量”（或称“0 秩”张量）。标量包含单个值，但没有“轴”。
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# “向量”（或称“1 秩”张量）就像一个值列表。向量有 1 个轴：
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# “矩阵”（或称“2 秩”张量）有 2 个轴：
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

# 张量的轴可能更多，下面是一个包含 3 个轴的张量：
# 轴 (有时候被称为 "维度")
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])
print(rank_3_tensor)

# 张量转化为 Numpy 数组
np.array(rank_2_tensor)
rank_2_tensor.numpy()

# 张量可以执行基本数学运算，包括加法、逐元素乘法和矩阵乘法。
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])  # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

print(a + b, "\n")  # 元素和
print(a * b, "\n")  # 元素乘
print(a @ b, "\n")  # 矩阵乘积

# 各种运算 (op) 都可以使用张量。
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# 查询最大值
print(tf.reduce_max(c))
# 查询最大值的索引
print(tf.argmax(c))
# 计算 softmax
print(tf.nn.softmax(c))

# 4 秩张量，形状：[3, 2, 4, 5]
rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# 索引
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

# 使用标量编制索引会移除轴
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

# 使用切片编制索引会保留轴
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())