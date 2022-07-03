import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def cast_reduce_min_max():
    x = tf.constant([1., 2., 3.],
                    dtype=tf.float64)
    print(x)

    y = tf.cast(x, tf.int32)
    print(y)

    print(tf.reduce_min(y))
    print(tf.reduce_max(y))


def reduce_mean_sum():
    x = tf.constant([[1, 2, 3, ],
                     [2, 2, 3]])
    print(x)

    print(tf.reduce_mean(x))

    print(tf.reduce_sum(x, axis=1))


def tf_variable():
    w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
    print(w)


def math_function():
    a = tf.ones([1, 3])
    b = tf.fill([1, 3], 3.)

    print(a)
    print(b)

    print(tf.add(a, b))  # 加法
    print(tf.subtract(a, b))  # 剑法
    print(tf.multiply(a, b))  # 乘法
    print(tf.divide(a, b))  # 除法

    c = tf.fill([1, 2], 3.)
    print(c)

    print(tf.square(c))  # 平方
    print(tf.sqrt(c))  # 开方
    print(tf.pow(c, 3))  # n 次方

    d = tf.ones([3, 2])
    e = tf.fill([2, 3], 3.)
    print(tf.matmul(d, e))  # 矩阵乘
