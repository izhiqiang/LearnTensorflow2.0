import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


def tf_where():
    a = tf.constant([1, 2, 3, 1, 1])
    b = tf.constant([0, 1, 3, 4, 5])
    c = tf.where(tf.greater(a, b), a, b)  # 等价于 a > b ? a : b
    print("c:", c)


def random():
    rdm = np.random.RandomState(seed=1)
    a = rdm.rand()
    b = rdm.rand(2, 3)

    print(a)
    print(b)


def vstack():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.vstack((a, b))
    print("c:\n", c)


def mgrid():
    x, y = np.mgrid[1:3:1, 2:4:0.5]
    grid = np.c_[x.ravel(), y.ravel()]
    print("x:\n", x)
    print("y:\n", y)
    print("grid:\n", grid)


