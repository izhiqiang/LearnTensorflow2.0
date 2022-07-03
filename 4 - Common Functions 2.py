import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


def features_labels():
    features = tf.constant([12, 23, 10, 17])
    labels = tf.constant([0, 1, 1, 0])

    datasets = tf.data.Dataset.from_tensor_slices((features, labels))
    print(datasets)
    for element in datasets:
        print(element)


def calculate_backpropagation():
    with tf.GradientTape() as tape:
        w = tf.Variable(tf.constant(3.0))
        loss = tf.pow(w, 2)

    grad = tape.gradient(loss, w)
    print(grad)


def ergodic():
    seq = ['one', 'two', 'three']
    for i, element in enumerate(seq):
        print(i, element)


def onehot_encoding():
    classes = 3
    labels = tf.constant([1, 0, 2])
    output = tf.one_hot(labels, depth=classes)
    print(output)


def softmax():
    x = tf.constant([1.01, 2.01, -0.66])
    x_pro = tf.nn.softmax(x)

    print("After softmax, x_pro is:", x_pro)


def assign_sub():
    x = tf.Variable(4)
    x.assign_sub(1)
    print(x)


def argmax():
    test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    print(test)
    print(tf.argmax(test, axis=0))
    print(tf.argmax(test, axis=1))
