import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# 指数衰减学习率
def decay_lr():
    epoch = 40
    LR_BASE = 0.2
    LR_DECAY = 0.99
    LR_STEP = 1

    w = tf.Variable(tf.constant(5, dtype=tf.float32))

    for epoch in range(epoch):
        lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
        with tf.GradientTape() as tape:
            loss = tf.square(w + 1)
        grads = tape.gradient(loss, w)

        w.assign_sub(lr * grads)
        print("After %s epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy(), loss, lr))


