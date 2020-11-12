import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 添加一层神经层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 权重  矩阵
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 推荐值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 没激活的值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 激活函数执行
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10])

    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(20000):
        # 提取100个数据
        batch_xs, batch_ys = mnist.train.next_batch(100)
        result = sess.run(train_step, feed_dict={
            xs: batch_xs, ys: batch_ys
        })
        if i % 50 == 0:
            print("损失：",sess.run(cross_entropy, feed_dict={
                xs: batch_xs, ys: batch_ys
            }))
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
            # print(sess.run(prediction, feed_dict={
            #     xs: batch_xs
            # }))
