import tensorflow as tf


# 正态分布的 4X4X4 三维矩阵，平均值 0， 标准差 1
normal = tf.truncated_normal([4, 4, 4], mean=0.0, stddev=1.0)
#
a = tf.Variable(tf.random_normal([2,2],seed=1))
b = tf.Variable(tf.truncated_normal([2,2],seed=2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(normal))

