import tensorflow as tf


def tf3_4():
    # 两层网络（全连接）
    # 定义输出和参数
    # x = tf.constant([[0.7, 0.5]])
    x = tf.placeholder(tf.float32, shape=[1, 2])
    # 定义神经元个数   2个输入类型，5个神经元
    w1 = tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1))
    # 5个神经元 输出 1个返回值
    w2 = tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))

    # 定义向前传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(y, feed_dict={x: [[0.7, 0.5]]}))


def tf3_5():
    # 两层网络（全连接）
    # 定义输出和参数
    # x = tf.constant([[0.7, 0.5]])
    x = tf.placeholder(tf.float32, shape=[None, 2])
    # 定义神经元个数   2个输入类型，5个神经元
    w1 = tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1))
    # 5个神经元 输出 1个返回值
    w2 = tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))

    # 定义向前传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.4], [0.9, 0.6], [0.4, 0.5]]}))


if __name__ == '__main__':
    tf3_5()
