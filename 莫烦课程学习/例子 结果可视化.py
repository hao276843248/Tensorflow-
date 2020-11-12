import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    x_data = np.linspace(-50, 50, 300)[:, np.newaxis]
    noise = np.random.normal(0, 5, x_data.shape)
    #
    y_data = np.square(x_data) - 1 + noise

    # plt.scatter(y_data)

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    # 输入层
    l1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
    l2 = add_layer(l1, 20, 10, activation_function=tf.nn.relu)
    l3 = add_layer(l2, 10, 10, activation_function=tf.nn.relu)

    # 输出层
    predition = add_layer(l3, 10, 1, activation_function=None)

    # loss = tf.square(y_data-predition)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition)
                                        , reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, 500000):
            sess.run(train_step, feed_dict={
                xs: x_data, ys: y_data
            })
            if i % 50 == 0:
                try:
                    # 移除第一条线
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                predition_value = sess.run(predition, feed_dict={
                    xs: x_data
                })
                # 输出一个 线 红色 宽度 5
                lines = ax.plot(x_data, predition_value, "r-", lw=5)
                # 图片暂停
                plt.pause(0.1)
                # plt.ion()
                # plt.show()
                print("loss:",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
