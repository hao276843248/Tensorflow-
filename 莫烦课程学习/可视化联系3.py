import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加一层神经层
def add_layer(inputs, in_size, out_size, n_layer_name, activation_function=None):
    layer_name = f"layer{n_layer_name}"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # 权重  矩阵
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name + "/Weights", Weights)
        with tf.name_scope("biasesa"):
            # 推荐值不为0
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.summary.histogram(layer_name + "/biases", Weights)
        with tf.name_scope("Wx_plus_b"):
            # 没激活的值
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # 激活函数执行
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + "/output", outputs)
        return outputs


if __name__ == '__main__':
    # 可视化训练结果
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    print(- 0.5 + noise)

    # plt.scatter(y_data)
    with tf.name_scope("inputs"):
        xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
        ys = tf.placeholder(tf.float32, [None, 1], name="y_input")
    # 输入层
    l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
    # 输出层
    predition = add_layer(l1, 10, 1, 2, activation_function=None)

    with tf.name_scope("loss"):
        # loss = tf.square(y_data-predition)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
        tf.summary.scalar("loss", loss)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    #     python C:\Users\goldwind\AppData\Roaming\Python\Python36\site-packages\tensorboard\main.py --logdir=logs
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        sess.run(train_step, feed_dict={
            xs: x_data, ys: y_data
        })
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result,i)
