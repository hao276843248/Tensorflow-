import tensorflow as tf


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
