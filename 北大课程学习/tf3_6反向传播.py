# 0 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 12345
# 基于seed生成随机数
rng = np.random.RandomState(seed)
# 生成32行2列的矩阵, 表示，32组 体积和重量,作为输入集合
X = rng.rand(32, 2)
# 从32行2列中取出一行 判断如果小于1 赋值1，否则0
# 作为输入集合的标签（正确答案）
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print(X)
print(Y)

# 输入参数
x = tf.placeholder(tf.float32, shape=[None, 2])
# 结果参数
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# 权重设置
w1 = tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1))
#
w2 = tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.002).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.002, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    print("w1", sess.run(w1))
    print("w2", sess.run(w2))
    for i in range(5000):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 50 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("loss", total_loss)

    print("训练后w1", sess.run(w1))
    print("训练后w2", sess.run(w2))