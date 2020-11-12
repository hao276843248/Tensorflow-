import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 123456

rdm = np.random.RandomState(seed)
X = rdm.rand(32, 2)
Y = [[2 * x1 + x2 + (rdm.random() / 10.0 - 0.05)] for (x1, x2) in X]
print(X)
print(Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1))
b = tf.Variable(tf.ones((1,)), dtype=tf.float32)
y = tf.matmul(x, w1) + b

# 绝对值为
# loss_mse = tf.reduce_mean(abs(y - y_))
loss_abs = tf.reduce_mean(tf.abs(y - y_))
loss_mse = tf.reduce_mean(tf.pow(y - y_, 2))
# 自定义损失函数
# y<y_ 预测的y少了,损失利润  (y-y_)*9
# y>=y_ 预测的y多了,损失利润  (y-y_)*1
# tf.where( tf.greater(y, y_),(y>y_ ? ),(真:说明预测的销量 多了 损失了成本成本比较低),(假:说明损失了利润利润较高,则损失函数会变大，也就是要优化，测尽量要多预测))
# loss_mse = tf.reduce_mean(tf.where(tf.greater(y, y_), abs(y - y_) * 1, abs(y - y_) * 9))
# loss_mse = tf.reduce_mean(tf.where(tf.greater(y, y_), abs(y - y_) * 9, abs(y - y_) * 1))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss_mse)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w1))
    STEPS = 5000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 20 == 0:
            print("loss_mse", sess.run(loss_mse, feed_dict={x: X[start:end], y_: Y[start:end]}))
            print("loss_abs", sess.run(loss_abs, feed_dict={x: X[start:end], y_: Y[start:end]}))
            print("=======")
