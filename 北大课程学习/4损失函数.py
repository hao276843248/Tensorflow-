import tensorflow as tf

"""
均方误差 （可用来判断线性问题）
loss_mse = tf.reduce_mean(tf.square(y_-y))
"""

"""
交叉熵，ce(cross entropy) 表征两个概率分布之间的距离
交叉熵越大 两个概率分布也远，交叉熵越大概率分布越近
"""
y = "预测值"
y_ = "真实值"
# tf.clip_by_value(y, 1e-12, 1.0) 当y大于1是y=1,当y
ce = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
