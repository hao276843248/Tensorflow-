import tensorflow as tf

y = "预测值"
y_ = "真实值"
"""
均方误差 （可用来判断线性问题） 
# 也可以用绝对值 但是相差变化.当两个数很接近时变化不是很大
# 用平方 但是相差变化.当两个数很接近时变化会很大
"""
# tf.square(y_ - y)  求平方
# 平方在求平均值 两个值之差约小 则 越相似 接近
loss_mse = tf.reduce_mean(tf.square(y_ - y))

"""
交叉熵，ce(cross entropy) 表征两个概率分布之间的距离
交叉熵越大 两个概率分布也远，交叉熵越大概率分布越近
"""
# 损失函数
# tf.clip_by_value(y, 1e-12, 1.0) 当y大于1是y=1,当y 控制预测y 在0-1范围内
ce = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
# 当n分类的n个输出（y1,y2,..yn）通过softmax()函数  使结果符合概率分布
# output = tf.nn.softmax(logits=y)
