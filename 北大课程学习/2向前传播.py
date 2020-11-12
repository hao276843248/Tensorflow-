import tensorflow as tf

sess = tf.Session()
"""参数"""
# 即为 w,b 这些权重，通常用变量标识,随机给初始值
w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
#                 正态分布随机数  产生2*3的矩阵，标准差2  均值0  随机种子1 如果去掉 每次生成的随机值将不一致
# tf.truncated_normal() 去掉过大偏离点的正态分布：如果随机出来的数据超过两个标准差 这重新生成
# tf.random_uniform() 生成平均分布的值
zero = tf.zeros([3, 4], dtype=tf.float32)  # 生成全是0的
print(sess.run(zero))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
tf.ones([3, 4], dtype=tf.float32)  # 生成全是1的
fil = tf.fill([1, 2], 5.0)  # 生成全是定值数组
#            生成矩阵, 矩阵内的值
print(sess.run(fil))
# [[5. 5.]]
tf.constant([1, 2, 3], dtype=tf.float32)  # 生成确定的值

"""
神经网络实现过程
1.准备数据,提前特征，输入神经网络
2、搭建神经网络，输入到输出，
3、大量特征数据给NN迭代优化神经网络
4、使用训练好的神经网络预测，分类
"""
