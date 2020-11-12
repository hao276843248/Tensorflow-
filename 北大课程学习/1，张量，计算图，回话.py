import tensorflow as tf

'''张量'''
a = tf.constant([[1, 2]], dtype=tf.float32)
b = tf.constant([[3], [4]], dtype=tf.float32)
# 0 阶 标量  s=123
# 1 阶 向量  s=[1,2,3]
# 2 阶 矩阵  s=[[1,2,3],[2,3,4]]
# 3 阶 张量  s=[[[[..... n多个括号
# 张量可标识 0到n阶的数组 (列表)
result = a + b
print(a)
# Tensor("Const:0", shape=(2,), dtype=float32)
#        节点名(常量):第0个输出   维度 ：1伟数组长度为2 数据类型 float32
print(result)
# 计算图只描述计算过程，不运算结果
# Tensor("add:0",        shape=(2,),        dtype=float32)
#        节点名:第0个输出   维度 ：1伟数组长度为2 数据类型 float32

# y=xw  >> =x1*w1+x2*w2
'''图运算'''
y = tf.matmul(a, b)
print(y)
"""会话"""
with tf.Session() as sess:
    print(sess.run(y))
sess = tf.Session()
