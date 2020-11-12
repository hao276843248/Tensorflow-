import tensorflow as tf

martix1 = tf.constant([[3,3]])
martix2 = tf.constant([[2],[2]])

product = tf.matmul(martix1,martix2)


# sess=tf.Session()
# result=sess.run(product)
# print(result)
# sess.close()

with tf.Session() as sess:
    print(sess.run(product))
