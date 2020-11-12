import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

biases = tf.Variable(tf.zeros((1,)))
init = tf.initialize_all_variables()
# placeholder 与 feed_dict 绑定
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(biases))
    print(sess.run(output, feed_dict={input1: [4.2], input2: [8]}))
