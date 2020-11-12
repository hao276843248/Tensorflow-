import tensorflow as tf
import numpy as np


## Save to file
def saves():
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name="weights")
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name="biases")
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "my_net/save_nte.ckpt")
        print("Save to path:", save_path)


def remember():
    # restore variable
    pass
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
    # 不用 init
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_nte.ckpt")
        print("weights:", sess.run(W))
        print("biases:", sess.run(b))

if __name__ == '__main__':
    remember()