import tensorflow  as tf

state=tf.Variable(0,name="name")
print(state.name)
con=tf.constant(1)

new_value=tf.add(state,con)
update=tf.assign(state,new_value)

# 变量初始化
init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,3):
        sess.run(update)
        print(sess.run(state))