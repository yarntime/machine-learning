import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
CELL_SIZE = 32
LR = 0.02

tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)

init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(
    rnn_cell,
    tf_x,
    initial_state = init_state,
    time_major = False,
)

outs2D = tf.reshape(outputs, [-1, CELL_SIZE])

net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)

outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

plt.figure(1, figsize=(12, 5)); plt.ion()

for step in range(60):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP)

    x = np.sin(steps)[np.newaxis, :, np.newaxis]
    y = np.cos(steps)[np.newaxis, :, np.newaxis]

    if 'final_s_' not in globals():
        feed_dict = {tf_x: x, tf_y: y}
    else:
        feed_dict = {tf_x: x, tf_y: y, init_state: final_s_}

    _, pred_, final_s_ = sess.run([train_op, outs, final_state], feed_dict)

    plt.plot(steps, y.flatten(), 'r-'); plt.plot(steps, pred_.flatten(), 'b--')
    plt.ylim((-1.2, 1.2)); plt.draw(); plt.pause(0.05)

plt.ioff(); plt.show()
