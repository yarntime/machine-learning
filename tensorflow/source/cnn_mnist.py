import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets("./mnist/", one_hot=True)
test_x = mnist.test.images[0:2000]
test_y = mnist.test.labels[0:2000]

print(mnist.test.images.shape)
print(mnist.test.labels.shape)

tf_x = tf.placeholder(tf.float32, [None, 28 * 28])
image = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, [None, 10])

conv1 = tf.layers.conv2d(
    inputs = image,
    filters = 16,
    kernel_size = 3,
    strides = 1,
    padding = "same",
    activation = tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
    inputs = conv1,
    pool_size = 2,
    strides = 2,
)

conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters = 32,
    kernel_size = 3,
    strides = 1,
    padding = "same",
    activation = tf.nn.relu
)

pool2 = tf.layers.max_pooling2d(
    inputs = conv2,
    pool_size = 2,
    strides = 2,
)

flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

output = tf.layers.dense(flat, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y, logits = output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(
    labels = tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    for step in range(1000):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], feed_dict={tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:
            accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
