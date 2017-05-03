import math
import time
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_integer("hidden_units", 100, "Number of units in the hidden layer of the NN")

tf.app.flags.DEFINE_string("data_dir", "MNIST_data", "Directory for storing mnist data")

tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

logdir = "traing_log"


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")

    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        start_time = time.time()
        with tf.device(
                tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                               cluster=cluster)):
            hid_w = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS),
                name="hid_w")

            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
                               name="sm_w")

            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])

            y_ = tf.placeholder(tf.float32, [None, 10])

            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)

            hid = tf.nn.relu(hid_lin)

            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver()

            summary_op = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir="train_logs", init_op=init_op,
                                     summary_op=summary_op, saver=saver, global_step=global_step, save_model_secs=600)
            mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
            with sv.managed_session(server.target) as sess:
                step = 0

                while not sv.should_stop() and step < 100000:
                    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                    train_feed = {x: batch_xs, y_: batch_ys}

                    _, step = sess.run([train_op, global_step], feed_dict=train_feed)

                    if step % 100 == 0:
                        print("global step: {}, accuracy:{}".format(step, sess.run(accuracy,
                                                                                   feed_dict={x: mnist.test.images,
                                                                                              y_: mnist.test.labels})))

                if sv.is_chief:
                    sess.graph._unsafe_unfinalize()
                    classification_inputs = utils.build_tensor_info(x)
                    classification_outputs_classes = utils.build_tensor_info(y)
                
                    classification_signature = signature_def_utils.build_signature_def(
                        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
                        outputs={
                            signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                            signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_classes
                        },
                        method_name=signature_constants.CLASSIFY_METHOD_NAME)
                
                    tensor_info_x = utils.build_tensor_info(x)
                    tensor_info_y = utils.build_tensor_info(y)
                
                    prediction_signature = signature_def_utils.build_signature_def(
                        inputs={'images': tensor_info_x},
                        outputs={'scores': tensor_info_y},
                        method_name=signature_constants.PREDICT_METHOD_NAME)
                
                    builder = saved_model_builder.SavedModelBuilder(logdir)
                    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                
                    builder.add_meta_graph_and_variables(sess,
                                                         [tag_constants.SERVING],
                                                         signature_def_map={
                                                             'predict_images':
                                                                 prediction_signature,
                                                             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                                 classification_signature,
                                                         },
                                                         clear_devices=True,
                                                         legacy_init_op=legacy_init_op)
                    sess.graph.finalize()
                    builder.save()

    end_time = time.time()
    print('waste time:{}'.format(end_time - start_time))
    sv.stop()


if __name__ == "__main__":
    tf.app.run()
