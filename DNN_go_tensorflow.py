import tensorflow as tf
import go_parser
from datetime import datetime

def main():
    t1 = datetime.now()
    conv()
    t2 = datetime.now()
    print "time spent: ", t2-t1


def conv():
    train_data, val_data, test_data = go_parser.parse_games(
        num_games=1000, onehot=True)
    go_data = go_parser.prepare_data_sets(train_data, val_data, test_data)
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 361])
    y_ = tf.placeholder(tf.float32, shape=[None, 361])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_board = tf.reshape(x, [-1, 19, 19, 1])

    h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 361])
    b_fc2 = bias_variable([361])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = go_data.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: go_data.test.images, y_: go_data.test.labels, keep_prob: 1.0}))


def basic_softmax_NN():
    train_data, val_data, test_data = go_parser.parse_games(
        num_games='All', onehot=True)
    go_data = go_parser.prepare_data_sets(train_data, val_data, test_data)

    x = tf.placeholder(tf.float32, [None, 361])
    W = tf.Variable(tf.zeros([361, 361]))
    b = tf.Variable(tf.zeros([361]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 361])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    for i in range(20000):
        batch = go_data.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print("test accuracy %g"%sess.run(accuracy, feed_dict={
        x: go_data.test.features, y_: go_data.test.labels}))




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    main()