import tensorflow as tf
import go_parser
from datetime import datetime

def main():
    t1 = datetime.now()
    basic_softmax_NN()
    t2 = datetime.now()

    print "time spent: ", t2-t1



def basic_softmax_NN():
    train_data, val_data, test_data = go_parser.parse_games(
        num_games='All', onehot=True)
    go_data = go_parser.prepare_data_sets(train_data, val_data, test_data)

    x = tf.placeholder(tf.float32, [None, 361])
    W1 = tf.Variable(tf.zeros([361, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.zeros([100, 361]))
    b2 = tf.Variable(tf.zeros([361]))
    y = tf.nn.softmax(tf.matmul(y1, W2) + b2)
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