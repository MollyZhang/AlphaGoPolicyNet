import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import copy
import go_parser as gp
from datetime import datetime

def main():
    go_data = gp.parse_games(num_games=1000, first_n_moves=10000, onehot=True)
    basic_3layer_NN(go_data)


def conv(go_data, learning_rate=1e-4, drop_out_rate=0.5,
         conv_patch_size=6, conv_features=20, hidden_nodes=2000,
         dropout_rate=0.5, verbose=True, pooling=True):

    go_data.train._epochs_completed = 0
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 361])
    y_ = tf.placeholder(tf.float32, shape=[None, 361])

    W_conv1 = weight_variable([conv_patch_size, conv_patch_size, 1, conv_features])
    b_conv1 = bias_variable([conv_features])
    x_board = tf.reshape(x, [-1, 19, 19, 1])

    h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

    b_fc1 = bias_variable([hidden_nodes])

    if pooling:
        h_pool1 = max_pool_2x2(h_conv1)
        W_fc1 = weight_variable([10 * 10 * conv_features, hidden_nodes])
        h_pool2_flat = tf.reshape(h_pool1, [-1, 10*10*conv_features])
    else:
        W_fc1 = weight_variable([19 * 19 * conv_features, hidden_nodes])
        h_pool2_flat = tf.reshape(h_conv1, [-1, 19*19*conv_features])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([hidden_nodes, 361])
    b_fc2 = bias_variable([361])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    sess.run(tf.initialize_all_variables())

    return train(go_data, sess, train_step, accuracy, x, y, y_, keep_prob,
                 dropout_rate, verbose)


def basic_3layer_NN(go_data, verbose=True,
                    learning_rate=1.0,
                    hidden_layer_num=2000,
                    dropout_rate=0.5):

    x = tf.placeholder(tf.float32, [None, 361])
    W1 = weight_variable([361, hidden_layer_num])
    b1 = bias_variable([hidden_layer_num])
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    keep_prob = tf.placeholder(tf.float32)
    y_drop = tf.nn.dropout(y1, keep_prob)

    W2 = weight_variable([hidden_layer_num, 361])
    b2 = bias_variable([361])

    y = tf.nn.softmax(tf.matmul(y_drop, W2) + b2)

    y_ = tf.placeholder(tf.float32, [None, 361])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess.run(tf.initialize_all_variables())
    # saver = tf.train.Saver()
    return train(go_data, sess, train_step, accuracy, x, y, y_, keep_prob,
                 dropout_rate, verbose)



def train(go_data, sess, train_step, accuracy, x, y, y_,
          keep_prob, dropout_rate, verbose):
    go_data.train._epochs_completed = 0
    best_accuracy = 0
    previous_epoch = 0
    epoch_times = []
    t1 = datetime.now()
    best_accu_updated = 0   # how many epochs ago is the best accuracy updated#
    train_accuracies = []
    val_accuracies = []
    while best_accu_updated < 10 and previous_epoch <= 50:
        batch = go_data.train.next_batch(128)
        if go_data.train.epochs_completed > previous_epoch:
            previous_epoch = go_data.train.epochs_completed
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})
            val_accuracy = accuracy.eval(feed_dict={
                x: go_data.validation.features, y_: go_data.validation.labels, keep_prob:1.0})
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            if verbose:
                print("epoch %d: training accuracy %g, validation accuracy %g" %(
                    previous_epoch, train_accuracy, val_accuracy))
            if val_accuracy > best_accuracy:
                if verbose: print "best accuracy"
                best_accu_updated = 0
                best_accuracy = copy.deepcopy(val_accuracy)
            best_accu_updated += 1

        t3 = datetime.now()
        train_step.run(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: (1-dropout_rate)})
        t4 = datetime.now()
        epoch_times.append((t4-t3).total_seconds())
    epoch_time = sum(epoch_times)/go_data.train.epochs_completed

    t2 = datetime.now()
    training_time = (t2-t1).total_seconds()

    # if modelfile:
    #     saver.save(sess, modelfile)

    probabilities = y
    boards = go_data.test.features[:10]
    moves = go_data.test.labels[:10]
    probs = []
    for i in range(10):
        feed_dict = {x: [boards[i]], y_: [moves[i]], keep_prob: 1.0}
        probs.append(probabilities.eval(feed_dict=feed_dict, session=sess))
    return boards, moves, probs

    # return train_accuracies, val_accuracies, training_time, epoch_time


def basic_softmax_NN():
    go_data = go_parser.parse_games(num_games=100, onehot=True)

    x = tf.placeholder(tf.float32, [None, 361])
    W = tf.Variable(tf.zeros([361, 361]))
    b = tf.Variable(tf.zeros([361]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 361])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    best_accuracy = 0
    previous_epoch = 0

    while go_data.train.epochs_completed < 50:
        batch = go_data.train.next_batch(500)
        if go_data.train.epochs_completed > previous_epoch:
            previous_epoch = go_data.train.epochs_completed
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict={
                x: go_data.validation.features, y_: go_data.validation.labels})
            print("epoch %d: training accuracy %g, validation accuracy %g" %(
                previous_epoch, train_accuracy, val_accuracy))
            if val_accuracy > best_accuracy:
                print "best accuracy"
                best_accuracy = copy.deepcopy(val_accuracy)

        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    test_accuracy = accuracy.eval(feed_dict={
        x: go_data.test.features, y_: go_data.test.labels})
    print "test accuracy %f" %test_accuracy

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
