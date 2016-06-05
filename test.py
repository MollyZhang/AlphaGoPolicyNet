import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print mnist

batch = mnist.train.next_batch(50)
print batch
print type(batch[0]), batch[0].shape
print type(batch[1]), batch[1].shape