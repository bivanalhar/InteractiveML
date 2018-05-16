import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#variable to put on the input image and one-hot encoded label
input_image = tf.placeholder(tf.float32, [None, 784])
real_label = tf.placeholder(tf.float32, [None, 10])

#the parameters to be trained : weight and bias
weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

#begin the classification process
y = tf.nn.softmax(tf.matmul(input_image, weight) + bias)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(real_label * tf.log(y), reduction_indices = [1]))

train_step = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cross_entropy)
#finish the classification process

