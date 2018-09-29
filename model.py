import pickle
import numpy as np
import tensorflow as tf

#define the model to be used
def conv_cifar10(input_image):
	#define the parameter used for classification
	weight_1 = tf.Variable(tf.random_normal(shape = [3, 3, 3, 32]))
	weight_2 = tf.Variable(tf.random_normal(shape = [3, 3, 32, 64]))
	weight_3 = tf.Variable(tf.random_normal(shape = [3, 3, 64, 128]))

	#now construct the model for the training and testing
	#CIFAR-10 Images are bunch of 1-D images which is actually the flattened version
	#of 32x32x3 images. it needs to be reformatted so that we may use CNN

	#define the first convolutional layer, together with its activation function
	conv1 = tf.nn.conv2d(input_image, weight_1, strides = [1, 1, 1, 1], padding = "VALID")
	conv1 = tf.nn.relu(conv1)
	max_pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	#define the second convolutional layer, together with its activation function
	conv2 = tf.nn.conv2d(max_pool1, weight_2, strides = [1, 1, 1, 1], padding = "VALID")
	conv2 = tf.nn.relu(conv2)
	max_pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	#define the third convolutional layer, together with its activation function
	conv3 = tf.nn.conv2d(max_pool2, weight_3, strides = [1, 1, 1, 1], padding = "VALID")
	conv3 = tf.nn.relu(conv3)
	max_pool3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	#define the fully connected layer
	pool_flat = tf.reshape(max_pool3, [-1, 3 * 3 * 128])
	fc1 = tf.layers.dense(inputs = pool_flat, units = 512, activation = tf.nn.relu)
	fc2 = tf.layers.dense(inputs = fc1, units = 256, activation = tf.nn.relu)
	fc3 = tf.layers.dense(inputs = fc2, units = 128, activation = tf.nn.relu)
	fc4 = tf.layers.dense(inputs = fc3, units = 64, activation = tf.nn.relu)
	fc_final = tf.layers.dense(inputs = fc4, units = 10)

	return fc_final, weight_1, weight_2, weight_3