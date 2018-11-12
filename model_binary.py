import pickle
import numpy as np
import tensorflow as tf

#matrices will soon be changed, considering the input from the pickles
with open("matrix_mul_and_count.p", "rb") as file:
	matrices_1, dummy = pickle.load(file)

matrices_0 = [[1.0 for j in range(5)] for i in range(5)]
# matrices_0c = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]

matrices = [[[matrices_1[i][j] for k in range(256)] for j in range(5)] for i in range(5)]

#define the model to be used
def conv_cifar10(input_image):
	#define the parameter used for classification
	# weight_1 = tf.Variable(tf.random_normal(shape = [3, 3, 3, 32]))
	# weight_2 = tf.Variable(tf.random_normal(shape = [3, 3, 32, 64]))
	# weight_3 = tf.Variable(tf.random_normal(shape = [3, 3, 64, 128]))
	# weight_4 = tf.Variable(tf.random_normal(shape = [3, 3, 128, 256]))
	# weight_5 = tf.Variable(tf.random_normal(shape = [3, 3, 256, 256]))

	weight_1 = tf.get_variable("weight_1", shape = [3, 3, 3, 32], initializer = tf.contrib.layers.xavier_initializer())
	weight_2 = tf.get_variable("weight_2", shape = [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
	weight_3 = tf.get_variable("weight_3", shape = [3, 3, 64, 128], initializer = tf.contrib.layers.xavier_initializer())
	weight_4 = tf.get_variable("weight_4", shape = [3, 3, 128, 256], initializer = tf.contrib.layers.xavier_initializer())
	weight_5 = tf.get_variable("weight_5", shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer())

	#now construct the model for the training and testing
	#CIFAR-10 Images are bunch of 1-D images which is actually the flattened version
	#of 32x32x3 images. it needs to be reformatted so that we may use CNN

	#define the first convolutional layer, together with its activation function
	# input_image = tf.Print(input_image, [input_image], "Input Image :", summarize = 32)
	# input_image = tf.multiply(input_image, matrices)
	conv1_1 = tf.nn.conv2d(input_image, weight_1, strides = [1, 1, 1, 1], padding = "VALID")
	conv1_1 = tf.nn.relu(conv1_1)
	conv1_2 = tf.nn.conv2d(conv1_1, weight_2, strides = [1, 1, 1, 1], padding = "VALID")
	conv1_2 = tf.nn.relu(conv1_2)

	max_pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	#define the second convolutional layer, together with its activation function
	conv2_1 = tf.nn.conv2d(max_pool1, weight_3, strides = [1, 1, 1, 1], padding = "VALID")
	conv2_1 = tf.nn.relu(conv2_1)
	conv2_2 = tf.nn.conv2d(conv2_1, weight_4, strides = [1, 1, 1, 1], padding = "VALID")
	# conv2_2 = tf.Print(conv2_2, [conv2_2], "Matrix after convolution : ", summarize = 75)
	conv2_2 = tf.nn.relu(conv2_2)
	# conv2_2 = tf.multiply(conv2_2, matrices)
	max_pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
	# max_pool2 = tf.transpose(max_pool2, perm = [0, 3, 1, 2])
	# max_pool2 = tf.Print(max_pool2, [max_pool2], "Matrix before multiplication : ", summarize = 25)
	# max_pool2 = tf.transpose(max_pool2, perm = [0, 2, 3, 1])

	#define the last convolutional layer, together with its activation function
	max_pool2 = tf.multiply(max_pool2, matrices)
	conv3 = tf.nn.conv2d(max_pool2, weight_5, strides = [1, 1, 1, 1], padding = "VALID")
	conv3 = tf.nn.relu(conv3)

	#define the fully connected layer
	pool_flat = tf.reshape(conv3, [-1, 3 * 3 * 512])
	fc1 = tf.layers.dense(inputs = pool_flat, units = 5 * 512, activation = tf.nn.relu)
	fc2 = tf.layers.dense(inputs = fc1, units = 2 * 512, activation = tf.nn.relu)
	fc3 = tf.layers.dense(inputs = fc2, units = 128, activation = tf.nn.relu)
	fc_final = tf.layers.dense(inputs = fc3, units = 2)
	# fc_final = tf.Print(fc_final, [tf.multiply(100.0, tf.nn.softmax(fc_final))], "After softmax : ", summarize = 10)

	return fc_final, weight_1, weight_2, weight_3, weight_4, weight_5