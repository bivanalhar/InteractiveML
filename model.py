import numpy as np
import tensorflow as tf
import cifar10_func as cf

#define the graph input
input_image = tf.placeholder(tf.float32, [None, 32, 32, 3]) #size of CIFAR-10 image is 32 x 32
input_label = tf.placeholder(tf.float32, [None, 10]) #there are 10 possible labels for CIFAR-10

#define the model to be used
def conv_cifar10(input_image):
	#define the parameter used for classification
	weight_1 = tf.Variable(tf.random_normal(shape = [3, 3, int(data.get_shape()[3]), 32]))
	weight_2 = tf.Variable(tf.random_normal(shape = [3, 3, 32, 64]))

	#now construct the model for the training and testing
	#CIFAR-10 Images are bunch of 1-D images which is actually the flattened version
	#of 32x32x3 images. it needs to be reformatted so that we may use CNN
	input_image = tf.reshape(input_image, shape = [None, 32, 32, 3])

	#define the first convolutional layer, together with its activation function
	conv1 = tf.nn.conv2d(input_image, weight_1, strides = [1, 1, 1, 1], padding = "VALID")
	conv1 = tf.nn.relu(conv1)

	#define the second convolutional layer, together with its activation function
	conv2 = tf.nn.conv2d(conv1, weight_2, strides = [1, 1, 1, 1], padding = "VALID")
	conv2 = tf.nn.relu(conv2)

	#define the maximum and also average pooling layer
	max_pool = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
	avg_pool = tf.nn.avg_pool(max_pool, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	#define the fully connected layer
	avg_pool_flat = tf.reshape(avg_pool, [-1, 7 * 7 * 64])
	fc1 = tf.layers.dense(inputs = avg_pool_flat, units = 1024, activation = tf.nn.relu)
	fc_final = tf.layers.dense(inputs = fc1, units = 10, activation = tf.nn.relu)

	return fc_final

logits = conv_cifar10(input_image)
#now we reach the end of the model construction, next is the training method

#defining the loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels = input_label, logits = logits)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.05).minimize(loss)

correct = tf.equal(tf.argmax(input_label, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#initializing all the trainable parameters
init_op = tf.global_variables_initializer()