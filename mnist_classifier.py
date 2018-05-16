import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

tf.logging.set_verbosity(old_v)

def classifier():
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

	train_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cross_entropy)
	#finish the classification process

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {input_image : batch_xs, real_label : batch_ys})

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(real_label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = 100 * sess.run(accuracy, feed_dict={input_image: mnist.test.images, \
		real_label: mnist.test.labels})

	sess.close()
	return result