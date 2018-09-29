import numpy as np
import tensorflow as tf
import pickle
import model as md
import cifar10_func as cf

#define the graph input
input_image = tf.placeholder(tf.float32, [None, 32, 32, 3]) #size of CIFAR-10 image is 32 x 32
input_label = tf.placeholder(tf.float32, [None, 10]) #there are 10 possible labels for CIFAR-10

learning_rate = 0.0001

#defining the mode whether it's training or testing
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_mode', 0, 'the execution mode, whether it is training or testing')
flags.DEFINE_float('learning_rate', 0.0001, 'Rate of learning for the optimizer')

logits, weight_1, weight_2, weight_3 = md.conv_cifar10(input_image)
#now we reach the end of the model construction, next is the training method

#defining the loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = input_label, logits = logits))
loss += 0.01 * (tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2) + tf.nn.l2_loss(weight_3))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(input_label, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#initializing all the trainable parameters
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

learning_rate_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

with tf.Session() as sess:
	if FLAGS.training_mode == 1:
		for learning_rate1 in learning_rate_list:
			learning_rate = learning_rate1
			sess.run(init_op)

			for epoch in range(200):
				n_batches = 5
				loss_batch = [None, None, None, None, None]

				for batch_i in range(1, n_batches + 1):
					avg_loss = 0.
					features, labels = cf.load_batch(batch_i)
					# print(np.shape(labels), labels[0])
					total_batch = int(len(features) / 128) + 1
					ptr = 0

					for batch in range(total_batch):
						batch_features, batch_labels = features[ptr:ptr+128], labels[ptr:ptr+128]
						ptr += 128

						_, loss_val = sess.run([optimizer, loss], feed_dict = {input_image : batch_features, input_label : batch_labels})
						avg_loss += loss_val / total_batch
					loss_batch[batch_i - 1] = avg_loss

				if epoch % 5 == 4:
					print("Epoch " + str(epoch + 1) + " loss = " + str(np.average(loss_batch)))

			print("finished training with learning rate " + str(learning_rate))
			save_path = saver.save(sess, "./pretrained/cifar10_model_" + str(1e4 * learning_rate) + ".ckpt")

	else:
		saver.restore(sess, "./pretrained/cifar10_model_10.0.ckpt")

		acc = 0.
		features_test, labels_test = pickle.load(open('preprocess_testing.p', mode = 'rb'))
		total_batch = int(len(features_test) / 128) + 1
		ptr = 0

		for batch in range(total_batch):
			batch_features, batch_labels = features_test[ptr:ptr+128], labels_test[ptr:ptr+128]
			ptr += 128

			acc_ = sess.run(accuracy, feed_dict = {input_image : batch_features, input_label : batch_labels})
			acc += acc_ / total_batch
		print('Accuracy of the model ' + str(100 * acc))