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
# flags.DEFINE_float('learning_rate', 0.0001, 'Rate of learning for the optimizer')
flags.DEFINE_integer('checkpoint_mode', 0, 'to start training from checkpoint or initialize')

logits, weight_1, weight_2, weight_3 = md.conv_cifar10(input_image)
#now we reach the end of the model construction, next is the training method

#defining the loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = input_label, logits = logits))
loss += 0.005 * (tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2) + tf.nn.l2_loss(weight_3))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(input_label, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]

with tf.Session() as sess:
	if FLAGS.training_mode == 1: #means it enters the training mode, so the model shall be trained in order

		#1) to train the saved model/initialized model for each of the learning_rate value
		for learning_rate1 in learning_rate_list:
			learning_rate = learning_rate1

			if FLAGS.checkpoint_mode == 1: #means we will train the model from the checkpoint
				saver.restore(sess, "./pretrained/cifar10_model.ckpt")
			else: #means we will train the model from the initialization point
				sess.run(init_op)

			#NOTE : it will be changed accordingly, since the data for training will be different
			for epoch in range(150):
				n_batches = 5
				loss_batch = [None, None, None, None, None]

				for batch_i in range(1, n_batches + 1):
					avg_loss = 0.
					features, labels = cf.load_batch(batch_i)
					total_batch = int(len(features) / 128) + 1
					ptr = 0

					for batch in range(total_batch):
						batch_features, batch_labels = features[ptr:ptr+128], labels[ptr:ptr+128]
						ptr += 128

						_, loss_val = sess.run([optimizer, loss], feed_dict = {input_image : batch_features, input_label : batch_labels})
						avg_loss += loss_val / total_batch
					loss_batch[batch_i - 1] = avg_loss

				if epoch % 10 == 9:
					print("Epoch " + str(epoch + 1) + " loss = " + str(np.average(loss_batch)))

			print("finished training with learning rate " + str(learning_rate))
			#after all the possible learning rate has been finished, save each one of them
			save_path = saver.save(sess, "./pretrained/cifar10_model_" + str(1e4 * learning_rate) + ".ckpt")
			print("saved the model parameter for learning rate " + str(learning_rate))

		#2) to validate the already trained model, and then choose one with best validation accuracy
		acc_max = 0.
		desired_learning_rate = 0.
		for learning_rate1 in learning_rate_list:
			learning_rate = learning_rate1
			print("\ntry to validate the model parameter with learning rate " + str(learning_rate))
			saver.restore(sess, "./pretrained/cifar10_model_" + str(1e4 * learning_rate) + ".ckpt")

			acc_val = 0.
			features_val, labels_val = pickle.load(open('preprocess_validation.p', mode = 'rb'))
			total_batch = int(len(features_val) / 128) + 1
			ptr = 0

			for batch in range(total_batch):
				batch_features, batch_labels = features_val[ptr:ptr+128], labels_val[ptr:ptr+128]
				ptr += 128

				acc_ = sess.run(accuracy, feed_dict = {input_image : batch_features, input_label : batch_labels})
				acc_val += acc_ / total_batch

			if acc_max < acc_val:
				acc_max = acc_val
				desired_learning_rate = learning_rate

		saver.restore(sess, "./pretrained/cifar10_model_" + str(1e4 * desired_learning_rate) + ".ckpt")
		print("the chosen learning rate shall be " + str(desired_learning_rate))
		save_path = saver.save(sess, "./pretrained/cifar10_model.ckpt")

	else:
		saver.restore(sess, "./pretrained/cifar10_model.ckpt")

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