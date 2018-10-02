import numpy as np
import pickle

#1) function to load the CIFAR-10 batch and extract the features as well as labels
def load_cifar10_batch(folder_path, batch_id):
	with open(folder_path + '/data_batch_' + str(batch_id), mode = 'rb') as file:
		batch = pickle.load(file, encoding = 'latin1')

	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']

	return features, labels

#2) function to load all of the label names (there are 10 labels for this CIFAR-10)
def load_label_names():
	return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#3) function to do one-hot encoding towards the label input
def one_hot_encode(x):
	encoded = np.zeros((len(x), 10))
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	return encoded

#4) function to preprocess the CIFAR-10 dataset and then to put it all back
def preprocess_and_save(features, labels, filename):
	labels = one_hot_encode(labels)
	pickle.dump((features, labels), open(filename, 'wb'))

#5) function to preprocess the dataset AND extract the validation & testing dataset
def preprocess_and_save_data(folder_path):
	n_batches = 5
	valid_features = []
	valid_labels = []

	for batch_i in range(1, n_batches + 1):
		features, labels = load_cifar10_batch(folder_path, batch_i)

		index_of_validation = int(len(features) * 0.1)
		preprocess_and_save(features[:-index_of_validation], labels[:-index_of_validation],
							'preprocess_batch_' + str(batch_i) + '.p')

		valid_features.extend(features[-index_of_validation:])
		valid_labels.extend(labels[-index_of_validation:])

	preprocess_and_save(np.array(valid_features), np.array(valid_labels),
						'preprocess_validation.p')
	with open(folder_path + '/test_batch', mode = 'rb') as file:
		batch = pickle.load(file, encoding = 'latin1')

	test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	test_labels = batch['labels']
	preprocess_and_save(np.array(test_features), np.array(test_labels),
						'preprocess_testing.p')

#6) function to split the features and labels into batches of determined size
def batch_features_labels(features, labels, batch_size):
	for start in range(0, len(features), batch_size):
		end = min(start + batch_size, len(features))
		yield features[start:end], labels[start:end]

#7) function to load the preprocessed training data and return in batches of size or less
def load_preprocess_training_batch(batch_id, batch_size):
	filename = 'preprocess_batch_' + str(batch_id) + '.p'
	features, labels = pickle.load(open(filename, mode='rb'))

	# Return the training data in batches of size <batch_size> or less
	return batch_features_labels(features, labels, batch_size)

#8) function to load all batches and return
def load_batch(batch_id):
	filename = 'preprocess_batch_' + str(batch_id) + '.p'
	features, labels = pickle.load(open(filename, mode='rb'))

	# Return the training data in batches of size <batch_size> or less
	return features, labels

#9) build up the dictionary for the label naming
def build_dictionary_for_cifar10_image():
	names = load_label_names()
	label_dict = {}
	for i in range(len(names)):
		label_dict[i] = names[i]

	return label_dict

# preprocess_and_save_data("./cifar-10")