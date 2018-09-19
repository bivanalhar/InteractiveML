import numpy as np

#1.1) function to load the CIFAR-10 batch and extract the features as well as labels
def load_cifar10_batch(folder_path, batch_id):
	with open(folder_path + '/data_batch_' + str(batch_id), mode = 'rb') as file:
		batch = pickle.load(file, encoding = 'latin1')

	features = batch['data'].reshape((len(batch['data']), 3, 32, 32).transpose(0, 2, 3, 1))
	labels = batch['labels']

	return features, labels

#1.2) function to load all of the label names (there are 10 labels for this CIFAR-10)
def load_label_names():
	return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#1.3) function to normalize all the elements, to prevent the overflowing
def normalize(x):
	min_val = np.min(x)
	max_val = np.max(x)
	x = (x - min_val) / (max_val - min_val)
	return x

#1.4) function to do one-hot encoding towards the label input
def one_hot_encode(x):
	encoded = np.zeros((len(x), 10))
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	return encoded

#1.5) function to preprocess the CIFAR-10 dataset and then to put it all back
def preprocess_and_save(features, labels, filename):
	features = normalize(features)
	labels = one_hot_encode(labels)
	pickle.dump((features, labels), open(filename, 'wb'))

#1.6) function to preprocess the dataset AND extract the validation & testing dataset
def preprocess_and_save_data(folder_path, normalize, one_hot_encode):
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
						'preprocess_training.p')