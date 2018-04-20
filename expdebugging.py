"""
What to Implement : The Explanatory Debugging to Personalize Interactive ML
---------------------------------------------------------------------------

The code for this paper is inspired from the paper presented in IUI 2015 and
titled "Principles of Explanatory Debugging to Personalize Interactive Ma-
chine Learning"

What this code is doing is to firstly implement the algorithm to classify, but
further than that, it will also provide some explanation about the prediction

If possible, the code will also allow for the input by the user to modify the
learning algorithm (add/remove word, undo the change, etc)
"""

import os
import numpy as np
from nltk.tokenize import word_tokenize

vocab_list = []
cond_probability = {}

#extract the vocabulary from the sentence
def extractVocabulary(sentence):
	global vocab_list, cond_probability

	token_word = word_tokenize(sentence)
	token_word = [word.lower() for word in token_word if word.isalpha()]

	for word in token_word:
		if word not in vocab_list:
			vocab_list.append(word)
			cond_probability[word] = [0, 0]

#gathering all the vocabulary inside the data
#here we limit the number of vocabulary evaluated into around 10000
def gatherVocabulary():
	global vocab_list

	#extract the vocabulary from the class 1
	for files in os.listdir("./train/class1"):
		with open("./train/class1/" + files, "r", errors='ignore') as file:
			lines = file.readlines()

		lines = [line.strip() for line in lines if line.strip() is not '']
		for line in lines:
			extractVocabulary(line)
		if len(vocab_list) > 5000:
			break
	vocab_list = vocab_list[:5000]

	#extract the vocabulary from the class 2
	for files in os.listdir("./train/class2"):
		with open("./train/class2/" + files, "r", errors='ignore') as file:
			lines = file.readlines()

		lines = [line.strip() for line in lines if line.strip() is not '']		
		for line in lines:
			extractVocabulary(line)
		if len(vocab_list) > 10000:
			break
	vocab_list = vocab_list[:10000]

#defining the conditional probability for both classes
def calculateCondProbability():
	#accessing all the training documents
	for files in os.listdir("./train/class1"):
		file = open("./train/class1/" + files, "r", errors='ignore')
		for line in file:
			token_line = word_tokenize(line)
			token_line = [word.lower() for word in token_line if word.isalpha()]

			for word in token_line:
				if word in vocab_list:
					cond_probability[word][0] += 1
		file.close()

	for files in os.listdir("./train/class2"):
		file = open("./train/class2/" + files, "r", errors='ignore')
		for line in file:
			token_line = word_tokenize(line)
			token_line = [word.lower() for word in token_line if word.isalpha()]

			for word in token_line:
				if word in vocab_list:
					cond_probability[word][1] += 1
		file.close()

#defining the prior probability of both classes
def calculatePriorProbability():
	count_class1 = 0
	count_class2 = 0

	#counting on the number of messages on class 1
	for files in os.listdir("./train/class1"):
		count_class1 += 1

	#counting on the number of messages on class 2
	for files in os.listdir("./train/class2"):
		count_class2 += 1

	count_class = float(count_class1 + count_class2)

	#calculating the prior probability of each class
	prior_class1 = count_class1 / count_class
	prior_class2 = count_class2 / count_class

	return prior_class1, prior_class2

#defining the training process of the Naive Bayes Classifier
def trainDataset():
	pass

#defining the testing process of the Naive Bayes Classifier
def testDataset():
	pass

#defining the main function
def main():
	gatherVocabulary()
	calculateCondProbability()
	calculatePriorProbability()

	for element in cond_probability:
		if cond_probability[element] == [0,0]:
			print(element)

main()