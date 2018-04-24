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
stopwords = []

#extracting all the words inside the stopwords
with open("./stopwords.txt", "r", errors = 'ignore') as stopfile:
	stoplines = stopfile.readlines()

stoplines = [line.strip() for line in stoplines if line.strip() is not '']

#accessing the first class
for files in os.listdir("./train/class1"):
	with open("./train/class1/" + files, "r", errors = 'ignore') as file:
		lines = file.readlines()

	lines = [line.strip() for line in lines if line.strip() is not '']

	#either add vocabulary or increment the value of the existing one
	for line in lines:
		token_line = word_tokenize(line)
		token_line = [word.lower() for word in token_line if word.isalpha()]

		for word in token_line:
			if word not in stoplines:
				#if we need to add word in the vocabulary list
				if word not in vocab_list and len(vocab_list) < 10000:
					vocab_list.append(word)
					cond_probability[word] = [1.0, 0.0]

				#if we only need to increment the value inside the dictionary
				else:
					cond_probability[word][0] += 1.0

#accessing the second class
for files in os.listdir("./train/class2"):
	with open("./train/class2/" + files, "r", errors = 'ignore') as file:
		lines = file.readlines()

	lines = [line.strip() for line in lines if line.strip() is not '']

	#either add vocabulary or increment the value of the existing one
	for line in lines:
		token_line = word_tokenize(line)
		token_line = [word.lower() for word in token_line if word.isalpha()]

		for word in token_line:
			if word not in stoplines:
				#if we need to add word in the vocabulary list
				if word not in vocab_list and len(vocab_list) < 20000:
					vocab_list.append(word)
					cond_probability[word] = [0.0, 1.0]

				#if we only need to increment the value inside the dictionary
				else:
					cond_probability[word][1] += 1.0

#calculating the prior class probability
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