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

#begin to implement the naive bayes classification
#Step 1 : calculating the probability for each word
for word in cond_probability:

	#smoothing the term for the probability calculation
	cond_probability[word][0] += 1.0
	cond_probability[word][1] += 1.0

	#calculating the probability for each word
	cond_total_word = cond_probability[word][0] + cond_probability[word][1]
	cond_probability[word][0] = np.log(cond_probability[word][0]) - np.log(cond_total_word)
	cond_probability[word][1] = np.log(cond_probability[word][1]) - np.log(cond_total_word)

firstfile = [0,0]
secondfile = [0,0]

#Step 2 : calculating the probability for the test dataset
for files in os.listdir("./test/class1"):
	initial_prob = [0, 0]
	with open("./test/class1/" + files, "r", errors = 'ignore') as file:
		lines = file.readlines()

	lines = [line.strip() for line in lines if line.strip() is not '']

	for line in lines:
		token_line = word_tokenize(line)
		token_line = [word.lower() for word in token_line if word.isalpha()]

		token_line = [word for word in token_line if word in vocab_list]

		for word in token_line:
			initial_prob[0] += cond_probability[word][0]
			initial_prob[1] += cond_probability[word][1]

	if initial_prob[0] >= initial_prob[1]:
		firstfile[0] += 1.0
	else:
		firstfile[1] += 1.0

for files in os.listdir("./test/class2"):
	initial_prob = [0, 0]
	with open("./test/class2/" + files, "r", errors = 'ignore') as file:
		lines = file.readlines()

	lines = [line.strip() for line in lines if line.strip() is not '']

	for line in lines:
		token_line = word_tokenize(line)
		token_line = [word.lower() for word in token_line if word.isalpha()]

		token_line = [word for word in token_line if word in vocab_list]

		for word in token_line:
			initial_prob[0] += cond_probability[word][0]
			initial_prob[1] += cond_probability[word][1]

	# print(initial_prob[0], initial_prob[1])
	if initial_prob[0] >= initial_prob[1]:
		secondfile[0] += 1.0
	else:
		secondfile[1] += 1.0

#measuring the accuracy of the classification
correct_guess = firstfile[0] + secondfile[1]
wrong_guess = firstfile[1] + secondfile[0]
print("Accuracy for the test dataset is " + str(correct_guess * 100 / (correct_guess + wrong_guess)))

#now start for the interactive part
#Part 1 : Explainability

input_file = raw_input("please enter the document code that wants to be explained: ")

try:
	file_input = open("./test/class1/" + input_file, "r", errors = 'ignore')
	file_lines = file_input.readlines()
	file_input.close()
except:
	print("Sorry, no such file\n")

word_list = []
for line in file_lines:
	token_line = word_tokenize(line)
	token_line = [word.lower() for word in token_line if word.isalpha()]

	token_line = [word for word in token_line if word in vocab_list]

	for word in token_line:
		if word not in word_list:
			word_list.append(word)

#begin to explain the reasoning behind the prediction