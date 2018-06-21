import pickle

a = [14, 6]
b = [4, 16]
c = [13, 8]
d = [10, 5]
e = [4, 11]

with open("save_tuple.pickle", "wb") as file:
	pickle.dump((a, b, c, d, e), file)