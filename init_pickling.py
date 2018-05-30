import pickle

a = [14, 6]
b = [4, 16]

with open("save_tuple.pickle", "wb") as file:
	pickle.dump((a, b), file)