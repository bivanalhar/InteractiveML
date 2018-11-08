import pickle

image_count = 0
matrix_count = [[0 for i in range(5)] for j in range(5)]
matrix_count2 = [[0 for i in range(32)] for j in range(32)]
matrix_mul = [[1 for i in range(5)] for j in range(5)]
matrix_mul2 = [[1 for i in range(32)] for j in range(32)]

with open("count_img.pickle", "wb") as file:
	pickle.dump(image_count, file)

with open("matrix_img.pickle", "wb") as file:
	pickle.dump(matrix_count, file)

with open("matrix_img2.pickle", "wb") as file:
	pickle.dump(matrix_count2, file)

with open("matrix_mul.pickle", "wb") as file:
	pickle.dump(matrix_mul, file)

with open("matrix_mul2.pickle", "wb") as file:
	pickle.dump(matrix_mul2, file)