import pickle
import numpy as np

image_count = 1
matrix_mul = np.array([[1.0 for i in range(5)] for j in range(5)])

with open("matrix_mul_and_count.p", "wb") as file:
	pickle.dump([matrix_mul, image_count], file)