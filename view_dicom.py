import matplotlib.pyplot as plt
import pydicom
import os, csv, pickle
import shutil
import numpy as np

RootDir = r'CBIS-DDSM'
TargetDir = r'Mass-Train'

for root, dirs, files in os.walk((os.path.normpath(RootDir)), topdown = False):
	# if "/" in root and "_" in root:
	# 	file_class = root.split('/')[1].split('_')[0]
	# 	file_num = root.split('/')[1].split('_')[2] 
	# 	file_side = root.split('/')[1].split('_')[3]
	# 	file_type = root.split('/')[1].split('_')[4]
		
	# 	if len(files) > 0:
	# 		new_name = file_class + '_' + file_num + '_' + file_side + '_' + file_type + "_" + files[0]
	# 		newfile = os.path.join(root, new_name)
	# 		shutil.move(os.path.join(root, files[0]), newfile)

	for name in files:
		source_folder = os.path.join(root, name)
		shutil.copy2(source_folder, TargetDir)

# for file in os.listdir('Calc-Train'):
# 	name_file = 'Calc-Train/' + file
# 	ds = pydicom.read_file(name_file)
# 	print(int(ds.Rows) / int(ds.Columns))

# list_classify = []
# with open('calc_case_description_train_set.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file)
# 	for row in csv_reader:
# 		stored_row = [row[0][2:], row[2], row[3], row[9]]
# 		if stored_row not in list_classify:
# 			list_classify.append(stored_row)

# with open("calc-train.pickle", "wb") as file:
# 	pickle.dump(list_classify, file)