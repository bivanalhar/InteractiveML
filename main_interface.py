import pickle
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from lime import lime_image
import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import cifar10_func as cf
import model

import sys
sys.path.append("/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim")

slim = tf.contrib.slim

from nets import inception
from preprocessing import inception_preprocessing

session = tf.Session()
image_size = 32 #the size of the CIFAR-10 images shall be 32x32, so all input should be converted into this size
# image_size = 299

def transform_img_fn(path_list):
	out = []
	for f in path_list:
		file = open(f, 'rb')
		image_raw = tf.image.decode_jpeg(file.read(), channels=3)
		# image = tf.image.resize_images(image_raw, [image_size, image_size])
		# print(session.run(tf.reduce_max(image)), session.run(tf.reduce_min(image)))
		# image = tf.subtract(tf.divide(image, 128), 1)

		image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training = False)
		# image_max, image_min = tf.reduce_max(image), tf.reduce_min(image)
		# print(session.run(image_max), session.run(image_min))

		out.append(image)
	return session.run([out])[0]

# from datasets import imagenet
# names = imagenet.create_readable_names_for_imagenet_labels()
names = cf.build_dictionary_for_cifar10_image()

processed_images = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
# processed_images = tf.placeholder(tf.float32, shape = (None, 299, 299, 3))

logits, _, _, _, _, _ = model.conv_cifar10(processed_images)
probabilities = tf.nn.softmax(logits)

saver = tf.train.Saver()

checkpoints_dir = "./pretrained/cifar10_model.ckpt"
saver.restore(session, "./pretrained/cifar10_model.ckpt")

# import os
# with slim.arg_scope(inception.inception_v3_arg_scope()):
#     logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
# probabilities = tf.nn.softmax(logits)

# checkpoints_dir = '/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim/pretrained'
# init_fn = slim.assign_from_checkpoint_fn(
#     os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
#     slim.get_model_variables('InceptionV3'))
# init_fn(session)

def predict_fn(images):
	# print(session.run(shape, feed_dict = {processed_images : images}))
	return session.run(probabilities, feed_dict={processed_images: images})

from skimage.segmentation import mark_boundaries

# #providing the explanation towards the prediction provided here
# explainer = lime_image.LimeImageExplainer()

"""
This is the borderline between the LIME Image Explainer
and the interface we are about to build on for this application
Later on, we are aiming to integrate the explainer above with
the interface that we have in the bottom of this comment.
"""

#as we will provide the interface for the feedback model
#we will provide the platform in the form of image counting
#and also the matrix that best depicts those feedbacks
with open("count_img.pickle", "rb") as file:
	count_img = pickle.load(file)

with open("matrix_img.pickle", "rb") as file:
	matrix_img = pickle.load(file)

with open("matrix_mul.pickle", "rb") as file:
	matrix_mul = pickle.load(file)

#the main interface will provide the main screen
#of the program in general. this will just contain the
#2 buttons, one for the doctor and one for the patient
with open("save_tuple.pickle", "rb") as file:
	doc, pat = pickle.load(file)

print("the current probability of doctor seeing complex is {0:.2f}".format(float(doc[0])/(doc[0] + doc[1])))
print("the current probability of patient seeing simple is {0:.2f}\n".format(float(pat[1])/(pat[0] + pat[1])))

try:
	init = input("do you want to initialize the probability into 0.7 and 0.8?\n(0)no and (1)yes\n")
	assert (init == '0' or init == '1' or init == '')
except:
	print("Invalid input (the input should be only 0 or 1)")
	raise

if init == '1':
	print("initializing")
	import init_pickling

	with open("save_tuple.pickle", "rb") as file:
		doc, pat = pickle.load(file)

if init == '0' or init == '':
	print("proceeding")

patient_info = [[], [], [], [], [], []]
patient_info[0] += ["Andy Williams", "Bivan Harmanto", "Choi Seungmin", "Alberta Scrubb", "Alisa Kurt", "Kim Eunhye", "Takeshi Gouda", "Mira Strauss", "Shane Oh", "Harold Scrubb"]
patient_info[1] += ["Male", "Male", "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Male"]
patient_info[2] += [35, 22, 27, 50, 19, 21, 39, 27, 25, 53]
patient_info[3] += ["AB Positive", "O Positive", "B Negative", "A Positive", "B Positive", "O Negative", "O Positive", "B Negative", "A Positive", "B Positive"]
patient_info[4] += ["Professor", "Software Engineer", "Counsellor", "Housewife", "Chief Operating Officer", "Student", "Warrior", "Chief Technical Officer", "Student", "Veteran"]
patient_info[5] += ["American", "Indonesian", "Korean", "British", "Germanese", "Korean", "Japanese", "Germanese", "Korean", "British"]

class Window(Frame):

	#initializing the frame of the canvas
	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master

		self.init_complex_var()
		self.init_simple_var()

		#for the prior probability of the interface outcome
		self.init_complex = doc
		self.init_simple = pat

		#providing the platform for the interactivity
		self.count_img = count_img
		self.matrix_img = matrix_img
		self.matrix_mul = matrix_mul

		self.init_window()

	#initializing all the doctor-related variables
	def init_complex_var(self):
		self.img_bool = True
		self.med_bool = True

		#the initialization of the complex interface
		self.stat_bool = True
		self.text_bool = True
		self.explain_bool = True
		self.segments = None

		self.is_doctor = 0

	#initializing all the patient-related variables
	def init_simple_var(self):
		self.img_bool_patient = True
		self.med_bool_patient = True

		#the initialization of the simple interface
		self.simpletext_bool = True
		self.simplexplain_bool = True

		self.is_patient = 0

	#create the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Main")
		self.master.geometry("300x300")

		self.pack(fill = BOTH, expand = 1)

		#the initial window just contains the doctor and patient button
		doctorbutton = Button(self, text = "Doctor", font = ('Helvetica', '16'), command = lambda: self.listPatient(pd_info = 0), bg = "lightblue", height = 2)
		doctorbutton.pack(fill = X)

		patientbutton = Button(self, text = "Patient", font = ('Helvetica', '16'), command = lambda: self.listPatient(pd_info = 1), bg = "orange", height = 2)
		patientbutton.pack(fill = X)

		exitbutton = Button(self, text = "Exit", font = ('Helvetica', '12'), command = self.client_exit)
		exitbutton.pack(fill = X)

	def listPatient(self, pd_info = None):
		if pd_info == None:
			print("Identity not Confirmed : Patient or Doctor?")
			return

		top = self.top = Toplevel(bg = "white")
		top.title("List of Patients")

		top.geometry("300x830")

		button_1 = Button(top, text = "Patient #1\nAndy Williams", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 0), bg = "salmon", height = 3)
		button_1.pack(fill = X)

		button_2 = Button(top, text = "Patient #2\nBivan Alzacky", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 1), bg = "turquoise", height = 3)
		button_2.pack(fill = X)

		button_3 = Button(top, text = "Patient #3\nChoi Seungmin", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 2), bg = "salmon", height = 3)
		button_3.pack(fill = X)

		button_4 = Button(top, text = "Patient #4\nAlberta Scrubb", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 3), bg = "turquoise", height = 3)
		button_4.pack(fill = X)

		button_5 = Button(top, text = "Patient #5\nAlisa Kurt", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 4), bg = "salmon", height = 3)
		button_5.pack(fill = X)

		button_6 = Button(top, text = "Patient #6\nKim Eunhye", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 5), bg = "turquoise", height = 3)
		button_6.pack(fill = X)

		button_7 = Button(top, text = "Patient #7\nTakeshi Gouda", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 6), bg = "salmon", height = 3)
		button_7.pack(fill = X)

		button_8 = Button(top, text = "Patient #8\nMira Strauss", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 7), bg = "turquoise", height = 3)
		button_8.pack(fill = X)

		button_9 = Button(top, text = "Patient #9\nShane Oh", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 8), bg = "salmon", height = 3)
		button_9.pack(fill = X)

		button_10 = Button(top, text = "Patient #10\nHarold Scrubb", font = ('Helvetica', '11'), command = lambda: self.checkScreen(pd_info = pd_info, identity = 9), bg = "turquoise", height = 3)
		button_10.pack(fill = X)

		quitbutton = Button(top, text = "Quit", command = self.quitPatient, bg = "red")
		quitbutton.pack(fill = X)

	def quitPatient(self):
		self.top.destroy()

	def checkScreen(self, pd_info, identity):
		if pd_info == 0:
			probability = self.init_complex[0] / float(self.init_complex[0] + self.init_complex[1])
			print("Doctor : probability of showing complex is {0:.2f}".format(probability))

			if probability > 0.5:
				self.showComplex(id_ = identity)
			else:
				self.showSimple(id_ = identity)
		else:
			probability = self.init_simple[1] / float(self.init_simple[0] + self.init_simple[1])
			print("Patient : probability of showing simple is {0:.2f}".format(probability))

			if probability > 0.5:
				self.showSimple(id_ = identity)
			else:
				self.showComplex(id_ = identity)

	def showComplex(self, id_ = None):
		top = self.top_complex = Toplevel(bg = "lightblue")
		top.title("Complex Explanation Interface")

		top.geometry("950x725")

		self.sp_finish = False
		self.list_sp_outcast = []

		# 1.1) Displaying the Label Image of the Patient
		frame_info = Frame(top, height = 30, width = 200)
		frame_info.pack_propagate(False)
		frame_info.place(x=0, y=0)

		text_info = Text(frame_info, height = 20, width = 30, font = ("Helvetica", 14), bg = "aquamarine")
		text_info.tag_configure("center", justify = "center")
		text_info.insert("1.0", "Patient Photo")
		text_info.tag_add("center", '1.0', 'end')
		text_info.config(state = DISABLED)
		text_info.pack()

		# 1.2) Displaying the Image of the Patient
		avatar = Image.open(str(id_) + "_patinfo.jpeg")
		avatar = avatar.resize((197,197), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img = Label(top, image=render)
		self.img.image = render
		self.img.place(x = 0, y = 30)

		# 2) Displaying the personal information about the Patient
		frame_bio = Frame(top, height = 230, width = 550)
		frame_bio.pack_propagate(False)
		frame_bio.place(x=200, y=0)

		text_bio = Text(frame_bio, height = 20, width = 100, font = ("Helvetica", 16), bg = "gold")
		text_bio_t = " Patient Biodata Information\n"
		text_bio_t += " Name\t\t: " + str(patient_info[0][id_]) + "\n"
		text_bio_t += " Sex\t\t: " + str(patient_info[1][id_]) + "\n"
		text_bio_t += " Age\t\t: " + str(patient_info[2][id_]) + "\n"
		text_bio_t += " Blood Type\t\t: " + str(patient_info[3][id_]) + "\n"
		text_bio_t += " Occupation\t\t: " + str(patient_info[4][id_]) + "\n"
		text_bio_t += " Nationality\t\t: " + str(patient_info[5][id_])

		text_bio.insert(END, text_bio_t)
		text_bio.config(state = DISABLED)
		text_bio.pack()

		# 3) Displaying the Medical Information of the Patient
		frame_info_2 = Frame(top, height = 30, width = 200)
		frame_info_2.pack_propagate(False)
		frame_info_2.place(x=750, y=0)

		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Helvetica", 14), bg = "aquamarine")
		med_info.tag_configure("center", justify = "center")
		med_info.insert("1.0", "Medical Photo")
		med_info.tag_add("center", '1.0', 'end')
		med_info.config(state = DISABLED)
		med_info.pack()

		avatar = Image.open(str(id_) + "_medinfo.jpeg")
		avatar = avatar.resize((197,197), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.med = Label(top, image=render)
		self.med.image = render
		self.med.place(x = 750, y = 30)

		barbutton = Button(top, text = "Stats for Prediction", command = lambda: self.CalcAndShowBarChart(id_ = id_))
		barbutton.place(x = 20, y = 250)

		textbutton = Button(top, text = "Lists of Prediction", command = lambda: self.CalcAndShowTextPred(id_ = id_))
		textbutton.place(x = 345, y = 250)

		explainbutton = Button(top, text = "Explanation for Prediction", command = lambda: self.CalcExplainAndShow(id_ = id_))
		explainbutton.place(x = 675, y = 250)

		switchbutton = Button(top, text = "Switch to Simple Interface", command = lambda: self.switchSimple(id_ = id_), width = 35)
		switchbutton.place(x = 20, y = 640)

		quitbutton = Button(top, text = "Quit Complex", command = self.quitComplex, width = 35)
		quitbutton.place(x = 20, y = 670)

		fixpredbutton = Button(top, text = "Fix the Prediction", command = self.interactiveFixing, width = 34, height = 3)
		fixpredbutton.place(x = 345, y = 640)

	def showSimple(self, id_ = None):
		top = self.top_simple = Toplevel(bg = "wheat")
		top.title("Simple Explanation Interface")

		top.geometry("700x550")

		# 1.1) Displaying the Label Image of the Patient
		frame_info = Frame(top, height = 20, width = 150)
		frame_info.pack_propagate(False)
		frame_info.place(x=0, y=0)

		text_info = Text(frame_info, height = 20, width = 30, font = ("Arial", 10), bg = "aquamarine")
		text_info.tag_configure("center", justify = "center")
		text_info.insert("1.0", "Patient Photo")
		text_info.tag_add("center", '1.0', 'end')
		text_info.config(state = DISABLED)
		text_info.pack()

		# 1.2) Displaying the Image of the Patient
		avatar = Image.open(str(id_) + "_patinfo.jpeg")
		avatar = avatar.resize((147,147), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img = Label(top, image=render)
		self.img.image = render
		self.img.place(x = 0, y = 20)

		# 2) Displaying the personal information about the Patient
		frame_bio = Frame(top, height = 170, width = 400)
		frame_bio.pack_propagate(False)
		frame_bio.place(x = 150, y = 0)

		text_bio = Text(frame_bio, height = 20, width = 100, font = ("Helvetica", 11), bg = "gold")
		text_bio_t = " Patient Biodata Information\n"
		text_bio_t += " Name\t\t: " + str(patient_info[0][id_]) + "\n"
		text_bio_t += " Sex\t\t: " + str(patient_info[1][id_]) + "\n"
		text_bio_t += " Age\t\t: " + str(patient_info[2][id_]) + "\n"
		text_bio_t += " Blood Type\t\t: " + str(patient_info[3][id_]) + "\n"
		text_bio_t += " Occupation\t\t: " + str(patient_info[4][id_]) + "\n"
		text_bio_t += " Nationality\t\t: " + str(patient_info[5][id_])

		text_bio.insert(END, text_bio_t)
		text_bio.config(state = DISABLED)
		text_bio.pack()

		# 3) Displaying the Medical Information of the Patient
		frame_info_2 = Frame(top, height = 20, width = 150)
		frame_info_2.pack_propagate(False)
		frame_info_2.place(x = 550, y = 0)

		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Arial", 10), bg = "aquamarine")
		med_info.tag_configure("center", justify = "center")
		med_info.insert("1.0", "Medical Photo")
		med_info.tag_add("center", '1.0', 'end')
		med_info.config(state = DISABLED)
		med_info.pack()

		avatar = Image.open(str(id_) + "_medinfo.jpeg")
		avatar = avatar.resize((147,147), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.med = Label(top, image=render)
		self.med.image = render
		self.med.place(x = 550, y = 20)

		textbutton = Button(top, text = "Prediction and Alternatives", command = lambda: self.CalcAndSimplyShowText(id_ = id_))
		textbutton.place(x = 0, y = 180)

		explainbutton = Button(top, text = "Explanation for Best Prediction", command = lambda: self.CalcAndSimplyExplain(id_ = id_))
		explainbutton.place(x = 350, y = 180)

		switchbutton = Button(top, text = "Switch to Complex Interface", command = lambda: self.switchComplex(id_ = id_))
		switchbutton.place(x = 0, y = 500)

		quitbutton = Button(top, text = "Quit Complex", command = self.quitSimple)
		quitbutton.place(x = 350, y = 500)

	###############################################################################################
	####### BEGIN : Managing the button functionality for both complex and simple interface #######
	###############################################################################################

	def CalcProbImage(self, id_ = None):
		#do the calculation for the prediction towards the image, using pretrained model
		path_name = str(id_) + '_medinfo.jpeg'
		images = transform_img_fn([path_name])
		self.transformed_image = images[0]

		return predict_fn(images), self.transformed_image

	###################################################################################
	####### BEGIN : Managing the button functionality for the complex interface #######
	###################################################################################

	def CalcAndShowBarChart(self, id_ = None):
		#Step 1 : Do the Calculation for the prediction using the pretrained model
		prediction = self.CalcProbImage(id_ = id_)[0]
		name_list = []
		acc_list = []

		for x in prediction.argsort()[0][-5:]:
			name_list.insert(0, names[x].split(',')[0].capitalize())
			acc_list.insert(0, float(prediction[0, x]))

		data = [go.Bar(
					x = name_list,
					y = acc_list,
					orientation = 'v',
					width = 0.7
		)]
		layout = go.Layout(title = "Stats for Best Prediction", width = 800, height = 640)
		fig = go.Figure(data = data, layout = layout)

		py.image.save_as(fig, filename = str(id_) + "_statchart.jpeg")

		#Step 2 : Show or hide the bar chart (shows the highest 5 prediction accuracy)
		if self.stat_bool:
			self.showBarChart(id_ = id_)
		else:
			self.hideBarChart()

	def showBarChart(self, id_ = None):
		bar_info = Image.open(str(id_) + "_statchart.jpeg")
		bar_info = bar_info.resize((300, 320), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(bar_info)

		self.bar = Label(self.top_complex, image=render)
		self.bar.image = render
		self.bar.place(x = 20, y = 300)

		self.stat_bool = False

	def hideBarChart(self):
		self.bar.destroy()
		self.stat_bool = True

	def CalcAndShowTextPred(self, id_ = None):
		#Step 1 : Do the Calculation for the prediction using the pretrained model
		prediction = self.CalcProbImage(id_ = id_)[0]
		name_list = []
		acc_list = []

		for x in prediction.argsort()[0][-5:]:
			name_list.insert(0, names[x].split(",")[0].capitalize())
			acc_list.insert(0, float(prediction[0, x]))

		#Step 2 : Show or hide the text for showing prediction
		if self.text_bool:
			frame_bio = Frame(self.top_complex, height = 320, width = 300)
			frame_bio.pack_propagate(False)
			frame_bio.place(x = 345, y = 300)

			self.text_list = Text(frame_bio, height = 15, width = 40, font = ("Helvetica", 11), bg = "lightpink")
			text_list_t = ""
			
			for i in range(len(name_list) - 1):
				text_list_t += str(name_list[i]) + "\n" + str(acc_list[i]) + "\n\n"
			text_list_t += str(name_list[-1]) + "\n" + str(acc_list[-1])

			self.text_list.insert(END, text_list_t)
			self.text_list.config(state = DISABLED)
			self.text_list.pack()
			
			self.text_bool = False

		else:
			self.text_list.destroy()
			self.text_bool = True

	def CalcExplainAndShow(self, id_ = None):
		#Step 1 : Do the Calculation for the prediction using the pretrained model
		prediction = self.CalcProbImage(id_ = id_)
		name_list = []
		id_list = []

		for x in prediction[0].argsort()[0][-3:]:
			name_list.insert(0, names[x].split(',')[0].capitalize())
			id_list.insert(0, x)

		#Step 2 : Get the explanation for the prediction
		explainer = lime_image.LimeImageExplainer()

		#Step 3 : Create the button for each of the 5 best predictions
		button_1 = Button(self.top_complex, text = name_list[0], command = lambda: self.ExplainAndShow(name = name_list[0], id_ = id_list[0], explainer = explainer, image = prediction[1]), bg = "turquoise", width = 27)
		button_1.place(x = 675, y = 300)

		button_2 = Button(self.top_complex, text = name_list[1], command = lambda: self.ExplainAndShow(name = name_list[1], id_ = id_list[1], explainer = explainer, image = prediction[1]), bg = "tan", width = 27)
		button_2.place(x = 675, y = 330)

		button_3 = Button(self.top_complex, text = name_list[2], command = lambda: self.ExplainAndShow(name = name_list[2], id_ = id_list[2], explainer = explainer, image = prediction[1]), bg = "turquoise", width = 27)
		button_3.place(x = 675, y = 360)

	def ExplainAndShow(self, name = None, id_ = None, explainer = None, image = None):
		if self.explain_bool or (id_ != self.id_current):
			frame_bio = Frame(self.top_complex, height = 40, width = 245)
			frame_bio.pack_propagate(False)
			frame_bio.place(x = 675, y = 400)

			self.title_pict = Text(frame_bio, height = 2, width = 28, font = ("Helvetica", 15), bg = "yellow")
			text_list_t = name

			self.title_pict.insert(END, text_list_t)
			self.title_pict.tag_configure("center", justify = 'center')
			self.title_pict.tag_add('center', '1.0', 'end')
			self.title_pict.config(state = DISABLED)
			self.title_pict.pack()

			# print(np.max(image), np.min(image))

			explanation, self.segments = explainer.explain_instance_and_get_segments(image, predict_fn, top_labels = 5, hide_color = 0, num_samples = 1000)
			temp, _ = explanation.get_image_and_mask(id_, positive_only = False, num_features = 30, hide_rest = False)
			img_save = mark_boundaries(image = temp / 2 + 0.5, label_img = self.segments, color = (0,0,0))
			plt.imsave(fname = "explain_complex.jpeg", arr = img_save)

			self.id_current = id_

			explain_info = Image.open("explain_complex.jpeg")
			explain_info = explain_info.resize((240, 240), Image.ANTIALIAS)
			render = ImageTk.PhotoImage(explain_info)

			self.explain = Label(self.top_complex, image=render)
			self.explain.image = render
			self.explain.place(x = 675, y = 450)

			self.explain_bool = False

		else:
			self.title_pict.destroy()
			self.explain.destroy()
			self.explain_bool = True

	###################################################################################
	####### BEGIN : Managing the button functionality for the simple interface ########
	###################################################################################

	def CalcAndSimplyShowText(self, id_ = None):
		#Step 1 : Do the Calculation for the prediction using the pretrained model
		prediction = self.CalcProbImage(id_ = id_)[0]
		name_list = []
		acc_list = []

		for x in prediction.argsort()[0][-5:]:
			name_list.insert(0, names[x].split(",")[0].capitalize())
			acc_list.insert(0, float(prediction[0, x]))

		#Step 2 : Show or hide the text for showing prediction
		if self.simpletext_bool:
			frame_bio = Frame(self.top_simple, height = 260, width = 310)
			frame_bio.pack_propagate(False)
			frame_bio.place(x = 0, y = 220)

			if acc_list[0] > 0.7:
				classify_text = "(high confidence)"
				background = "yellow2"
			elif acc_list[0] > 0.4:
				classify_text = "(medium confidence)"
				background = "lightgrey"
			else:
				classify_text = "(low confidence)"
				background = "bisque3"

			self.simpletext_list = Text(frame_bio, height = 15, width = 37, font = ("Helvetica", 11), bg = background)
			text_list_t = "Image most likely refer to\n" + str(name_list[0]).upper() + " " + classify_text
			text_list_t += "\nwith confidence level " + str(100 * acc_list[0])[:10] + " percent\n\n"
			text_list_t += "The alternative prediction is as follows:\n"
			
			for i in range(1, len(name_list) - 1):
				text_list_t += str(name_list[i]) + "\t\t" + str(100 * acc_list[i])[:10] + " percent\n"
			text_list_t += str(name_list[-1]) + "\t\t" + str(100 * acc_list[-1])[:10] + " percent"

			self.simpletext_list.insert(END, text_list_t)
			self.simpletext_list.config(state = DISABLED)
			self.simpletext_list.pack()
			
			self.simpletext_bool = False

		else:
			self.simpletext_list.destroy()
			self.simpletext_bool = True

	def CalcAndSimplyExplain(self, id_ = None):
		#Step 1 : Do the Calculation for the prediction using the pretrained model
		prediction = self.CalcProbImage(id_ = id_)
		name_list = []
		id_list = []

		for x in prediction[0].argsort()[0][-1:]:
			name_list.insert(0, names[x].split(',')[0].capitalize())
			id_list.insert(0, x)

		#Step 2 : Get the explanation for the prediction
		explainer = lime_image.LimeImageExplainer()
		explanation, segments = explainer.explain_instance_and_get_segments(prediction[1], predict_fn, top_labels = 5, hide_color = 0, num_samples = 1000)
		temp, _ = explanation.get_image_and_mask(id_list[0], positive_only = False, num_features = 100, hide_rest = False)

		# img_save = mark_boundaries(image = temp / 2 + 0.5, label_img = mask)
		img_save = mark_boundaries(image = temp / 2 + 0.5, label_img = segments, color = (0,0,0))

		plt.imsave(fname = "explain_simple.jpeg", arr = img_save)

		if self.simplexplain_bool:
			explain_info = Image.open("explain_simple.jpeg")
			explain_info = explain_info.resize((250, 250), Image.ANTIALIAS)
			render = ImageTk.PhotoImage(explain_info)

			self.simplexplain = Label(self.top_simple, image=render)
			self.simplexplain.image = render
			self.simplexplain.place(x = 350, y = 220)

			self.simplexplain_bool = False

		else:
			self.simplexplain.destroy()
			self.simplexplain_bool = True
	
	def interactiveFixing(self):
		if self.segments.all() == None:
			print("Image explanation not here.\nPlease click on the Explaining the Prediction button\nand choose one to be explained")

		else:
			top = self.top_picture = Toplevel(bg = "grey")
			top.title("Editing the Picture")

			top.geometry("300x330")

			canvas = Canvas(self.top_picture, width = 300, height = 300)
			canvas.pack(expand = YES, fill = BOTH)

			open_file = Image.open("explain_complex.jpeg")
			img = ImageTk.PhotoImage(open_file)

			canvas.image = img

			finishbutton = Button(top, text = "Finish the Superpixel Choose", command = self.togglefinish)
			finishbutton.place(x = 0, y = 300)
			finishbutton.pack(fill = X)

			canvas.create_image(0, 0, image = img, anchor = "nw")
			canvas.bind("<Button 1>", self.printcoords)

	def printcoords(self, event):
		#to create the image that responds to the click of the mouse button
		if not self.sp_finish:
			takeout_area = self.segments[event.y, event.x]
			if takeout_area not in self.list_sp_outcast:
				self.list_sp_outcast.append(takeout_area)
			print("Superpixel number #" + str(self.segments[event.y, event.x]))

		else:
			print("already finish choosing the superpixel")	

	def togglefinish(self):
		self.sp_finish = True

		self.count_img += 1
		result_image = np.zeros(np.shape(np.transpose(self.transformed_image, (2, 0, 1))[0]))
		# print(np.shape(result_image))

		print(self.list_sp_outcast)
		for width in range(len(result_image)):
			for height in range(len(result_image[0])):
				if self.segments[width, height] in self.list_sp_outcast:
					result_image[width, height] = 1

		for i in range(5):
			for j in range(5):
				truncate_matrix = [[result_image[k][l] for l in range(4*j, 4*j+16)] for k in range(4*i, 4*i+16)]
				if np.sum(truncate_matrix) >= 128:
					self.matrix_img[i][j] += 1

		print("image counted for feedback is %d" % (self.count_img))
		if self.count_img == 10:
			print("the matrix representing the affected region shall be")
			print(self.matrix_img)
			for i in range(5):
				for j in range(5):
					self.matrix_mul[i][j] -= (self.matrix_mul[i][j] * self.matrix_img[i][j] / 10)
					self.matrix_mul[i][j] += (self.matrix_img[i][j] / 20)
					self.matrix_img[i][j] = 0
			self.count_img = 0
			print(self.matrix_mul)

		with open("count_img.pickle", "wb") as file:
			pickle.dump(self.count_img, file)
		with open("matrix_img.pickle", "wb") as file:
			pickle.dump(self.matrix_img, file)
		with open("matrix_mul.pickle", "wb") as file:
			pickle.dump(self.matrix_mul, file)
		# name_file = input("Put the name of the modified file below\n")
		# plt.imsave(fname = name_file + ".jpeg", arr = result_image)

	####################################################################
	#BEGIN Button 7 : Showing the operation for switching and quitting
	def quitComplex(self):
		self.is_doctor = 0
		self.top_complex.destroy()
		
		self.init_complex_var()

	def quitSimple(self):
		self.is_patient = 0
		self.top_simple.destroy()

		self.init_simple_var()

	def switchComplex(self, id_ = None):
		self.quitSimple()

		self.init_complex[0] += 1
		self.init_simple[0] += 1

		with open("save_tuple.pickle", "wb") as file:
			pickle.dump((self.init_complex, self.init_simple), file)

		self.init_complex_var()
		
		self.showComplex(id_ = id_)

	def switchSimple(self, id_ = None):
		self.quitComplex()

		self.init_complex[1] += 1
		self.init_simple[1] += 1

		with open("save_tuple.pickle", "wb") as file:
			pickle.dump((self.init_complex, self.init_simple), file)

		self.init_simple_var()

		self.showSimple(id_ = id_)
	####################################################################
	########################END Button 7################################
	####################################################################

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()