import pickle
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from lime import lime_image
import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

import sys
sys.path.append("/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim")

slim = tf.contrib.slim

from nets import inception
from preprocessing import inception_preprocessing

session = tf.Session()
image_size = inception.inception_v3.default_image_size

def transform_img_fn(path_list):
	out = []
	for f in path_list:
		file = open(f, 'rb')
		image_raw = tf.image.decode_jpeg(file.read(), channels=3)
		image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
		out.append(image)
	return session.run([out])[0]

from datasets import imagenet
names = imagenet.create_readable_names_for_imagenet_labels()

processed_images = tf.placeholder(tf.float32, shape = (None, 299, 299, 3))

import os
with slim.arg_scope(inception.inception_v3_arg_scope()):
	logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
probabilities = tf.nn.softmax(logits)

checkpoints_dir = '/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim/pretrained'
init_fn = slim.assign_from_checkpoint_fn(
	os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
	slim.get_model_variables('InceptionV3'))
init_fn(session)

def predict_fn(images):
	return session.run(probabilities, feed_dict={processed_images: images})

images1 = transform_img_fn(['cars.jpg'])
images2 = transform_img_fn(['cats.jpg'])

# plt.imshow(images[0] / 2 + 0.5)
preds = predict_fn(images1)
for x in preds.argsort()[0][-5:]:
	print(x, names[x], preds[0,x])

print("\n\n")

preds2 = predict_fn(images2)
for x in preds2.argsort()[0][-5:]:
	print(x, names[x], preds[0,x])

#providing the explanation towards the prediction provided here
explainer = lime_image.LimeImageExplainer()

# #the main interface will provide the main screen
# #of the program in general. this will just contain the
# #2 buttons, one for the doctor and one for the patient
# with open("save_tuple.pickle", "rb") as file:
# 	doc, pat, barpie, ctext, stext = pickle.load(file)

# print("the current probability of doctor seeing complex is {0:.2f}".format(float(doc[0])/(doc[0] + doc[1])))
# print("the current probability of patient seeing simple is {0:.2f}\n".format(float(pat[1])/(pat[0] + pat[1])))

# try:
# 	init = input("do you want to initialize the probability into 0.7 and 0.8?\n(0)no and (1)yes\n")
# 	assert (init == '0' or init == '1' or init == '')
# except:
# 	print("Invalid input (the input should be only 0 or 1)")
# 	raise

# if init == '1':
# 	print("initializing")
# 	import init_pickling

# 	with open("save_tuple.pickle", "rb") as file:
# 		doc, pat, barpie, ctext, stext = pickle.load(file)

# if init == '0' or init == '':
# 	print("proceeding")

# patient_info = [[], [], [], [], [], []]
# patient_info[0] += ["Andy Williams", "Bivan Harmanto", "Choi Seungmin"]
# patient_info[1] += ["Male", "Male", "Male"]
# patient_info[2] += [35, 22, 27]
# patient_info[3] += ["AB Positive", "O Positive", "B Negative"]
# patient_info[4] += ["Professor", "Software Engineer", "Counsellor"]
# patient_info[5] += ["American", "Indonesian", "Korean"]

# class Window(Frame):

# 	#initializing the frame of the canvas
# 	def __init__(self, master = None):
# 		Frame.__init__(self, master)
# 		self.master = master

# 		self.init_complex_var()
# 		self.init_simple_var()

# 		self.init_complex = doc #for the prior probability of the interface outcome
# 		self.init_simple = pat #for the prior probability of the interface outcome
# 		self.init_barpie = barpie #for the prior probability of the bar/pie stats outcome
# 		self.init_ctext = ctext #for the prior probability of the text in complex interface
# 		self.init_stext = stext #for the prior probability of the text in simple interface

# 		self.barpie_bool = None
# 		self.ctext_bool = None
# 		self.stext_bool = None
# 		self.impword_bool = True
# 		self.prior_bool = True

# 		self.init_window()

# 	#initializing all the doctor-related variables
# 	def init_complex_var(self):
# 		self.img_bool = True
# 		self.med_bool = True
# 		self.stat_bool = True
# 		self.text2_bool = True

# 		self.barpie_bool = None
# 		self.ctext_bool = None
# 		self.impword_bool = True
# 		self.prior_bool = True

# 		self.is_doctor = 0

# 	#initializing all the patient-related variables
# 	def init_simple_var(self):
# 		self.img_bool_patient = True
# 		self.med_bool_patient = True
# 		self.text_bool = True
# 		self.stext_bool = None

# 		self.is_patient = 0

# 	#create the window through the function init_window
# 	def init_window(self):
# 		#adding the operated button within the window
# 		self.master.title("Main")
# 		self.master.geometry("150x100")

# 		self.pack(fill = BOTH, expand = 1)

# 		#the initial window just contains the doctor and patient button
# 		doctorbutton = Button(self, text = "Doctor", command = lambda: self.listPatient(pd_info = 0), bg = "lightblue")
# 		doctorbutton.pack(fill = X)

# 		patientbutton = Button(self, text = "Patient", command = lambda: self.listPatient(pd_info = 1), bg = "orange")
# 		patientbutton.pack(fill = X)

# 		exitbutton = Button(self, text = "Exit", command = self.client_exit)
# 		exitbutton.pack(fill = X)

# 	def listPatient(self, pd_info = None):
# 		if pd_info == None:
# 			print("Identity not Confirmed : Patient or Doctor?")
# 			return

# 		top = self.top = Toplevel(bg = "white")
# 		top.title("List of Patients")

# 		top.geometry("150x170")

# 		button_1 = Button(top, text = "Patient #1\nAndy Williams", command = lambda: self.checkScreen(pd_info = pd_info, identity = 0), bg = "salmon")
# 		button_1.pack(fill = X)

# 		button_2 = Button(top, text = "Patient #2\nBivan Alzacky", command = lambda: self.checkScreen(pd_info = pd_info, identity = 1), bg = "turquoise")
# 		button_2.pack(fill = X)

# 		button_3 = Button(top, text = "Patient #3\nChoi Seungmin", command = lambda: self.checkScreen(pd_info = pd_info, identity = 2), bg = "salmon")
# 		button_3.pack(fill = X)

# 		quitbutton = Button(top, text = "Quit", command = self.quitPatient, bg = "red")
# 		quitbutton.pack(fill = X)

# 	def quitPatient(self):
# 		self.top.destroy()

# 	def checkScreen(self, pd_info, identity):
# 		barval = self.init_barpie[0] / float(self.init_barpie[0] + self.init_barpie[1])
# 		ctextval = self.init_ctext[0] / float(self.init_ctext[0] + self.init_ctext[1])
# 		stextval = self.init_stext[0] / float(self.init_stext[0] + self.init_stext[1])

# 		if pd_info == 0:
# 			probability = self.init_complex[0] / float(self.init_complex[0] + self.init_complex[1])
# 			print("Doctor : probability of showing complex is {0:.2f}".format(probability))

# 			if probability > 0.5:
# 				self.showComplex(id_ = identity, barval = barval, ctextval = ctextval, stextval = stextval)
# 			else:
# 				self.showSimple(id_ = identity, barval = barval, ctextval = ctextval, stextval = stextval)
# 		else:
# 			probability = self.init_simple[1] / float(self.init_simple[0] + self.init_simple[1])
# 			print("Patient : probability of showing simple is {0:.2f}".format(probability))

# 			if probability > 0.5:
# 				self.showSimple(id_ = identity, barval = barval, ctextval = ctextval, stextval = stextval)
# 			else:
# 				self.showComplex(id_ = identity, barval = barval, ctextval = ctextval, stextval = stextval)

# 	def showComplex(self, id_ = None, barval = None, ctextval = None, stextval = None):
# 		top = self.top_complex = Toplevel(bg = "lightblue")
# 		top.title("Complex Explanation Interface")

# 		top.geometry("1000x650")

# 		# 1.1) Displaying the Label Image of the Patient
# 		frame_info = Frame(top, height = 30, width = 200)
# 		frame_info.pack_propagate(False)
# 		frame_info.place(x=0, y=0)

# 		text_info = Text(frame_info, height = 20, width = 30, font = ("Helvetica", 16), bg = "aquamarine")
# 		text_info.tag_configure("center", justify = "center")
# 		text_info.insert("1.0", "Patient Photo")
# 		text_info.tag_add("center", '1.0', 'end')
# 		text_info.config(state = DISABLED)
# 		text_info.pack()

# 		# 1.2) Displaying the Image of the Patient
# 		avatar = Image.open(str(id_) + "_patinfo.jpg")
# 		avatar = avatar.resize((197,197), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(avatar)

# 		self.img = Label(top, image=render)
# 		self.img.image = render
# 		self.img.place(x = 0, y = 30)

# 		# 2) Displaying the personal information about the Patient
# 		frame_bio = Frame(top, height = 230, width = 600)
# 		frame_bio.pack_propagate(False)
# 		frame_bio.place(x=200, y=0)

# 		text_bio = Text(frame_bio, height = 20, width = 100, font = ("Helvetica", 16), bg = "gold")
# 		text_bio_t = " Patient Biodata Information\n"
# 		text_bio_t += " Name\t\t: " + str(patient_info[0][id_]) + "\n"
# 		text_bio_t += " Sex\t\t: " + str(patient_info[1][id_]) + "\n"
# 		text_bio_t += " Age\t\t: " + str(patient_info[2][id_]) + "\n"
# 		text_bio_t += " Blood Type\t\t: " + str(patient_info[3][id_]) + "\n"
# 		text_bio_t += " Occupation\t\t: " + str(patient_info[4][id_]) + "\n"
# 		text_bio_t += " Nationality\t\t: " + str(patient_info[5][id_])

# 		text_bio.insert(END, text_bio_t)
# 		text_bio.config(state = DISABLED)
# 		text_bio.pack()

# 		# 3) Displaying the Medical Information of the Patient
# 		frame_info_2 = Frame(top, height = 30, width = 200)
# 		frame_info_2.pack_propagate(False)
# 		frame_info_2.place(x=800, y=0)

# 		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Helvetica", 16), bg = "aquamarine")
# 		med_info.tag_configure("center", justify = "center")
# 		med_info.insert("1.0", "Medical Photo")
# 		med_info.tag_add("center", '1.0', 'end')
# 		med_info.config(state = DISABLED)
# 		med_info.pack()

# 		avatar = Image.open(str(id_) + "_medinfo.jpg")
# 		avatar = avatar.resize((197,197), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(avatar)

# 		self.med = Label(top, image=render)
# 		self.med.image = render
# 		self.med.place(x = 800, y = 30)

# 		statbutton = Button(top, text = "Show Stats Info", command = lambda: self.showhideStatImage(id_ = id_, barval = barval))
# 		statbutton.place(x = 0, y = 250)

# 		switchstatbutton = Button(top, text = "Switch Bar/Pie", command = lambda: self.switchbarpie(id_ = id_))
# 		switchstatbutton.place(x = 150, y = 250)

# 		switchbutton = Button(top, text = "Switch to Simple Interface", command = lambda: self.switchSimple(id_ = id_, barval = barval, ctextval = ctextval, stextval = stextval), width = 35)
# 		switchbutton.place(x = 0, y = 570)

# 		textbutton = Button(top, text = "Show Text Info", command = lambda: self.showhideTextImageComplex(id_ = id_, ctextval = ctextval))
# 		textbutton.place(x = 325, y = 250)

# 		switchtextbutton = Button(top, text = "Switch Comp/Simp", command = lambda: self.switchcomsimp(id_ = id_))
# 		switchtextbutton.place(x = 475, y = 250)

# 		impwordbutton = Button(top, text = "Important Words", command = lambda: self.importantWords(id_ = id_))
# 		impwordbutton.place(x = 675, y = 250)

# 		priorbutton = Button(top, text = "Prior Probability", command = self.PriorProbability)
# 		priorbutton.place(x = 825, y = 250)

# 		quitbutton = Button(top, text = "Quit Complex", command = self.quitComplex, width = 35)
# 		quitbutton.place(x = 0, y = 600)

# 	def showSimple(self, id_ = None, barval = None, ctextval = None, stextval = None):
# 		top = self.top_simple = Toplevel(bg = "wheat")
# 		top.title("Simple Explanation Interface")

# 		top.geometry("700x550")

# 		# 1.1) Displaying the Label Image of the Patient
# 		frame_info = Frame(top, height = 20, width = 150)
# 		frame_info.pack_propagate(False)
# 		frame_info.place(x=0, y=0)

# 		text_info = Text(frame_info, height = 20, width = 30, font = ("Arial", 10), bg = "aquamarine")
# 		text_info.tag_configure("center", justify = "center")
# 		text_info.insert("1.0", "Patient Photo")
# 		text_info.tag_add("center", '1.0', 'end')
# 		text_info.config(state = DISABLED)
# 		text_info.pack()

# 		# 1.2) Displaying the Image of the Patient
# 		avatar = Image.open(str(id_) + "_patinfo.jpg")
# 		avatar = avatar.resize((147,147), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(avatar)

# 		self.img = Label(top, image=render)
# 		self.img.image = render
# 		self.img.place(x = 0, y = 20)

# 		# 2) Displaying the personal information about the Patient
# 		frame_bio = Frame(top, height = 170, width = 400)
# 		frame_bio.pack_propagate(False)
# 		frame_bio.place(x=150, y=0)

# 		text_bio = Text(frame_bio, height = 20, width = 100, font = ("Helvetica", 11), bg = "gold")
# 		text_bio_t = " Patient Biodata Information\n"
# 		text_bio_t += " Name\t\t: " + str(patient_info[0][id_]) + "\n"
# 		text_bio_t += " Sex\t\t: " + str(patient_info[1][id_]) + "\n"
# 		text_bio_t += " Age\t\t: " + str(patient_info[2][id_]) + "\n"
# 		text_bio_t += " Blood Type\t\t: " + str(patient_info[3][id_]) + "\n"
# 		text_bio_t += " Occupation\t\t: " + str(patient_info[4][id_]) + "\n"
# 		text_bio_t += " Nationality\t\t: " + str(patient_info[5][id_])

# 		text_bio.insert(END, text_bio_t)
# 		text_bio.config(state = DISABLED)
# 		text_bio.pack()

# 		# 3) Displaying the Medical Information of the Patient
# 		frame_info_2 = Frame(top, height = 20, width = 150)
# 		frame_info_2.pack_propagate(False)
# 		frame_info_2.place(x=550, y=0)

# 		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Arial", 10), bg = "aquamarine")
# 		med_info.tag_configure("center", justify = "center")
# 		med_info.insert("1.0", "Medical Photo")
# 		med_info.tag_add("center", '1.0', 'end')
# 		med_info.config(state = DISABLED)
# 		med_info.pack()

# 		avatar = Image.open(str(id_) + "_medinfo.jpg")
# 		avatar = avatar.resize((147,147), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(avatar)

# 		self.med = Label(top, image=render)
# 		self.med.image = render
# 		self.med.place(x = 550, y = 20)

# 		textbutton = Button(top, text = "Show Text Info", command = lambda: self.showhideTextImage(id_ = id_, stextval = stextval))
# 		textbutton.place(x = 0, y = 180)

# 		switchtextbutton = Button(top, text = "Switch Simp/Comp", command = lambda: self.switchsimcomp(id_ = id_))
# 		switchtextbutton.place(x = 350, y = 180)

# 		switchbutton = Button(top, text = "Switch to Complex Interface", command = lambda: self.switchComplex(id_ = id_, barval = barval, ctextval = ctextval, stextval = stextval))
# 		switchbutton.place(x = 0, y = 500)

# 		quitbutton = Button(top, text = "Quit Complex", command = self.quitSimple)
# 		quitbutton.place(x = 350, y = 500)

# 	####################################################################
# 	#BEGIN Button 3a : Showing the operation for the statistic info
# 	def showhideStatImage(self, id_ = None, barval = None):
# 		if self.stat_bool:
# 			if barval > 0.5:
# 				self.showStatImage(id_ = id_, barpie = 0)
# 				self.barpie_bool = True
# 			else:
# 				self.showStatImage(id_ = id_, barpie = 1)
# 				self.barpie_bool = False
# 		else:
# 			self.hideStatImage()
# 			self.barpie_bool = None

# 	#defining the operation to open the Image
# 	def showStatImage(self, id_ = None, barpie = None):
# 		if barpie == 0:
# 			statinfo = Image.open(str(id_) + "_statinfo.jpg")
# 		else:
# 			statinfo = Image.open(str(id_) + "_pieinfo.jpg")

# 		statinfo = statinfo.resize((310, 200), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(statinfo)

# 		self.stat = Label(self.top_complex, image=render)
# 		self.stat.image = render
# 		self.stat.place(x = 0, y = 280)

# 		self.stat_bool = False

# 	def hideStatImage(self):
# 		self.stat.destroy()
# 		self.stat_bool = True

# 	def switchbarpie(self, id_ = None):
# 		if self.barpie_bool == None:
# 			print("statistics image not yet shown")

# 		elif self.barpie_bool:
# 			self.stat.destroy()
# 			self.showStatImage(id_ = id_, barpie = 1)
# 			self.init_barpie[1] += 1
# 			self.barpie_bool = False

# 		else:
# 			self.stat.destroy()
# 			self.showStatImage(id_ = id_, barpie = 0)
# 			self.init_barpie[0] += 1
# 			self.barpie_bool = True

# 		with open("save_tuple.pickle", "wb") as file:
# 			pickle.dump((self.init_complex, self.init_simple, self.init_barpie, self.init_ctext, self.init_stext), file)
# 	####################################################################
# 	########################END Button 3a###############################
# 	####################################################################

# 	####################################################################
# 	#BEGIN Button 3b : Showing the operation for the statistic info
# 	def showhideTextImage(self, id_ = None, stextval = None):
# 		if self.text_bool:
# 			if stextval > 0.5:
# 				self.showTextImage(id_ = id_, simcomp = 1)
# 				self.stext_bool = True
# 			else:
# 				self.showTextImage(id_ = id_, simcomp = 0)
# 				self.stext_bool = False
# 		else:
# 			self.hideTextImage()
# 			self.stext_bool = None

# 	#defining the operation to open the Image
# 	def showTextImage(self, id_ = None, simcomp = None):
# 		self.text = Text(self.top_simple, height = 12, width = 85, font = ("Helvetica", 11))

# 		self.text.pack()

# 		self.text.place(x = 0, y = 210)

# 		if simcomp == 0:
# 			file_open = open(str(id_) + "_text_patient.txt", 'r')
# 		else:
# 			file_open = open(str(id_) + "_textinfo.txt", "r")

# 		file_line = file_open.readlines()

# 		result = ''

# 		for line in file_line:
# 			result += line

# 		self.text.insert(END, result)
# 		self.text.config(state = DISABLED)

# 		self.text_bool = False

# 	def hideTextImage(self):
# 		self.text.destroy()
# 		self.text_bool = True

# 	def switchsimcomp(self, id_ = None):
# 		if self.stext_bool == None:
# 			print("text complex interface image not yet shown")

# 		elif self.stext_bool:
# 			self.text.destroy()
# 			self.showTextImage(id_ = id_, simcomp = 0)
# 			self.init_stext[1] += 1
# 			self.stext_bool = False

# 		else:
# 			self.text.destroy()
# 			self.showTextImage(id_ = id_, simcomp = 1)
# 			self.init_stext[0] += 1
# 			self.stext_bool = True

# 		with open("save_tuple.pickle", "wb") as file:
# 			pickle.dump((self.init_complex, self.init_simple, self.init_barpie, self.init_ctext, self.init_stext), file)
# 	####################################################################
# 	########################END Button 3b###############################
# 	####################################################################

# 	####################################################################
# 	#BEGIN Button 4 : Showing the operation for the text info complex
# 	def showhideTextImageComplex(self, id_ = None, ctextval = None):
# 		if self.text2_bool:
# 			if ctextval > 0.5:
# 				self.showTextImageComplex(id_ = id_, comsimp = 0)
# 				self.ctext_bool = True
# 			else:
# 				self.showTextImageComplex(id_ = id_, comsimp = 1)
# 				self.ctext_bool = False
# 		else:
# 			self.hideTextImageComplex()
# 			self.ctext_bool = None

# 	#defining the operation to open the Image
# 	def showTextImageComplex(self, id_ = None, comsimp = None):
# 		self.text2 = Text(self.top_complex, height = 17, width = 45, font = ("Helvetica", 10))

# 		self.text2.pack()

# 		self.text2.place(x = 325, y = 280)

# 		if comsimp == 0:
# 			file_open = open(str(id_) + "_textinfo.txt", 'r')
# 		else:
# 			file_open = open(str(id_) + "_text_patient.txt", "r")

# 		file_line = file_open.readlines()

# 		result = ''
# 		for line in file_line:
# 			result += line

# 		self.text2.insert(END, result)
# 		self.text2.config(state = DISABLED)

# 		self.text2_bool = False

# 	def hideTextImageComplex(self):
# 		self.text2.destroy()
# 		self.text2_bool = True

# 	def switchcomsimp(self, id_ = None):
# 		if self.ctext_bool == None:
# 			print("text complex interface image not yet shown")

# 		elif self.ctext_bool:
# 			self.text2.destroy()
# 			self.showTextImageComplex(id_ = id_, comsimp = 1)
# 			self.init_ctext[1] += 1
# 			self.ctext_bool = False

# 		else:
# 			self.text2.destroy()
# 			self.showTextImageComplex(id_ = id_, comsimp = 0)
# 			self.init_ctext[0] += 1
# 			self.ctext_bool = True

# 		with open("save_tuple.pickle", "wb") as file:
# 			pickle.dump((self.init_complex, self.init_simple, self.init_barpie, self.init_ctext, self.init_stext), file)
# 	####################################################################
# 	########################END Button 4################################
# 	####################################################################

# 	####################################################################
# 	#BEGIN Button 5 : Showing the operation for the important words
# 	def importantWords(self, id_ = None):
# 		if self.impword_bool:
# 			self.showImportantWords(id_ = id_)
# 		else:
# 			self.hideImportantWords()

# 	#defining the operation to open the Image
# 	def showImportantWords(self, id_ = None):
# 		self.impword = Text(self.top_complex, height = 6, width = 35, font = ("Helvetica", 11))
# 		self.impword.pack()
# 		self.impword.place(x = 675, y = 280)

# 		file_open = open(str(id_) + "_impwords.txt", "r")
# 		file_line = file_open.readlines()

# 		result = ''

# 		for line in file_line:
# 			result += line

# 		self.impword.insert(END, result)
# 		self.impword.config(state = DISABLED)

# 		self.impword_bool = False

# 	def hideImportantWords(self):
# 		self.impword.destroy()
# 		self.impword_bool = True
# 	####################################################################
# 	########################END Button 5################################
# 	####################################################################

# 	####################################################################
# 	#BEGIN Button 6 : Showing the operation for the prior probability
# 	def PriorProbability(self):
# 		if self.prior_bool:
# 			self.showPriorProbability()
# 		else:
# 			self.hidePriorProbability()

# 	#defining the operation to open the Image
# 	def showPriorProbability(self):
# 		comp_val = self.init_complex[0] / float(self.init_complex[0] + self.init_complex[1])
# 		simp_val = self.init_simple[1] / float(self.init_simple[0] + self.init_simple[1])
# 		barval = self.init_barpie[0] / float(self.init_barpie[0] + self.init_barpie[1])
# 		ctextval = self.init_ctext[0] / float(self.init_ctext[0] + self.init_ctext[1])
# 		stextval = self.init_stext[0] / float(self.init_stext[0] + self.init_stext[1])
		
# 		data = [go.Bar(
# 					x = [comp_val, simp_val, barval, ctextval, stextval],
# 					y = ["Doctor Showing Complex", "Patient Showing Simple", "Showing Bar-Pie Chart", "Doctor Showing Complex Text", "Patient Showing Simple Text"],
# 					orientation = 'h',
# 					width = 0.6
# 		)]
# 		layout = go.Layout(title = "Prior Probabilities", width = 800, height = 640)
# 		fig = go.Figure(data = data, layout = layout)

# 		py.image.save_as(fig, filename = "prior-chart.jpeg")

# 		prior_info = Image.open("prior-chart.jpeg")
# 		prior_info = prior_info.resize((285, 200), Image.ANTIALIAS)
# 		render = ImageTk.PhotoImage(prior_info)

# 		self.prior = Label(self.top_complex, image=render)
# 		self.prior.image = render
# 		self.prior.place(x = 675, y = 420)

# 		self.prior_bool = False

# 	def hidePriorProbability(self):
# 		self.prior.destroy()
# 		self.prior_bool = True
# 	####################################################################
# 	########################END Button 6################################
# 	####################################################################

# 	####################################################################
# 	#BEGIN Button 7 : Showing the operation for switching and quitting
# 	def quitComplex(self):
# 		self.is_doctor = 0
# 		self.top_complex.destroy()
		
# 		self.init_complex_var()

# 	def quitSimple(self):
# 		self.is_patient = 0
# 		self.top_simple.destroy()

# 		self.init_simple_var()

# 	def switchComplex(self, id_ = None, barval = None, ctextval = None, stextval = None):
# 		self.quitSimple()

# 		self.init_complex[0] += 1
# 		self.init_simple[0] += 1

# 		with open("save_tuple.pickle", "wb") as file:
# 			pickle.dump((self.init_complex, self.init_simple, self.init_barpie, self.init_ctext, self.init_stext), file)

# 		self.init_complex_var()
		
# 		self.showComplex(id_ = id_, barval = barval, ctextval = ctextval, stextval = stextval)

# 	def switchSimple(self, id_ = None, barval = None, ctextval = None, stextval = None):
# 		self.quitComplex()

# 		self.init_complex[1] += 1
# 		self.init_simple[1] += 1

# 		with open("save_tuple.pickle", "wb") as file:
# 			pickle.dump((self.init_complex, self.init_simple, self.init_barpie, self.init_ctext, self.init_stext), file)

# 		self.init_simple_var()

# 		self.showSimple(id_ = id_, barval = barval, ctextval = ctextval, stextval = stextval)
# 	####################################################################
# 	########################END Button 7################################
# 	####################################################################

# 	#defining the operation for the quit button
# 	def client_exit(self):
# 		exit()

# root = Tk()

# app = Window(root)
# root.mainloop()