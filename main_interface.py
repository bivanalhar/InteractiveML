import pickle
from tkinter import *
from PIL import Image, ImageTk

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
patient_info[0] += ["Andy Williams", "Bivan Harmanto", "Choi Seungmin"]
patient_info[1] += ["Male", "Male", "Male"]
patient_info[2] += [35, 22, 27]
patient_info[3] += ["AB Positive", "O Positive", "B Negative"]
patient_info[4] += ["Professor", "Software Engineer", "Counsellor"]
patient_info[5] += ["American", "Indonesian", "Korean"]

class Window(Frame):

	#initializing the frame of the canvas
	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master

		self.init_complex_var()
		self.init_simple_var()

		self.init_complex = doc #for the prior probability of the interface outcome
		self.init_simple = pat #for the prior probability of the interface outcome

		self.init_window()

	#initializing all the doctor-related variables
	def init_complex_var(self):
		self.img_bool = True
		self.med_bool = True
		self.stat_bool = True
		self.text2_bool = True

		self.is_doctor = 0

	#initializing all the patient-related variables
	def init_simple_var(self):
		self.img_bool_patient = True
		self.med_bool_patient = True
		self.text_bool = True

		self.is_patient = 0

	#create the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Main")
		self.master.geometry("150x100")

		self.pack(fill = BOTH, expand = 1)

		#the initial window just contains the doctor and patient button
		doctorbutton = Button(self, text = "Doctor", command = lambda: self.listPatient(pd_info = 0), bg = "lightblue")
		doctorbutton.pack(fill = X)

		patientbutton = Button(self, text = "Patient", command = lambda: self.listPatient(pd_info = 1), bg = "orange")
		patientbutton.pack(fill = X)

		exitbutton = Button(self, text = "Exit", command = self.client_exit)
		exitbutton.pack(fill = X)

	def listPatient(self, pd_info = None):
		if pd_info == None:
			print("Identity not Confirmed : Patient or Doctor?")
			return

		top = self.top = Toplevel(bg = "white")
		top.title("List of Patients")

		top.geometry("150x170")

		button_1 = Button(top, text = "Patient #1\nAndy Williams", command = lambda: self.checkScreen(pd_info = pd_info, identity = 0), bg = "salmon")
		button_1.pack(fill = X)

		button_2 = Button(top, text = "Patient #2\nBivan Alzacky", command = lambda: self.checkScreen(pd_info = pd_info, identity = 1), bg = "turquoise")
		button_2.pack(fill = X)

		button_3 = Button(top, text = "Patient #3\nChoi Seungmin", command = lambda: self.checkScreen(pd_info = pd_info, identity = 2), bg = "salmon")
		button_3.pack(fill = X)

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

		top.geometry("1000x650")

		# 1.1) Displaying the Label Image of the Patient
		frame_info = Frame(top, height = 30, width = 200)
		frame_info.pack_propagate(False)
		frame_info.place(x=0, y=0)

		text_info = Text(frame_info, height = 20, width = 30, font = ("Arial", 16), bg = "aquamarine")
		text_info.tag_configure("center", justify = "center")
		text_info.insert("1.0", "Patient Photo")
		text_info.tag_add("center", '1.0', 'end')
		text_info.config(state = DISABLED)
		text_info.pack()

		# 1.2) Displaying the Image of the Patient
		avatar = Image.open(str(id_) + "_patinfo.jpg")
		avatar = avatar.resize((197,197), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img = Label(top, image=render)
		self.img.image = render
		self.img.place(x = 0, y = 30)

		# 2) Displaying the personal information about the Patient
		frame_bio = Frame(top, height = 230, width = 600)
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
		frame_info_2.place(x=800, y=0)

		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Arial", 16), bg = "aquamarine")
		med_info.tag_configure("center", justify = "center")
		med_info.insert("1.0", "Medical Photo")
		med_info.tag_add("center", '1.0', 'end')
		med_info.config(state = DISABLED)
		med_info.pack()

		avatar = Image.open(str(id_) + "_medinfo.jpg")
		avatar = avatar.resize((197,197), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.med = Label(top, image=render)
		self.med.image = render
		self.med.place(x = 800, y = 30)

		statbutton = Button(top, text = "Show Statistics Info", command = lambda: self.showhideStatImage(id_ = id_), bg = "sandybrown")
		statbutton.place(x = 0, y = 250)

		switchbutton = Button(top, text = "Switch to Simple Interface", command = lambda: self.switchSimple(id_ = id_))
		switchbutton.place(x = 0, y = 600)

		textbutton = Button(top, text = "Show Text Info", command = lambda: self.showhideTextImageComplex(id_ = id_))
		textbutton.place(x = 500, y = 250)

		quitbutton = Button(top, text = "Quit Complex", command = self.quitComplex)
		quitbutton.place(x = 500, y = 600)

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
		avatar = Image.open(str(id_) + "_patinfo.jpg")
		avatar = avatar.resize((147,147), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img = Label(top, image=render)
		self.img.image = render
		self.img.place(x = 0, y = 20)

		# 2) Displaying the personal information about the Patient
		frame_bio = Frame(top, height = 170, width = 400)
		frame_bio.pack_propagate(False)
		frame_bio.place(x=150, y=0)

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
		frame_info_2.place(x=550, y=0)

		med_info = Text(frame_info_2, height = 20, width = 30, font = ("Arial", 10), bg = "aquamarine")
		med_info.tag_configure("center", justify = "center")
		med_info.insert("1.0", "Medical Photo")
		med_info.tag_add("center", '1.0', 'end')
		med_info.config(state = DISABLED)
		med_info.pack()

		avatar = Image.open(str(id_) + "_medinfo.jpg")
		avatar = avatar.resize((147,147), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.med = Label(top, image=render)
		self.med.image = render
		self.med.place(x = 550, y = 20)

		textbutton = Button(top, text = "Show Text Info", command = lambda: self.showhideTextImage(id_ = id_))
		textbutton.place(x = 0, y = 180)

		switchbutton = Button(top, text = "Switch to Complex Interface", command = lambda: self.switchComplex(id_ = id_))
		switchbutton.place(x = 0, y = 500)

		quitbutton = Button(top, text = "Quit Complex", command = self.quitSimple)
		quitbutton.place(x = 350, y = 500)

	####################################################################
	#BEGIN Button 3a : Showing the operation for the statistic info
	def showhideStatImage(self, id_ = None):
		if self.stat_bool:
			self.showStatImage(id_ = id_)
		else:
			self.hideStatImage()

	#defining the operation to open the Image
	def showStatImage(self, id_ = None):
		statinfo = Image.open(str(id_) + "_statinfo.jpg")
		width, height = statinfo.size

		width = int(width * 0.8)
		height = int(height * 0.83)

		statinfo = statinfo.resize((width, height), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(statinfo)

		self.stat = Label(self.top_complex, image=render)
		self.stat.image = render
		self.stat.place(x = 0, y = 280)

		self.stat_bool = False

	def hideStatImage(self):
		self.stat.destroy()
		self.stat_bool = True
	####################################################################
	########################END Button 3a###############################
	####################################################################

	####################################################################
	#BEGIN Button 3b : Showing the operation for the statistic info
	def showhideTextImage(self, id_ = None):
		if self.text_bool:
			self.showTextImage(id_ = id_)
		else:
			self.hideTextImage()

	#defining the operation to open the Image
	def showTextImage(self, id_ = None):
		self.text = Text(self.top_simple, height = 10, width = 75, font = 12)

		self.text.pack()

		self.text.place(x = 0, y = 210)
		file_open = open(str(id_) + "_text_patient.txt", 'r')
		file_line = file_open.readlines()

		result = ''

		for line in file_line:
			result += line

		self.text.insert(END, result)
		self.text.config(state = DISABLED)

		self.text_bool = False

	def hideTextImage(self):
		self.text.destroy()
		self.text_bool = True
	####################################################################
	########################END Button 3b###############################
	####################################################################

	####################################################################
	#BEGIN Button 4 : Showing the operation for the text info complex
	def showhideTextImageComplex(self, id_ = None):
		if self.text2_bool:
			self.showTextImageComplex(id_ = id_)
		else:
			self.hideTextImageComplex()

	#defining the operation to open the Image
	def showTextImageComplex(self, id_ = None):
		self.text2 = Text(self.top_complex, height = 15, width = 45, font = 12)

		self.text2.pack()

		self.text2.place(x = 500, y = 280)
		file_open = open(str(id_) + "_textinfo.txt", 'r')
		file_line = file_open.readlines()

		result = ''
		for line in file_line:
			result += line

		self.text2.insert(END, result)
		self.text2.config(state = DISABLED)

		self.text2_bool = False

	def hideTextImageComplex(self):
		self.text2.destroy()
		self.text2_bool = True
	####################################################################
	########################END Button 4################################
	####################################################################

	# ####################################################################
	# #BEGIN Button 5 : Showing the operation for switching and quitting
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
	# ####################################################################
	# ########################END Button 4################################
	# ####################################################################

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()