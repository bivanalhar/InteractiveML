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
	print("initializing\n")
	import init_pickling

	with open("save_tuple.pickle", "rb") as file:
		doc, pat = pickle.load(file)

if init == '0' or init == '':
	print("proceeding\n")

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
		doctorbutton = Button(self, text = "Doctor", command = self.checkDoctor, bg = "lightblue")
		doctorbutton.pack(fill = X)

		patientbutton = Button(self, text = "Patient", command = self.checkPatient, bg = "orange")
		patientbutton.pack(fill = X)

		exitbutton = Button(self, text = "Exit", command = self.client_exit)
		exitbutton.pack(fill = X)

	def checkDoctor(self):
		probability = self.init_complex[0] / float(self.init_complex[0] + self.init_complex[1])
		print("Doctor : probability of showing complex is {0:.2f}".format(probability))

		if self.init_complex[0] > self.init_complex[1]:
			self.showComplex()
		else:
			self.showSimple()

	def checkPatient(self):
		probability = self.init_simple[1] / float(self.init_simple[0] + self.init_simple[1])
		print("Patient : probability of showing simple is {0:.2f}".format(probability))

		if self.init_simple[0] >= self.init_simple[1]:
			self.showComplex()
		else:
			self.showSimple()

	def showComplex(self):
		# import doctor_interface
		if self.is_doctor == 1:
			print("The Doctor interface is already opened")
			return 0

		self.is_doctor = 1

		#will show the interface for the Doctor information
		top = self.top = Toplevel(bg = "lightblue")
		top.title("Complex Explanation Interface")

		top.geometry("1000x650")

		imagebutton = Button(top, text = "Show Patient Info", command = self.showhideImage)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(top, text = "Show Medical Info", command = self.showhideMedImage)
		medbutton.place(x = 300, y = 10)

		statbutton = Button(top, text = "Show Statistics Info", command = self.showhideStatImage)
		statbutton.place(x = 0, y = 200)

		switchbutton = Button(top, text = "Switch to Simple Interface", command = self.switchSimple)
		switchbutton.place(x = 0, y = 600)

		textbutton = Button(top, text = "Show Text Info", command = self.showhideTextImageComplex)
		textbutton.place(x = 600, y = 10)

		quitbutton = Button(top, text = "Quit Complex", command = self.quitComplex)
		quitbutton.place(x = 300, y = 600)

	def showSimple(self):
		#will show the interface for the Patient information

		if self.is_patient == 1:
			print("The Patient interface is already opened")
			return 0

		self.is_patient = 1

		top = self.top2 = Toplevel(bg = "orange")
		top.title("Simple Explanation Interface")

		top.geometry("600x650")

		imagebutton = Button(top, text = "Show Patient Info", command = self.showhideImagePatient)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(top, text = "Show Medical Info", command = self.showhideMedImagePatient)
		medbutton.place(x = 300, y = 10)

		textbutton = Button(top, text = "Show Text Info", command = self.showhideTextImage)
		textbutton.place(x = 0, y = 200)

		switchbutton = Button(top, text = "Switch to Complex Interface", command = self.switchComplex)
		switchbutton.place(x = 0, y = 600)

		quitbutton = Button(top, text = "Quit Simple", command = self.quitSimple)
		quitbutton.place(x = 300, y = 600)

	###################################################################
	#BEGIN Button 1a: Showing the operation for the patients info (Doctor)
	def showhideImage(self):
		if self.img_bool:
			self.showImage()
		else:
			self.hideImage()

	#defining the operation to open the Image
	def showImage(self):
		avatar = Image.open("male_avatar.png")
		avatar = avatar.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img = Label(self.top, image=render)
		self.img.image = render
		self.img.place(x = 0, y = 40)

		self.img_bool = False

	def hideImage(self):
		self.img.destroy()
		self.img_bool = True
	####################################################################
	########################END Button 1a###############################
	####################################################################

	###################################################################
	#BEGIN Button 1b: Showing the operation for the patients info (Patient)
	def showhideImagePatient(self):
		if self.img_bool_patient:
			self.showImagePatient()
		else:
			self.hideImagePatient()

	#defining the operation to open the Image
	def showImagePatient(self):
		avatar = Image.open("male_avatar.png")
		avatar = avatar.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		self.img2 = Label(self.top2, image=render)
		self.img2.image = render
		self.img2.place(x = 0, y = 40)

		self.img_bool_patient = False

	def hideImagePatient(self):
		self.img2.destroy()
		self.img_bool_patient = True
	####################################################################
	########################END Button 1b###############################
	####################################################################

	####################################################################
	#BEGIN Button 2a : Showing the operation for the medical info (Doctor)
	def showhideMedImage(self):
		if self.med_bool:
			self.showMedImage()
		else:
			self.hideMedImage()

	#defining the operation to open the Image
	def showMedImage(self):
		medinfo = Image.open("mnist_info.png")
		medinfo = medinfo.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(medinfo)

		self.med = Label(self.top, image=render)
		self.med.image = render
		self.med.place(x = 300, y = 40)

		self.med_bool = False

	def hideMedImage(self):
		self.med.destroy()
		self.med_bool = True
	####################################################################
	########################END Button 2a###############################
	####################################################################

	####################################################################
	#BEGIN Button 2b : Showing the operation for the medical info (Patient)
	def showhideMedImagePatient(self):
		if self.med_bool_patient:
			self.showMedImagePatient()
		else:
			self.hideMedImagePatient()

	#defining the operation to open the Image
	def showMedImagePatient(self):
		medinfo = Image.open("mnist_info.png")
		medinfo = medinfo.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(medinfo)

		self.med2 = Label(self.top2, image=render)
		self.med2.image = render
		self.med2.place(x = 300, y = 40)

		self.med_bool_patient = False

	def hideMedImagePatient(self):
		self.med2.destroy()
		self.med_bool_patient = True
	####################################################################
	########################END Button 2b###############################
	####################################################################

	####################################################################
	#BEGIN Button 3a : Showing the operation for the statistic info
	def showhideStatImage(self):
		if self.stat_bool:
			self.showStatImage()
		else:
			self.hideStatImage()

	#defining the operation to open the Image
	def showStatImage(self):
		statinfo = Image.open("stat_info.png")
		width, height = statinfo.size

		width = int(width * 0.8)
		height = int(height * 0.8)

		statinfo = statinfo.resize((width, height), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(statinfo)

		self.stat = Label(self.top, image=render)
		self.stat.image = render
		self.stat.place(x = 0, y = 230)

		self.stat_bool = False

	def hideStatImage(self):
		self.stat.destroy()
		self.stat_bool = True
	####################################################################
	########################END Button 3a###############################
	####################################################################

	####################################################################
	#BEGIN Button 3b : Showing the operation for the statistic info
	def showhideTextImage(self):
		if self.text_bool:
			self.showTextImage()
		else:
			self.hideTextImage()

	#defining the operation to open the Image
	def showTextImage(self):
		self.text = Text(self.top2, height = 3, width = 60, font = 15)

		self.text.pack()

		self.text.place(x = 0, y = 230)

		result = "The patient is most likely to have label 0\n"
		result += "the alternative diagnosis may be 6 or 9\n"
		result += "this diagnosis is based on the curvature of the image given"
		self.text.insert(END, result)

		self.text_bool = False

	def hideTextImage(self):
		self.text.delete(1.0, END)
		self.text_bool = True
	####################################################################
	########################END Button 3b###############################
	####################################################################

	####################################################################
	#BEGIN Button 4 : Showing the operation for the text info complex
	def showhideTextImageComplex(self):
		if self.text2_bool:
			self.showTextImageComplex()
		else:
			self.hideTextImageComplex()

	#defining the operation to open the Image
	def showTextImageComplex(self):
		self.text2 = Text(self.top, height = 6, width = 35, font = 12)

		self.text2.pack()

		self.text2.place(x = 600, y = 40)

		result = "Prediction : 0\n"
		result += "Closest Alternative : 6 or 9\n"
		result += "Reason : Picture full of curvature"
		self.text2.insert(END, result)

		self.text2_bool = False

	def hideTextImageComplex(self):
		self.text2.delete(1.0, END)
		self.text2_bool = True
	####################################################################
	########################END Button 3b###############################
	####################################################################

	####################################################################
	#BEGIN Button 5 : Showing the operation for switching and quitting
	def quitComplex(self):
		self.is_doctor = 0
		self.top.destroy()
		
		self.init_complex_var()

	def quitSimple(self):
		self.is_patient = 0
		self.top2.destroy()

		self.init_simple_var()

	def switchComplex(self):
		self.quitSimple()

		self.init_complex[0] += 1
		self.init_simple[0] += 1

		with open("save_tuple.pickle", "wb") as file:
			pickle.dump((self.init_complex, self.init_simple), file)

		self.init_complex_var()
		
		self.showComplex()

	def switchSimple(self):
		self.quitComplex()

		self.init_complex[1] += 1
		self.init_simple[1] += 1

		with open("save_tuple.pickle", "wb") as file:
			pickle.dump((self.init_complex, self.init_simple), file)

		self.init_simple_var()

		self.showSimple()
	####################################################################
	########################END Button 4################################
	####################################################################

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()