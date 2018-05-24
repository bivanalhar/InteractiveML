from tkinter import *
from PIL import Image, ImageTk

#the main interface will provide the main screen
#of the program in general. this will just contain the
#2 buttons, one for the doctor and one for the patient

class Window(Frame):

	#initializing the frame of the canvas
	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master

		self.img_bool = True
		self.med_bool = True
		self.stat_bool = True

		self.img_bool_patient = True
		self.med_bool_patient = True
		self.text_bool = True

		self.is_doctor = 0
		self.is_patient = 0

		self.init_doctor = [14, 6] #for the prior probability of the interface outcome
		self.init_patient = [4, 16] #for the prior probability of the interface outcome

		self.init_window()

	#create the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Main")
		self.master.geometry("150x150")

		self.pack(fill = BOTH, expand = 1)

		#the initial window just contains the doctor and patient button
		doctorbutton = Button(self, text = "Doctor Information", command = self.checkDoctor)
		doctorbutton.pack(fill = X)

		patientbutton = Button(self, text = "Patient Information", command = self.checkPatient)
		patientbutton.pack(fill = X)

		exitbutton = Button(self, text = "Exit", command = self.client_exit)
		exitbutton.pack(fill = X)

	def checkDoctor(self):
		probability = self.init_doctor[0] / float(self.init_doctor[0] + self.init_doctor[1])
		print("Doctor : probability of showing doctor is {0:.5f}".format(probability))

		if self.init_doctor[0] > self.init_doctor[1]:
			self.showDoctor()
		else:
			self.showPatient()

	def checkPatient(self):
		probability = self.init_patient[1] / float(self.init_patient[0] + self.init_patient[1])
		print("Patient : probability of showing doctor is {0:.5f}".format(probability))

		if self.init_patient[0] >= self.init_patient[1]:
			self.showDoctor()
		else:
			self.showPatient()

	def showDoctor(self):
		# import doctor_interface
		if self.is_doctor == 1:
			print("The Doctor interface is already opened")
			return 0

		self.is_doctor = 1

		#will show the interface for the Doctor information
		top = self.top = Toplevel(bg = "lightblue")
		top.title("Doctor Interface")

		top.geometry("600x650")

		imagebutton = Button(top, text = "Show Patient Info", command = self.showhideImage)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(top, text = "Show Medical Info", command = self.showhideMedImage)
		medbutton.place(x = 300, y = 10)

		statbutton = Button(top, text = "Show Statistics Info", command = self.showhideStatImage)
		statbutton.place(x = 0, y = 200)

		switchbutton = Button(top, text = "Switch to Patient Interface", command = self.switchPatient)
		switchbutton.place(x = 0, y = 600)

		quitbutton = Button(top, text = "Quit Doctor", command = self.quitDoctor)
		quitbutton.place(x = 300, y = 600)

	def showPatient(self):
		#will show the interface for the Patient information

		if self.is_patient == 1:
			print("The Patient interface is already opened")
			return 0

		self.is_patient = 1

		top = self.top2 = Toplevel(bg = "orange")
		top.title("Patient Interface")

		top.geometry("600x650")

		imagebutton = Button(top, text = "Show Patient Info", command = self.showhideImagePatient)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(top, text = "Show Medical Info", command = self.showhideMedImagePatient)
		medbutton.place(x = 300, y = 10)

		textbutton = Button(top, text = "Show Text Info", command = self.showhideTextImage)
		textbutton.place(x = 0, y = 200)

		switchbutton = Button(top, text = "Switch to Doctor Interface", command = self.switchDoctor)
		switchbutton.place(x = 0, y = 600)

		quitbutton = Button(top, text = "Quit Patient", command = self.quitPatient)
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
	########################END Button 1a###############################
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
	#BEGIN Button 4 : Showing the operation for switching and quitting
	def quitDoctor(self):
		self.is_doctor = 0
		self.top.destroy()

	def quitPatient(self):
		self.is_patient = 0
		self.top2.destroy()

	def switchDoctor(self):
		self.quitPatient()

		self.init_doctor[0] += 1
		self.init_patient[0] += 1
		
		self.showDoctor()

	def switchPatient(self):
		self.quitDoctor()

		self.init_doctor[1] += 1
		self.init_patient[1] += 1

		self.showPatient()
	####################################################################
	########################END Button 4################################
	####################################################################

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()