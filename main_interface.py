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

		self.init_window()

	#create the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Main")
		self.master.geometry("150x150")

		self.pack(fill = BOTH, expand = 1)

		#the initial window just contains the doctor and patient button
		doctorbutton = Button(self, text = "Doctor Information", command = self.showDoctor)
		doctorbutton.pack(fill = X)

		patientbutton = Button(self, text = "Patient Information", command = self.showPatient)
		patientbutton.pack(fill = X)

		exitbutton = Button(self, text = "Exit", command = self.client_exit)
		exitbutton.pack(fill = X)

	def showDoctor(self):
		# import doctor_interface

		#will show the interface for the Doctor information
		top = self.top = Toplevel()
		top.title("Doctor Interface")

		top.geometry("600x650")

		imagebutton = Button(top, text = "Show Patient Info", command = self.showhideImage)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(top, text = "Show Medical Info", command = self.showhideMedImage)
		medbutton.place(x = 300, y = 10)

		statbutton = Button(top, text = "Show Statistics Info", command = self.showhideStatImage)
		statbutton.place(x = 0, y = 200)

	def showPatient(self):
		#will show the interface for the Patient information
		pass

	###################################################################
	#BEGIN Button 1: Showing the operation for the patients info
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
	########################END Button 1################################
	####################################################################

	####################################################################
	#BEGIN Button 2 : Showing the operation for the medical info
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
	########################END Button 2################################
	####################################################################

	####################################################################
	#BEGIN Button 3 : Showing the operation for the statistic info
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
	########################END Button 3################################
	####################################################################

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()
