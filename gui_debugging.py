from tkinter import *
from PIL import Image, ImageTk
import mnist_classifier

img_bool = True
med_bool = True
stat_bool = True

class Window(Frame):

	#initializing the frame for the canvas
	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master
		self.init_window()

	#creating the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Medical Information")
		self.master.geometry("600x650")
		self.master.configure(background = "lightblue")

		self.pack(fill = BOTH, expand = 1)

		quitbutton = Button(self, text = "Quit", command = self.client_exit)
		quitbutton.place(x = 550, y = 620)

		imagebutton = Button(self, text = "Show Patient Info", command = self.showhideImage)
		imagebutton.place(x = 0, y = 10)

		medbutton = Button(self, text = "Show Medical Info", command = self.showhideMedImage)
		medbutton.place(x = 300, y = 10)

		statbutton = Button(self, text = "Show Statistics Info", command = self.showhideStatImage)
		statbutton.place(x = 0, y = 200)

		classifybut = Button(self, text = "Classify", command = self.classify)
		classifybut.place(x = 300, y = 200)

		# #adding the menu bar within the window
		# menu = Menu(self.master)
		# self.master.config(menu = menu)

		# #adding the File onto the Menu Bar
		# file = Menu(menu)
		# file.add_command(label = "Exit", command = self.client_exit)
		# menu.add_cascade(label = "File", menu = file)

		# #adding the Edit onto the Menu Bar
		# edit = Menu(menu)
		# edit.add_command(label = "Show Image", command = self.showImage)
		# edit.add_command(label = "Show Text", command = self.showText)

		# menu.add_cascade(label = "Edit", menu = edit)

	###################################################################
	#BEGIN Button 1: Showing the operation for the patients info
	def showhideImage(self):
		global img_bool

		if img_bool:
			self.showImage()
		else:
			self.hideImage()

	#defining the operation to open the Image
	def showImage(self):
		avatar = Image.open("male_avatar.png")
		avatar = avatar.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(avatar)

		global img, img_bool

		img = Label(self, image=render)
		img.image = render
		img.place(x = 0, y = 40)

		img_bool = False

	def hideImage(self):
		global img, img_bool

		img.destroy()
		img_bool = True

	####################################################################
	########################END Button 1################################
	####################################################################

	####################################################################
	#BEGIN Button 2 : Showing the operation for the medical info
	def showhideMedImage(self):
		global med_bool

		if med_bool:
			self.showMedImage()
		else:
			self.hideMedImage()

	#defining the operation to open the Image
	def showMedImage(self):
		medinfo = Image.open("mnist_info.png")
		medinfo = medinfo.resize((150, 150), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(medinfo)

		global med, med_bool

		med = Label(self, image=render)
		med.image = render
		med.place(x = 300, y = 40)

		med_bool = False

	def hideMedImage(self):
		global med, med_bool

		med.destroy()
		med_bool = True

	####################################################################
	########################END Button 2################################
	####################################################################

	####################################################################
	#BEGIN Button 3 : Showing the operation for the statistic info
	def showhideStatImage(self):
		global stat_bool

		if stat_bool:
			self.showStatImage()
		else:
			self.hideStatImage()

	#defining the operation to open the Image
	def showStatImage(self):
		statinfo = Image.open("stat_info.png")
		# statinfo = statinfo.resize((288, 174), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(statinfo)

		global stat, stat_bool

		stat = Label(self, image=render)
		stat.image = render
		stat.place(x = 0, y = 230)

		stat_bool = False

	def hideStatImage(self):
		global stat, stat_bool

		stat.destroy()
		stat_bool = True

	####################################################################
	########################END Button 3################################
	####################################################################

	####################################################################
	#BEGIN Button 4 : Classification Button
	def classify(self):
		result = mnist_classifier.classifier()

		global top

		top = Toplevel()
		top.title("Classification Result")

		res_text = "Accuracy\n" + str(result)[:5] + "\npercent"  

		msg = Message(top, text = res_text)
		msg.config(font = ("Arial", 20), justify = CENTER)
		msg.pack()

		top.geometry("150x150")

		quitchildbut = Button(top, text = "Quit", command = self.child_destroy)
		quitchildbut.place(x = 50, y = 120)

	####################################################################
	########################END Button 4################################
	####################################################################

	# def showText(self):
	# 	text = Label(self, text = "Collection of Hearts")
	# 	text.pack()
	# 	text.place(x = 0, y = 370)

	#defining the operation for the quit button
	def client_exit(self):
		exit()

	def child_destroy(self):
		global top
		top.destroy()

root = Tk()

app = Window(root)
root.mainloop()