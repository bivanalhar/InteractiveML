from tkinter import *
from PIL import Image, ImageTk

img_bool = True

class Window(Frame):

	#initializing the frame for the canvas
	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master
		self.init_window()

	#creating the window through the function init_window
	def init_window(self):
		#adding the operated button within the window
		self.master.title("Debugging Interface")
		self.master.geometry("500x400")
		self.master.configure(background = "lightblue")

		self.pack(fill = BOTH, expand = 1)
		quitbutton = Button(self, text = "Quit", command = self.client_exit)
		quitbutton.place(x = 0, y = 0)

		imagebutton = Button(self, text = "Show Picture", command = self.showhideImage)
		imagebutton.place(x = 0, y = 30)

		#adding the menu bar within the window
		menu = Menu(self.master)
		self.master.config(menu = menu)

		#adding the File onto the Menu Bar
		file = Menu(menu)
		file.add_command(label = "Exit", command = self.client_exit)
		menu.add_cascade(label = "File", menu = file)

		#adding the Edit onto the Menu Bar
		edit = Menu(menu)
		edit.add_command(label = "Show Image", command = self.showImage)
		edit.add_command(label = "Show Text", command = self.showText)

		menu.add_cascade(label = "Edit", menu = edit)

	def showhideImage(self):
		global img_bool

		if img_bool:
			self.showImage()
		else:
			self.hideImage()

	#defining the operation to open the Image
	def showImage(self):
		kh = Image.open("kingdom_hearts.png")
		render = ImageTk.PhotoImage(kh)

		global img, img_bool

		img = Label(self, image=render)
		img.image = render
		img.place(x = 100, y = 100)

		img_bool = False

	def hideImage(self):
		global img, img_bool

		img.destroy()
		img_bool = True

	def showText(self):
		text = Label(self, text = "Collection of Hearts")
		text.pack()
		text.place(x = 0, y = 370)

	#defining the operation for the quit button
	def client_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()