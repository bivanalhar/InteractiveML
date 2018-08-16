from tkinter import *

class Window(Frame):

	def __init__(self, master = None):
		Frame.__init__(self, master)
		self.master = master

		self.init_window()

	def init_window(self):
		self.master.title("Main")
		self.master.geometry("200x150")

		self.pack(fill = BOTH, expand = 1)

		button_exit = Button(self, text = "Exit Application", command = self.top_exit, height = 2, font = ('Arial', '16'))
		button_exit.pack(fill = X)

	def top_exit(self):
		exit()

root = Tk()

app = Window(root)
root.mainloop()