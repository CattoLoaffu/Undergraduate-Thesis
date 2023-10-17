from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from model import generateSubtitleToFile
from video_prepare import generate
import os

# function
def askDirectory() :
    global rd
    rd = filedialog.askdirectory(title="Select Directory to Store")
    label2 = Label(root,text="Choose Directory to Store temporary chucked sound file :",foreground="green").grid(row = 1, column = 0, sticky = W, pady = 2)

def askVideoFile() :
    global filepath
    vidFile = filedialog.askopenfile(title="Select File")
    filepath = os.path.abspath(vidFile.name)
    label1 = Label(root,text="Choose a video file : ",foreground="green").grid(row = 0, column = 0, sticky = W, pady = 2)

def askTextFile() :
    global textfilepath
    txtFile = filedialog.askopenfile(title="Select File",filetypes=[("text files",".txt")])
    textfilepath = os.path.abspath(txtFile.name)
    label3 = Label(root,text="Choose text file to save subtitle:" ,foreground="green").grid(row = 4, column = 0, sticky = W, pady = 2)

def generateChunckFile() :
    l = Label(root,text="Generating...").grid(row = 3, column = 1, pady = 2)
    generate(rd,filepath)
    com = Tk()
    com.geometry("100x100")
    comlabel = Label(com,text="Complete").pack()

def generateSubtitle() :
    com = Tk()
    com.geometry("600x200")
    runlabel = Label(com,text="Running...").pack()
    label = Label(com,text="This process may use most of RAM as well, recommend to do not run any other program").pack()
    generateSubtitleToFile(rd,filepath,textfilepath)
    comlabel = Label(com,text="Complete").pack()


#label //generate chunck file
if __name__ == "__main__" :
    root = Tk()
    root.title("Subtitle Gen")
    root.geometry("600x300")
    
    # menu
    mainMenu = Menu()
    root.config(menu=mainMenu)
    
    # add menu
    mainMenu.add_cascade(label="File")
    
    label1 = Label(root,text="Choose a video file : ").grid(row = 0, column = 0, sticky = W, pady = 2)
    label2 = Label(root,text="Choose Directory to Store temporary chucked sound file :").grid(row = 1, column = 0, sticky = W, pady = 2)
    btn = Button(root,text="Choose",command=askVideoFile).grid(row = 0, column = 1, pady = 2)
    btn2 = Button(root,text="Choose",command=askDirectory).grid(row = 1, column = 1, pady = 2)
    genbtn = Button(root,text="Generate chunck file",command=generateChunckFile).grid(row = 2, column = 1, pady = 2)
    labels = Label(root,text="" ).grid(row = 3, column = 0, sticky = W, pady = 2)
    
    # label //generate subtitle
    label3 = Label(root,text="Choose text file to save subtitle:" ).grid(row = 4, column = 0, sticky = W, pady = 2)
    btn3 = Button(root,text="Choose",command=askTextFile).grid(row = 4, column = 1, pady = 2)
    subbtn = Button(root,text="Generate subtitle text file",command=generateSubtitle).grid(row = 5, column = 1, pady = 2)
    root.mainloop()

