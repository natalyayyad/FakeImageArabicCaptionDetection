from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import labeler

class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("Real/Fake Detector - NLP Project")
        self.minsize(960,300)
        self.configure(background='#80c1ff')
        self.resizable(False, False)
        self.path = StringVar()

        #TITLE IN CENTER
        self.label1 = tk.Label(self, text="REAL/FAKE DETECTOR",bg='#80c1ff')
        self.label1.config(font=("Courier 18 bold"))
        self.label1.place(y=14,x=350)

        self.label3 = tk.Label(self, text="Caption in Arabic",bg='#80c1ff')
        self.label3.config(font=("Courier 12 bold"))
        self.label3.place(y=60,x=30)

        self.txt = tk.Text(self, width=80, height=2.5)
        self.txt.place(y=60, x=250)

        # create a Scrollbar and associate it with entry
        self.scrollb = tk.Scrollbar(self, command=self.txt.yview, orient=VERTICAL, width=20)
        self.scrollb.place(y=60, x=895)
        self.txt['yscrollcommand'] = self.scrollb.set

        #Browse for a picture
        self.selectlabel = tk.Label(self, text="Select Picture", bg='#80c1ff')
        self.selectlabel.config(font=("Courier 12 bold"))
        self.selectlabel.place(y=120, x=30)
        self.browsebButton()
        self.trainButton()
        self.sysInfo()

        #Button Execution
        self.detectButton()
        self.clearButton()

    def browsebButton(self):
        self.button = tk.Button(self, text ="Browse",height="1",width=10,command = self.fileDialog, bg="white")
        self.button.config(font=("Courier", 12))
        self.button.place(y=120, x=250)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir = "/", title = "Select a file", filetype = (("jpeg","*.jpg"), ("png","*.png"),("All Files","*.*")))
        self.label = tk.Label(self, text="",bg='#80c1ff')
        self.label.configure(text="")
        self.label.configure(text = self.filename)
        self.label.place(y=120, x=360)
        self.path.set(self.filename)
        self.createCanvasImage(self.filename,250,310)

    def detectButton(self):
        self.button1 = tk.Button(self, text="Detect",height="1",width=10, command=self.detect,bg="white")
        self.button1.place(y=240, x=500)
        self.button1.config(font=("Courier", 12))

    def clearButton(self):
        self.buttonclear = tk.Button(self, text="clear", height="1",width=10,command=self.clear,bg="white")
        self.buttonclear.place(y=240, x=250)
        self.buttonclear.config(font=("Courier", 12))

    def clear(self):
        self.txt.delete(1.0,'end')
        self.label.destroy()
        self.predicationLabel.destroy()
        self.canvas.destroy()
        self.minsize(960, 300)

    def createCanvasImage(self,filename, X, Y):
        self.minsize(960, 750)
        self.canvas =tk.Canvas(self,height=400, width=450)
        self.canvas.place(x=X,y=Y)
        self.image = Image.open(filename)
        self.resized = self.image.resize((450, 400), Image.ANTIALIAS)
        self.canvas.image = ImageTk.PhotoImage(self.resized)
        self.canvas.create_image(0,0,image=self.canvas.image, anchor ="nw")

    def trainButton(self):
        self.buttontrain = tk.Button(self, text="Train System", height="1", width=15, command=self.msgBox, bg="white")
        self.buttontrain.place(y=180, x=250)
        self.buttontrain.config(font=("Courier", 12))

    def msgBox(self):
        messagebox.showinfo("Info", "System Finished Training")
        accuracy = labeler.trainingPhase("Labeled.csv")
        self.accuracyLabel = tk.Label(self, text="System Accuracy: "+ str(accuracy), bg='#FFFF00')
        self.accuracyLabel.config(font=("Courier" ,12))
        self.accuracyLabel.place(y=184, x=445)

    def sysInfo(self):
        self.buttonInfo = tk.Button(self, text="Show details", height="1", width=15, command=self.moreInfo, bg="white")
        self.buttonInfo.place(y=180, x=700)
        self.buttonInfo.config(font=("Courier", 12))

    def moreInfo(self):
        window = Toplevel()
        window.title("More Information")
        window.minsize(200, 200)
        window.configure(background='#80c1ff')
        window.txt = tk.Text(window,width=80, height = 15)
        window.txt.grid(row=0, column=0, sticky='nsew')
        window.txt.insert(INSERT, labeler.report + "\n"+ labeler.print_cm(labeler.cm, labeler.classes))
        window.txt.config(state="disabled")

    def detect(self):
        self.canvas.destroy()
        #Progress Bar
        self.progressBar = Progressbar(self, length = 200, orient = HORIZONTAL, maximum=100, value=0,mode = 'determinate')
        self.progressBar.place(y=280, x=400)
        bar(self)
        self.progressBar.destroy()
        prediction = labeler.classification(self.path.get(), self.txt.get(1.0, 'end').replace('"', "").replace("(", "").replace(")", "").replace("{", "").replace("}", ""))

        #Display Result [Fake or Real]
        self.predicationLabel = tk.Label(self, text=prediction.upper()+"!", bg='#FFFF00', fg='#000000')
        self.predicationLabel.config(font=("Courier", 20))
        self.predicationLabel.place(y=280, x=450)

        #Display Picture To the left
        self.createCanvasImage(self.filename, 20, 320)

        self.w = tk.Canvas(self, width=430, height=400,borderwidth=1, highlightthickness=0,background='white')
        self.w.config(scrollregion=[0, 0, 1000, 1000])
        self.w.place(y=320, x=500)
        result = labeler.translate( self.txt.get(1.0, 'end').replace('"', "").replace("(", "").replace(")", "").replace("{", "").replace("}", ""))
        result['translatedText']=result['translatedText'].replace('&#39;', "'")
        a = result['translatedText'].split()
        ret = ''
        for i in range(0, len(a), 8):
            ret += ' '.join(a[i:i + 8]) + '\n'

        labels = labeler.label_image(r"" + self.path.get())
        self.w.create_text(95, 20, text="CAPTION IN ENGLISH", font=('courier -16 bold'))
        self.w.create_text(260,130, text=ret,font=('courier', -14))
        self.w.create_text(35, 230, text="LABELS", font=('courier -16 bold'))
        self.w.create_text(75, 330, text=labels,font=('courier', -14))
        self.w.xview_moveto(0.0)
        self.hbar = tk.Scrollbar(self, orient=HORIZONTAL)
        self.hbar.pack(side=BOTTOM, fill=X)
        self.hbar.config(command=self.w.xview)
        self.w.config(xscrollcommand=self.hbar.set)

        self.vbar = tk.Scrollbar(self, orient=VERTICAL)
        self.vbar.pack(side=RIGHT, fill=Y)
        self.vbar.config(command=self.w.yview)
        self.w.config(yscrollcommand=self.vbar.set)


def bar(self):
    import time
    self.progressBar['value'] = 10
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 20
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 40
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 50
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 60
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 80
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 90
    root.update_idletasks()
    time.sleep(1)

    self.progressBar['value'] = 100
    root.update_idletasks()


if __name__ =='__main__':
    root = Root()
    root.mainloop()