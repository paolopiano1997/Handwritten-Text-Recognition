from guizero import App, Text,PushButton, TextBox, Box
import tkinter as tk
from tkinter import filedialog
from main import start
import os
import ntpath

def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=
        [('image files','.png'),('image files','.jpg'),('image files','.jpeg')])
    filename.show()
    filename.value=file_path
    startInference.enable()
    
def start_inference():
    try:
        name = filename.value
        f = open(name,'r')
        start(name)
        print("Inference Finished")
        print("Filename: " + name)
        head, tail = ntpath.split(name)
        name =  tail or ntpath.basename(head)
        name = os.path.splitext(name)[0]
        print("Name: " + name)
        fout = open('output/' + name + '_output.txt','r')
        fperc = open('output/' + name + '_probability.txt','r')
        outLabel.show()
        output.show()
        output.clear()
        output.append(fout.read())
        output.width = 128
        output.height = 12
        percLabel.show()
        perc.show()
        perc.clear()
        perc.append(fperc.read() + "%")
    except OSError:
        app.error("Error","File not found of not a valid extension: " + name)
        
def close():
     if app.yesno("Close", "Do you want to quit?"):
        app.destroy()
        exit(0)
        
if __name__ == '__main__':
    app = App(title="Handwritten Text Recognition")
    app.when_closed = close
    box=Box(app,width="fill")
    message = Text(box, text="Handwritten Text Recognition")
    file = Text(box)
    chooseFile = PushButton(box,command=choose_file,text="Choose file...")
    filename = Text(box,visible=None)
    startInference = PushButton(box, command=start_inference,text="Start inference",enabled=False)
    outLabel = Text(box,text="Output",visible=None)
    output = TextBox(box,visible=None,multiline=True, scrollbar=True)
    percLabel =Text(box,text="Probability",visible=None)
    perc = TextBox(box,visible=None)
    app.display()
