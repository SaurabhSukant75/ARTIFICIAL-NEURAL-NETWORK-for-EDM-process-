# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:36:23 2019

@author: saurabhsukant75
"""
from tkinter import messagebox
from BCP import Neural_network
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tkinter import *
#object creation
NN = Neural_network()
#normalization
scalerX = MaxAbsScaler()
scalery = MaxAbsScaler()
#function for early stop method
def earlyStop(x_test,y_test,x_train,y_train,gamma=.2):
        y_predict=NN.predict(x_test)
        accuracy_1=NN.accuracy(y_predict,y_test)
        accuracy_2=accuracy_1+.01
        epoches=100
        while(accuracy_2>accuracy_1):
            accuracy_1=accuracy_2
            epoches+=20
            for _ in range(epoches):
                NN.train(x_train,y_train,gamma)
            y_predict=NN.predict(x_test)    
            accuracy_2=NN.accuracy(y_predict,y_test) 
        return epoches
name="file:///C:/Users/dell/Desktop/MRR Calculator/copper_mildSteel.csv"
def copper():
    global name
    name="file:///C:/Users/dell/Desktop/MRR Calculator/copper_"
def brass():
    global name
    name="file:///C:/Users/dell/Desktop/MRR Calculator/brass_" 
def mildSteel():
    global name
    name+="mildSteel.csv"  
    trigger()     
def castIron():
    global name
    name+="castIron.csv"  
    trigger()
def trigger():

     df=pd.read_csv(name)
     #df["creater_depth"]=df["CD(μm)"]
     X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:59,0:4], 
                        df.iloc[:59,4:6], test_size=0.33, random_state=4)

     # fit and transform
     X_train = scalerX.fit_transform(X_train)
     Y_train = scalery.fit_transform(Y_train)
     X_test = scalerX.transform(X_test)
     Y_test = scalery.transform(Y_test)
     #learning of algorithm
     NN.fit(X_train, Y_train)
     epoch=earlyStop(X_test,Y_test,X_train,Y_train,.2)
     global A
     A=NN.accuracy(NN.predict(X_test),Y_test)
     print("epoch:",epoch)







#GUI DEvelopement






def donothing():
   messagebox.showwarning("Warning","Data not available yet. Choose other option")
def guidelines():
    messagebox.showinfo("Guidelines","1) Choose Toolmaterial first                                                          2) Then choose Workpiece material                                               3) Enter v,i,ton,toff then hit submit botton")
def about():
    messagebox.showinfo("About","This is AI based mrr calculator for die-sinking EDM machine")
def kill_win():
    root.destroy()
root = Tk()
root.title("EDM calculator (AI-based)")
root.geometry("350x250+300+300")  

def mrrpredict():
   filewin = Toplevel(root)
   filewin.geometry("350x250+300+300")
   v=int(e1.get())
   i=int(e2.get())
   ton=int(e3.get())
   toff=int(e4.get())
   test_case=np.array([v,i,ton,toff])
   test_case=test_case.reshape(1,-1)
   test_case = scalerX.transform(test_case)
   
   p_mrr=NN.predict(test_case)
   accuracy=Label(filewin,text="OVERALL ACCURACY OF ANN MODEL :    "+str(A))
   accuracy.pack()
   msg = Label(filewin, text="MRR PREDICTED  :"+"    "+str(p_mrr[0,0:1]))
   msg.pack()
   msg = Label(filewin, text="CRATER DEPTH   :"+"    "+str(p_mrr[0,1:2]))
   msg.pack()
   e1.delete(0, END)
   e2.delete(0, END)
   e3.delete(0, END)
   e4.delete(0, END)
   button = Button(filewin, text = "Back",bg="#FF0000",height=2,width=10,padx=2,command=filewin.destroy)
   button.pack(side=BOTTOM)

   
        
menubar = Menu(root,bg="blue")

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Copper", command=copper)
filemenu.add_command(label="Brass", command=brass)
filemenu.add_command(label="Graphite ", command=donothing)
filemenu.add_command(label="Molybdenum", command=donothing)
filemenu.add_command(label="Silver tungsten", command=donothing)
menubar.add_cascade(label="Tool Material", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Mild Steel", command=mildSteel)
editmenu.add_command(label="Cast Iron", command=castIron)
editmenu.add_command(label="Aluminium", command=donothing)
editmenu.add_command(label="Tungstan", command=donothing)
editmenu.add_command(label="Titanium", command=donothing)

menubar.add_cascade(label="Workpiece Material", menu=editmenu)

Quit = Menu(menubar, tearoff=0)
Quit.add_command(label="quit", command=kill_win)
menubar.add_cascade(label="Quit", menu=Quit)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="guidelines", command=guidelines)
helpmenu.add_command(label="About...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)


root.config(menu=menubar)

v=Label(root,text="        ").grid(row=0)
i=Label(root,text="       ").grid(row=1)

v=Label(root,text="        ").grid(row=2)
i=Label(root,text="       ").grid(row=3)
ton=Label(root,text="           ").grid(row=4)
toff=Label(root,text="      ").grid(row=5)
v=Label(root,text="voltage (v) :").grid(row=2,column=1)
i=Label(root,text="current (A) :").grid(row=3,column=1)
ton=Label(root,text="ton (micro sec.):").grid(row=4,column=1)
toff=Label(root,text="toff (micro sec.):").grid(row=5,column=1)

e1=Entry(root)
e2=Entry(root)
e3=Entry(root)
e4=Entry(root)
e1.grid(row=2,column=2)
e2.grid(row=3,column=2)
e3.grid(row=4,column=2)
e4.grid(row=5,column=2)
click=Button(root,text="submit",relief="raised",state="active",command=mrrpredict)
v=Label(root,text="        ").grid(row=6)
click.grid(row=7,column=2)

root.mainloop()
