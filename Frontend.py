
#The frontEnd program for execution


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
import tkinter as tk
from tkinter import ttk

LARGE_FONT = ("Verdana",12)
ACC_FONT = (" Sans Serif", 10)
style.use("ggplot")



class SeaofBTCapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self,"Electrical Load Forecasting at Source")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageLR, PageSVM, PageARIMA, PageLSTM, PageMLP):
            frame = F(container, self)
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()



class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self,text = "Select the Algorithm", font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)

        
        button1 =ttk.Button(self,text="Linear Regression",
                           command=lambda: controller.show_frame(PageLR))
        button1.pack()

        button2 = ttk.Button(self, text="Support Vector Machine",
                            command=lambda: controller.show_frame(PageSVM))
        button2.pack()

        button3 = ttk.Button(self, text="ARIMA",
                            command=lambda: controller.show_frame(PageARIMA))
        button3.pack()

        button4 = ttk.Button(self, text="MLP",
                            command=lambda: controller.show_frame(PageMLP))
        button4.pack()

        button5 = ttk.Button(self, text="LSTM",
                            command=lambda: controller.show_frame(PageLSTM))
        button5.pack()
        button6 =ttk.Button(self,text="Quit",
                           command=quit)
        button6.pack()

	
	


class PageLR(tk.Frame):

    def __init__(self,parent, controller):
        import LR
        tk.Frame.__init__(self, parent)

        button1 = ttk.Button(self, text="Back Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Execute",
                            command=LR.execLR)
        button2.pack()


class PageSVM(tk.Frame):

    def __init__(self,parent, controller):
        import SVM
        tk.Frame.__init__(self, parent)

        button1 = ttk.Button(self, text="Back Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Execute",
                             command=SVM.execSVM)
        button2.pack()


class PageARIMA(tk.Frame):

    def __init__(self,parent, controller):
        import ARIMA
        tk.Frame.__init__(self, parent)

        button1 = ttk.Button(self, text="Back Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Execute",
                             command=ARIMA.execARIMA)
        button2.pack()


class PageMLP(tk.Frame):

    def __init__(self,parent, controller):
        import MLP
        tk.Frame.__init__(self, parent)

        button1 = ttk.Button(self, text="Back Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Execute",
                             command=MLP.execMLP)
        button2.pack()

class PageLSTM(tk.Frame):

    def __init__(self,parent, controller):
        import LSTM
        tk.Frame.__init__(self, parent)

        button1 = ttk.Button(self, text="Back Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Execute",
                             command=LSTM.execLSTM)
        button2.pack()

app = SeaofBTCapp()
app.mainloop()
