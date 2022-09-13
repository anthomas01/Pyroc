import argparse
import os
import tkinter as tk
from tkinter import font as tkFont
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

matplotlib.use('TkAgg')
try:
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
    warnings.filterwarnings('ignore', category=UserWarning)
except:
    pass

class Interface():
    def __init__(self, mode='2d', figsize=4):
        self.mode = mode

        self.root = tk.Tk()
        self.root.wm_title('Example Gui')
        self.root.geometry('800x600')

        try:
            icon_dir = os.path.dirname(os.path.abspath(__file__))
            icon_name = 'cyrocIcon.png'
            icon_dir_full = os.path.join(icon_dir, '..', '..', 'assets', icon_name)
            icon = tk.PhotoImage(file=icon_dir_full)
            self.root.tk.call(icon)
        except:
            pass
        
        figsize = (figsize,figsize)

        # Instantiate the MPL figure
        self.fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
        if mode=='3d': ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        plt.grid(True)
        # Link the MPL figure onto the TK canvas and pack it
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a toolbar to explore the figure like normal MPL behavior
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Increase the font size
        matplotlib.rcParams.update({'font.size': 16})

        self.draw_gui()

    def quit(self):
        '''
        Destroy GUI window cleanly if quit button pressed.
        '''
        self.root.quit()
        self.root.destroy()

    def error_display(self, string='That option is not supported'):
        '''
        Display error string on canvas when invalid options selected.
        '''
        self.f.clf()
        a = self.f.add_subplot(111)
        a.text(0.05, 0.9, 'Error: ' + string, fontsize=20, transform=a.transAxes)
        self.canvas.draw()

    def warning_display(self, string='That option is not supported'):
        '''
        Display warning message on canvas as necessary.
        '''
        a = plt.gca()
        a.text(0.05, 0.9, 'Warning: ' + string, fontsize=20, transform=a.transAxes)
        self.canvas.draw()

    def note_display(self, string=''):
        '''
        Display warning message on canvas as necessary.
        '''
        a = plt.gca()
        a.text(0.05, 0.5, string, fontsize=20, transform=a.transAxes)
        self.canvas.draw()

    def draw_gui(self):
        """
        Create the frames and widgets in the bottom section of the canvas.
        """
        font = tkFont.Font(family="Helvetica", size=15)

        sel_frame = tk.Frame(self.root)
        sel_frame.pack(side=tk.RIGHT)

        quitButton = tk.Button(sel_frame, text="Quit", command=self.quit, font=font)
        quitButton.grid(row=1, column=1, padx=25, sticky=tk.N)
 
p = Interface(mode='3d')
tk.mainloop()
