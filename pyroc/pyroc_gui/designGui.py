import argparse
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import warnings
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.spatial import Delaunay

from ..pyroc_utils.designVars import *
from ..pyroc_utils.miscVar import *

#Testing

matplotlib.use('TkAgg')
try:
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
    warnings.filterwarnings('ignore', category=UserWarning)
except:
    pass

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, orient, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        if orient=='vertical':
            scrollbar = ttk.Scrollbar(self, orient=orient, command=canvas.yview)
        elif orient=='horizontal':
            scrollbar = ttk.Scrollbar(self, orient=orient, command=canvas.xview)
        else:
            raise Exception('Orient must be in [\'horizontal\', \'vertical\']')
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(
                scrollregion=canvas.bbox('all')
            )
        )

        if orient=='vertical':
            canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
        else:
            canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
            canvas.configure(xscrollcommand=scrollbar.set)
            canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.BOTTOM, fill='x')

        

class PyrocDesign(object):
    '''
    Base class for analyzing design space

    Parameters
    ----------------
    geo: object
        CSTMulti object for computing surface from coefficients
    '''

    def __init__(self, geo, name='designGUI', mode='2d', figsize=4, res=[100,10]):
        self.geo = geo
        self.name = name
        self.mode = mode

        self.root = tk.Tk()
        self.root.wm_title('PYRoc Design GUI - v0.0.1')
        self.root.geometry('800x600')
        self.root.protocol('WM_DELETE_WINDOW', self.quit)

        try:
            icon_dir = os.path.dirname(os.path.abspath(__file__))
            icon_name = 'cyrocIcon.gif'
            icon_dir_full = os.path.join(icon_dir, '..', '..', 'assets', icon_name)
            self.icon = tk.PhotoImage(file=icon_dir_full)
            self.root.tk.call('wm', 'iconphoto', self.root._w, self.icon)
        except:
            pass

        surf0 = np.array([])
        for ptSetName in self.geo.embeddedSurfaces:
            coords = self.geo.getAttachedPoints(ptSetName)
            if len(surf0)>0:
                surf0 = np.append(surf0,coords,axis=0)
            else:
                surf0 = coords

        #Check dimensions        
        self.ndim = len(surf0[0,:])

        #Set plot bounds, etc
        bounds = np.array([[np.min(surf0[:,_]), np.max(surf0[:,_])] for _ in range(self.ndim)])
        width = 0.25
        self.lim = np.array([[bounds[_,0]-width, bounds[_,1]+width] for _ in range(self.ndim)])
        self.textLim = np.array([[tk.StringVar(value=str(self.lim[_,0])), tk.StringVar(value=str(self.lim[_,1]))] for _ in range(self.ndim)])
        self.resolution = np.array([tk.IntVar(value=int(res[0])), tk.IntVar(value=int(res[1]))])
        self.sensitivity = None
        
        # Instantiate the MPL figure
        self.fig = plt.figure(figsize=(figsize,figsize), dpi=100, facecolor='white')
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

        if self.mode in ['2d','2dcut','3d']:
            self.fig.clf()
            if self.mode in ['2d', '2dcut']:
                self.plotAx = self.fig.add_subplot(1, 1, 1)
            else:
                self.plotAx = self.fig.add_subplot(1, 1, 1, projection='3d')
        else:
            raise Exception(str(self.mode)+' mode not implemented')

        self.dvDict = self.createDvDict(self.geo)
        self.setupGeo()
        self.draw_gui()
        self.update(None)

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
        self.fig.clf()
        a = self.fig.add_subplot(111)
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
        '''
        Create the frames and widgets in the bottom section of the canvas.
        '''
        fontS = tkFont.Font(family='Helvetica', size=8)
        fontM = tkFont.Font(family='Helvetica', size=10)
        fontL = tkFont.Font(family='Helvetica', size=12)

        dv_frame  = ScrollableFrame(self.root,orient='horizontal')

        #Create Sliders for Each Design Variable
        surfs = list(self.dvDict.keys())
        keys = [list(self.dvDict[_].keys()) for _ in surfs]
        c=0
        for _ in range(len(surfs)):
            surf = surfs[_]
            for __ in range(len(keys[_])):
                key = keys[_][__]
                dv = self.dvDict[surf][key]
                if dv.getValue()<0:
                    dvSlider = tk.Scale(dv_frame.scrollable_frame, from_=dv.lower, to=dv.upper, orient='vertical',
                                        variable=dv.value, command=self.update, resolution=dv.getValue()*1e-3, font=fontS)
                else:
                    dvSlider = tk.Scale(dv_frame.scrollable_frame, from_=dv.upper, to=dv.lower, orient='vertical',
                                        variable=dv.value, command=self.update, resolution=dv.getValue()*1e-3, font=fontS)
                dvSlider.grid(row=0, column=c)
                dvLabel = tk.Label(dv_frame.scrollable_frame, text=key+'S'+str(_), font=fontS)
                dvLabel.grid(row=1, column=c)
                dvEntry = ttk.Entry(dv_frame.scrollable_frame, validatecommand=self.update, textvariable=dv.value)
                dvEntry.grid(row=2, column=c)
                dvHideSens = tk.Radiobutton(dv_frame.scrollable_frame, command=self.showSens, text='Disable Sens', variable=dv.showSens, value=0, font=fontM)
                dvHideSens.grid(row=3, column=c)
                dvShowSensX = tk.Radiobutton(dv_frame.scrollable_frame, command=self.showSens, text='X Sensitivity', variable=dv.showSens, value=1, font=fontM)
                dvShowSensX.grid(row=4, column=c)
                dvShowSensY = tk.Radiobutton(dv_frame.scrollable_frame, command=self.showSens, text='Y Sensitivity', variable=dv.showSens, value=2, font=fontM)
                dvShowSensY.grid(row=5, column=c)
                dvShowSensZ = tk.Radiobutton(dv_frame.scrollable_frame, command=self.showSens, text='Z Sensitivity', variable=dv.showSens, value=3, font=fontM)
                dvShowSensZ.grid(row=6, column=c)
                c+=1
        
        dv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        cmd_frame = ttk.Frame(self.root)

        xlimLabel = ttk.Label(cmd_frame, text='x-limits', font=fontS)
        xlimLabel.grid(row=1, column=1)
        xlimlBox = ttk.Entry(cmd_frame, textvariable=self.textLim[0,0])
        xlimlBox.grid(row=1, column=2)
        xlimuBox = ttk.Entry(cmd_frame, textvariable=self.textLim[0,1])
        xlimuBox.grid(row=1, column=3)

        ylimLabel = tk.Label(cmd_frame, text='y-limits', font=fontS)
        ylimLabel.grid(row=2, column=1)
        ylimlBox = ttk.Entry(cmd_frame, textvariable=self.textLim[1,0])
        ylimlBox.grid(row=2, column=2)
        ylimuBox = ttk.Entry(cmd_frame, textvariable=self.textLim[1,1])
        ylimuBox.grid(row=2, column=3)

        if self.ndim==3:
            zlimLabel = tk.Label(cmd_frame, text='z-limits', font=fontS)
            zlimLabel.grid(row=3, column=1)
            zlimlBox = ttk.Entry(cmd_frame, textvariable=self.textLim[2,0])
            zlimlBox.grid(row=3, column=2)
            zlimuBox = ttk.Entry(cmd_frame, textvariable=self.textLim[2,1])
            zlimuBox.grid(row=3, column=3)

        updateLimits = tk.Button(cmd_frame, text='Update Limits', command=self.setLimits, font=fontM)
        updateLimits.grid(row=self.ndim+1, column=1, columnspan=3)

        quitButton = tk.Button(cmd_frame, text='Quit', command=self.quit, font=fontM)
        quitButton.grid(row=1, column=5, padx=25, sticky=tk.N)

        xresLabel = tk.Label(cmd_frame, text='x-res', font=fontS)
        xresLabel.grid(row=2, column=4)
        xresSlider = tk.Scale(cmd_frame, from_=10, to=1000, orient='horizontal',
                               variable=self.resolution[0], resolution=1, command=self.update, font=fontS)
        xresSlider.grid(row=2, column=5)

        xresLabel = tk.Label(cmd_frame, text='y-res', font=fontS)
        xresLabel.grid(row=3, column=4)
        yresSlider = tk.Scale(cmd_frame, from_=10, to=1000, orient='horizontal',
                               variable=self.resolution[1], resolution=1, command=self.update, font=fontS)
        yresSlider.grid(row=3, column=5)

        cmd_frame.pack(side=tk.RIGHT)

    def createDvDict(self, geo):
        dvDict = OrderedDict()
        dvList = geo.coef
        for _ in range(len(dvList)):
            dvSet = dvList[_]
            dvDict['dvSet'+str(_)] = OrderedDict()
            for __ in range(len(dvSet)):
                dv = dvSet[__]
                if isScalar(dv):
                    if dv==0:
                        dvDict['dvSet'+str(_)]['dv'+str(__)] = TkDesignVar('dv'+str(__),tk.DoubleVar(value=dv),-1,1,tk.IntVar(value=0))
                    else:
                        dvDict['dvSet'+str(_)]['dv'+str(__)] = TkDesignVar('dv'+str(__),tk.DoubleVar(value=dv),-2*dv,5*dv,tk.IntVar(value=0))
                else:
                    dvDict['dvSet'+str(_)][dv[0]] = TkDesignVar(dv[0],tk.DoubleVar(value=dv[1]),dv[2],dv[3],tk.IntVar(value=0))
        return dvDict

    def setLimits(self):
        for _ in range(self.ndim):
            for __ in range(2):
                self.lim[_,__] = float(self.textLim[_,__].get())
        self.update(None)

    def showSens(self):
        surfs = list(self.dvDict.keys())
        keys = [list(self.dvDict[_].keys()) for _ in surfs]
        for _ in range(len(surfs)):
            surf = surfs[_]
            for __ in range(len(keys[_])):
                key = keys[_][__]
                dv = self.dvDict[surf][key]
                sens = dv.showSens.get()
                if sens>0:
                    self.sensitivity = [_,__,sens-1]
                    return

    def makeColormap(self, seq):
        cdict = {'red': [], 'green': [], 'blue': []}
        x = np.linspace(0,1, len(seq))
        for _ in range(len(seq)):
            tone = seq[_]
            cdict['red'].append([x[_], 1-tone, 1-tone])
            cdict['green'].append([x[_], 0, 0])
            cdict['blue'].append([x[_], tone, tone])
        return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)

    def setupGeo(self):
        if self.mode == '2d':
            x = np.append(np.linspace(0.0,1.0,self.resolution[0].get()),
                          np.linspace(0.0,1.0,self.resolution[0].get()))
            parameterization = np.vstack([x, positiveOrNegative(len(x))]).T
            self.geo.attachPoints(parameterization, self.name, parameterization=True)

        elif self.mode == '2dcut':
            x = np.append(np.linspace(0.0,1.0,self.resolution[0].get()),
                          np.linspace(0.0,1.0,self.resolution[0].get()))
            parameterization = np.vstack([x, np.zeros_like(x), positiveOrNegative(len(x))]).T
            self.geo.attachPoints(parameterization, self.name, parameterization=True)

        elif self.mode =='3d':
            x = np.append(np.linspace(1e-6,1.0-1e-6,self.resolution[0].get()),
                          np.linspace(1e-6,1.0-1e-6,self.resolution[0].get()))
            y = np.append(np.linspace(1e-6,1.0-1e-6,self.resolution[1].get()),
                          np.linspace(1e-6,1.0-1e-6,self.resolution[1].get()))
            xv,yv = np.meshgrid(x,y)
            xv,yv = xv.flatten(), yv.flatten()
            parameterization = np.vstack([xv, yv, positiveOrNegative(len(xv))]).T
            self.geo.attachPoints(parameterization, self.name, parameterization=True)

    def update(self, e):
        self.geo.coef = [[self.dvDict[_][__].getValue() for __ in self.dvDict[_].keys()] for _ in self.dvDict.keys()]
        self.geo.updateCoeffs()
        self.plotAx.cla()

        if self.mode == '2d':
            surf = self.geo.embeddedSurfaces[self.name]
            x = np.append(np.linspace(0.0,1.0,self.resolution[0].get()),
                          np.linspace(0.0,1.0,self.resolution[0].get()))
            parameterization = np.vstack([x, np.zeros_like(x)]).T
            surf.__init__(parameterization, self.geo.embeddedParams, parameterization=True)
            newCoords = surf.coordinates
            self.plotAx.plot(newCoords[:,0], newCoords[:,1])
            self.plotAx.set_xlabel('x')
            self.plotAx.set_ylabel('z')
            self.plotAx.axes.set_xlim([self.lim[0,0],self.lim[0,1]])
            self.plotAx.axes.set_ylim([self.lim[1,0],self.lim[1,1]])

        elif self.mode == '2dcut':
            surf = self.geo.embeddedSurfaces[self.name]
            x = np.append(np.linspace(0.0,1.0,self.resolution[0].get()),
                          np.linspace(0.0,1.0,self.resolution[0].get()))
            parameterization = np.vstack([x, np.zeros_like(x), np.zeros_like(x)]).T
            surf.__init__(parameterization, self.geo.embeddedParams, parameterization=True)
            newCoords = surf.coordinates
            self.plotAx.plot(newCoords[:,0], newCoords[:,2])
            self.plotAx.set_xlabel('x')
            self.plotAx.set_ylabel('z')
            self.plotAx.axes.set_xlim([self.lim[0,0],self.lim[0,1]])
            self.plotAx.axes.set_ylim([self.lim[1,0],self.lim[1,1]])

        elif self.mode =='3d':
            surf = self.geo.embeddedSurfaces[self.name]
            x = np.append(np.linspace(1e-6,1.0-1e-6,self.resolution[0].get()),
                          np.linspace(1e-6,1.0-1e-6,self.resolution[0].get()))
            y = np.append(np.linspace(1e-6,1.0-1e-6,self.resolution[1].get()),
                          np.linspace(1e-6,1.0-1e-6,self.resolution[1].get()))
            xv,yv = np.meshgrid(x,y)
            xv,yv = xv.flatten(), yv.flatten()
            tri = Delaunay(np.array([xv,yv]).T)
            parameterization = np.vstack([xv, yv, np.zeros_like(xv)]).T
            surf.__init__(parameterization, self.geo.embeddedParams, parameterization=True)
            newCoords = surf.coordinates
            #newCoords = np.sort(surf.coordinates,axis=0)
            # if self.sensitivity is not None and self.sensitivity[0]==i:
            #     sensitivity = surf.calcJacobian(newCoords)[self.sensitivity[2]::3,3*self.sensitivity[1]+self.sensitivity[2]]
            # else:
            #     sensitivity = newCoords[:,2]
            # if np.max(sensitivity)==np.min(sensitivity):
            #     normSens = sensitivity
            # else:
            #     normSens = np.abs(sensitivity-np.min(sensitivity))/(np.max(sensitivity)-np.min(sensitivity))
            #cmap = self.makeColormap(normSens)
            self.plotAx.plot_trisurf(newCoords[:,0], newCoords[:,1], newCoords[:,2], triangles=tri.simplices)#,cmap=cmap)
            self.plotAx.set_xlabel('x')
            self.plotAx.set_ylabel('y')
            self.plotAx.set_zlabel('z')
            self.plotAx.axes.set_xlim3d([self.lim[0,0],self.lim[0,1]])
            self.plotAx.axes.set_ylim3d([self.lim[1,0],self.lim[1,1]])
            self.plotAx.axes.set_zlim3d([self.lim[2,0],self.lim[2,1]])
        self.canvas.draw()