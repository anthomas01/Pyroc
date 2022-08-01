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

from .designVars import DesignVar
from .cst2d import *
from .cst3d import *

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
        Geometry object for computing surface from coefficients
    '''

    def __init__(self, geo=None, mode='2d', figsize=4, res=[100,10]):
        self.geo = geo
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
        for surf in self.geo.surfaces:
            surf.updateCoords()
            if len(surf0)>0:
                surf0 = np.append(surf0,surf.updateCoords(),axis=0)
            else:
                surf0 = surf.updateCoords()
        self.ndim = len(surf0[0,:])
        bounds = np.array([[np.min(surf0[:,_]), np.max(surf0[:,_])] for _ in range(self.ndim)])
        width = 0.25
        self.lim = np.array([[bounds[_,0]-width, bounds[_,1]+width] for _ in range(self.ndim)])
        self.textLim = np.array([[tk.StringVar(value=str(self.lim[_,0])), tk.StringVar(value=str(self.lim[_,1]))] for _ in range(self.ndim)])
        self.resolution = np.array([tk.IntVar(value=int(res[0])), tk.IntVar(value=int(res[1]))])
        
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

        if self.mode in ['2d','3d']:
            self.fig.clf()
            if self.mode=='2d':
                self.plotAx = self.fig.add_subplot(1, 1, 1)
            else:
                self.plotAx = self.fig.add_subplot(1, 1, 1, projection='3d')
        else:
            raise Exception(str(self.mode)+' mode not implemented')

        self.dvDict = self.createDvDict()
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
                dvSlider = tk.Scale(dv_frame.scrollable_frame, from_=dv.upper, to=dv.lower, orient='vertical',
                                     variable=dv.value, command=self.update, resolution=dv.getValue()*1e-3, font=fontS)
                dvSlider.grid(row=0, column=c)
                dvLabel = tk.Label(dv_frame.scrollable_frame, text=key+'S'+str(_), font=fontS)
                dvLabel.grid(row=1, column=c)
                dvEntry = ttk.Entry(dv_frame.scrollable_frame, textvariable=dv.value)
                dvEntry.grid(row=2, column=c)
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

    def createDvDict(self):
        dvDict = OrderedDict()
        dvList = self.geo.getCoeffs()
        for _ in range(len(dvList)):
            dvSet = dvList[_]
            dvDict['dvSet'+str(_)] = {}
            for __ in range(len(dvSet)):
                dv = dvSet[__]
                if type(dv) == list:
                    dvDict['dvSet'+str(_)][dv[0]] = DesignVar(dv[0],tk.DoubleVar(value=dv[1]),dv[2],dv[3])
                else:
                    dvDict['dvSet'+str(_)]['dv'+str(__)] = DesignVar('dv'+str(__),tk.DoubleVar(value=dv),-2*dv,5*dv)
        return dvDict

    def setLimits(self):
        for _ in range(self.ndim):
            for __ in range(2):
                self.lim[_,__] = float(self.textLim[_,__].get())
        self.update(None)

    def update(self, e):
        self.geo.updateCoeffs([[self.dvDict[_][__].getValue() for __ in self.dvDict[_].keys()] for _ in self.dvDict.keys()])
        self.plotAx.cla()

        if self.mode == '2d':
            for surf in self.geo.surfaces:
                x=np.linspace(0.0,1.0,self.resolution[0].get())
                surf.setPsiZeta(x)
                newCoords = surf.updateCoords()
                self.plotAx.plot(newCoords[:,0], newCoords[:,1])
            self.plotAx.set_xlabel('x')
            self.plotAx.set_ylabel('y')
            self.plotAx.axes.set_xlim([self.lim[0,0],self.lim[0,1]])
            self.plotAx.axes.set_ylim([self.lim[1,0],self.lim[1,1]])
        elif self.mode =='3d':
            for surf in self.geo.surfaces:
                newCoords = surf.updateCoords()
                param = surf.getParam()
                xv, yv = param[:,0], param[:,1]
                tri = Delaunay(np.array([xv,yv]).T)
                self.plotAx.plot_trisurf(newCoords[:,0], newCoords[:,1], newCoords[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
            self.plotAx.set_xlabel('x')
            self.plotAx.set_ylabel('y')
            self.plotAx.set_zlabel('z')
            self.plotAx.axes.set_xlim3d([self.lim[0,0],self.lim[0,1]])
            self.plotAx.axes.set_ylim3d([self.lim[1,0],self.lim[1,1]])
            self.plotAx.axes.set_zlim3d([self.lim[2,0],self.lim[2,1]])
        self.canvas.draw()


#Examples
# class GeoEx():
#     def __init__(self, surfaces=[], coeffPairs=None):
#         self.surfaces = surfaces
#         self.coeffPairs = coeffPairs

#     def updateCoeffs(self,coeffs):
#         for _ in range(len(self.surfaces)):
#             surf = self.surfaces[_]
#             oldCoeffs = surf.getCoeffs()
#             for __ in range(len(oldCoeffs)): #Loop through coefficients
#                 if self.coeffPairs is not None:
#                     for i in range(len(self.coeffPairs[:,0])):
#                         if self.coeffPairs[i,2]==_ and self.coeffPairs[i,3]==__: #Coefficient is bound
#                             coeffs[_][__] = coeffs[self.coeffPairs[i,0]][self.coeffPairs[i,1]]
#             surf.updateCoeffs(coeffs[_])
#             surf.updateCoords()     

#     def getCoeffs(self):
#         coeffs = []
#         for _ in range(len(self.surfaces)):
#             surf = self.surfaces[_]
#             coeffs.append(surf.getCoeffs())
#         return coeffs

#2d Example
# x=np.linspace(0,1.0,100)
# z=np.zeros_like(x)
# arr = np.array(list(zip(x,z)))
# cst = CSTAirfoil2D(arr,order=7)
# geo = GeoEx([CSTAirfoil2D(arr,order=3,shapeOffset=0.01),CSTAirfoil2D(arr,order=2,shapeScale=-1.0,shapeOffset=0.00)])

#3d Example
# rootChord = 1.0
# tipChord = 1.0 * rootChord
# span = 6.0 * rootChord
# sweepAngle = np.radians(5.0)
# twistAngle = np.radians(5.0)
# refAxes = np.array([[1.0,0.0,0.0],[np.sin(sweepAngle),np.cos(sweepAngle),0.0]])

# xVals = np.linspace(0,rootChord,80)
# yVals = np.linspace(0,span,50)
# xv,yv = np.meshgrid(xVals,yVals)
# xv = xv.flatten()
# yv = yv.flatten()
# psiVals = np.linspace(0,1,len(xVals))
# etaVals = np.linspace(0,1,len(yVals))
# psiv, etav = np.meshgrid(psiVals,etaVals)
# psiv = psiv.flatten()
# etav = etav.flatten()
# surface1 = np.array(list(zip(xv,yv,np.zeros(len(xv)))))
# surface2 = np.array(list(zip(xv,yv,np.zeros(len(xv)))))

# def wingExtFunc(eta, *coeffs):
#     coeffs = coeffs[0]
#     return coeffs[0]*np.power(eta,coeffs[1])

# def wingExtModFunc(pts, eta, *coeffs):
#     coeffs = coeffs[0]
#     # def angle(eta, *coeffs):
#     #     coeffs = coeffs[0]
#     #     return (np.exp(10*eta)-1)*coeffs[0] 
#     # newPts = []
#     # for _ in range(len(pts)):
#     #     pt = pts[_]
#     #     pt[2] += -coeffs[-1]*np.log(np.cos(angle(eta,coeffs)))
#     #     newPts.append(pt)
#     #return np.array(newPts)
#     return pts

# def wingChordModFunc(eta, *coeffs):
#     coeffs = coeffs[0]
#     return (rootChord-eta*(rootChord-coeffs[0]))

# def wingTwistFunc(pts, eta, *coeffs):
#     coeffs = coeffs[0]
#     def angle(eta, *coeffs):
#         coeffs = coeffs[0]
#         return eta*coeffs[0]
#     newPts = np.zeros([len(pts),3])
#     for _ in range(len(pts)):
#         pt = pts[_]
#         rotV = rotation.rotVbyW(pt,np.array([0.0,1.0,0.0]),angle(eta,coeffs))
#         newPts[_,:] = rotV
#     return newPts

# wingSS = CSTWing3D(surface1, csClassCoeffs=[0.5,1.0], extrudeFunc=wingExtFunc, extClassCoeffs=[2.0,3.0], extModFunc=wingExtModFunc, extModCoeffs=[4e-5, 5.0], chordModFunc=wingChordModFunc,
#                    chordModCoeffs=[1e-1*rootChord], csModFunc=wingTwistFunc, csModCoeffs=[twistAngle], refAxes=refAxes, order=[2,0])
# wingSS.setPsiEtaZeta(psiVals=psiv,etaVals=etav)

# wingPS = CSTWing3D(surface2, csClassCoeffs=[0.5,1.0], extrudeFunc=wingExtFunc, extClassCoeffs=[2.0,3.0], extModFunc=wingExtModFunc, extModCoeffs=[4e-5, 5.0], chordModFunc=wingChordModFunc,
#                    chordModCoeffs=[1e-1*rootChord], csModFunc=wingTwistFunc, csModCoeffs=[twistAngle], refAxes=refAxes, order=[2,0], shapeScale=-1.0)
# wingPS.setPsiEtaZeta(psiVals=psiv,etaVals=etav)

#                 #surf1, coeffN1, surf2, coeffN2
# globalCoeffPairs = [[0,0,1,0],
#                     [0,1,1,1],
#                     [0,5,1,5],
#                     [0,6,1,6],
#                     [0,7,1,7],
#                     [0,8,1,8],
#                     [0,9,1,9],
#                     [0,10,1,10],
#                     [0,11,1,11]]
# geo = GeoEx([wingSS, wingPS], np.array(globalCoeffPairs))

#############################

# p = PyrocDesign(geo,mode='2d')
# p.root.mainloop()
