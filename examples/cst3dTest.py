#Imports
from pyroc import *
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

##########################################################################

#Constants
rootChord = 1.0
tipChord = 0.25 * rootChord
totalSweep = np.radians(20.0)
totalShear = 0.25
totalTwist = np.radians(10.0)

psiVals = np.linspace(0, 1, 100)
etaVals = np.linspace(0, 1, 10)
psi, eta = np.meshgrid(psiVals, etaVals)
psi, eta = psi.flatten(), eta.flatten()
psiEtaZeta = np.array(list(zip(psi,eta,np.zeros_like(psi))))
tri = Delaunay(np.array([psi,eta]).T)

##########################################################################

##Plot Airfoil Example:
# span = 1.0
# upper = CSTAirfoil3D(psiEtaZeta, refSpan=span, chordCoeffs=[rootChord])
# lower = CSTAirfoil3D(psiEtaZeta, refSpan=span, chordCoeffs=[rootChord], shapeScale=-1.0)

#psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T
#upper.setPsiEtaZeta(psiEtaZeta)
#uCoords = upper.updateCoords()
#lower.setPsiEtaZeta(psiEtaZeta)
#lCoords = lower.updateCoords()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_trisurf(uCoords[:,0], uCoords[:,1], uCoords[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# ax.plot_trisurf(lCoords[:,0], lCoords[:,1], lCoords[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.axes.set_xlim3d(left=-span/2, right=span/2) 
# ax.axes.set_ylim3d(bottom=0, top=span) 
# ax.axes.set_zlim3d(bottom=-span/2, top=span/2)
# plt.show()

##########################################################################

##Plot Wing ex: ## Origin at LE of root chord
# span = 4.0 * rootChord
# upper = CSTWing3D(psiEtaZeta, refSpan=span, sweepCoeffs=[totalSweep], shearCoeffs=[totalShear], twistCoeffs=[totalTwist],
#                   chordCoeffs=[rootChord, tipChord])
# lower = CSTWing3D(psiEtaZeta, refSpan=span, sweepCoeffs=[totalSweep], shearCoeffs=[totalShear], twistCoeffs=[totalTwist],
#                   chordCoeffs=[rootChord, tipChord], shapeScale=-1.0)

#psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T
#upper.setPsiEtaZeta(psiEtaZeta)
#uCoords = upper.updateCoords()
#lower.setPsiEtaZeta(psiEtaZeta)
#lCoords = lower.updateCoords()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_trisurf(uCoords[:,0], uCoords[:,1], uCoords[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# ax.plot_trisurf(lCoords[:,0], lCoords[:,1], lCoords[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.axes.set_xlim3d(left=-span/2, right=span/2) 
# ax.axes.set_ylim3d(bottom=0, top=span) 
# ax.axes.set_zlim3d(bottom=-span/2, top=span/2)
# plt.show()

##########################################################################

#3d Design GUI Example
span = 4.0 * rootChord
upper = CSTWing3D(psiEtaZeta, refSpan=span, sweepCoeffs=[totalSweep], shearCoeffs=[totalShear], twistCoeffs=[totalTwist],
                  chordCoeffs=[rootChord, tipChord])
lower = CSTWing3D(psiEtaZeta, refSpan=span, sweepCoeffs=[totalSweep], shearCoeffs=[totalShear], twistCoeffs=[totalTwist],
                  chordCoeffs=[rootChord, tipChord], shapeScale=-1.0)

psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T
upper.setPsiEtaZeta(psiEtaZeta)
uCoords = upper.updateCoords()
lower.setPsiEtaZeta(psiEtaZeta)
lCoords = lower.updateCoords()

geo = GeoEx(surfaces=[upper, lower])
p = PyrocDesign(geo,mode='3d')
p.root.mainloop()