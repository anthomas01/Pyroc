#imports
from pyroc import *
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

##########################################################################

def func(psi, eta):
    return (psi)*(1-psi)

#Constants
rootChord = 1.0
tipChord = 0.25 * rootChord
totalSweep = np.radians(20.0)
totalShear = 0.25
totalTwist = np.radians(10.0)

psiVals = np.linspace(0, 1, 10)
etaVals = np.linspace(0, 1, 5)
psi, eta = np.meshgrid(psiVals, etaVals)
psi, eta = psi.flatten(), eta.flatten()
psiEtaZeta = np.vstack([psi, eta, func(psi, eta)]).T

##########################################################################
##                       [  class ] [          shape           ][chord][TE]
##Plot Pseudo2D Example: [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
# span = 1.0
# coordsUpper = psiEtaZeta.copy()
# upper = CSTAirfoil3D(coordsUpper, refSpan=span, chordCoeffs=[rootChord])
# coordsLower = psiEtaZeta.copy()
# coordsLower[:,2] *= -1
# lower = CSTAirfoil3D(coordsLower, refSpan=span, chordCoeffs=[rootChord], shapeScale=-1.0)
# coords = np.append(coordsUpper, coordsLower, 0)

# airfoil = CSTMultiParam()
# airfoil.attachCSTParam(upper, 'airfoil_upper')
# airfoil.attachCSTParam(lower, 'airfoil_lower')
# airfoil.attachPoints(coords, 'airfoil')

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# airfoil.plotMulti(ax)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.axes.set_xlim3d(left=-0.5, right=1.5) 
# ax.axes.set_ylim3d(bottom=-0.5, top=1.5) 
# ax.axes.set_zlim3d(bottom=-1, top=1)
# plt.show()

##########################################################################

#3d Design GUI Example
##                       [ class ] [sweep] [spanMod] [shape] [chord]  [offset]
##Plot Pseudo2D Example: [0.5, 1.0, 0.0,   0.0, 0.0, 1.0x18, 1.0, 1.0, 0.0]
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
coords = np.append(uCoords,lCoords,0)

wing = CSTMultiParam()
wing.attachCSTParam(upper, 'wing_upper')
wing.attachCSTParam(lower, 'wing_lower')
wing.attachPoints(coords, 'wing')
wing.embeddedParams['wing_lower'].addConstraint('linked_wing', wing.embeddedParams['wing_upper'], [2,3,4,23,24], [2,3,4,23,24])

p = PyrocDesign(wing,mode='3d')
p.root.mainloop()