import scipy.optimize as scp
import numpy as np
from .cst2d import *

class CST3DParam(object):
    """
    Baseclass for storing cst parameterization information in 3d

    Parameters
    ----------
    surface : ndarray
        Boundary surface to parameterize
        must be x,y,z=[N,3]

    refSpan : float
        Reference length in extrusion direction or along rotation axis

    refAxes : ndarray
        Reference axes - Use defined in subclass?
            If rotation: first row initial theta axis, second is rotation axis
            If extrusion: first row  is initial psi axis, second is eta axis
        By default refAxes=ndarray([[1.0,0.0,0.0],
                                    [0.0,1.0,0.0]])

    TODO Rotation/Extrusion/Modification definition modes encompassed
         Multiple sections (Piecewise functions) - Higher level of abstraction
         Axes variation?
    """

    def __init__(self, surface, csClassFunc=None, spanClassFunc=None, refLenFunc=None, csOffsetFunc=None,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], shapeCoeffs=[], chordCoeffs=[], offsetCoeffs=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0):
        ##Coordinates
        #Original coordinates. Used for comparing fit, ie printing fit residuals
        self.origSurface = surface
        #Number of coordinate points
        self.nPts = len(self.origSurface[:,0])
        #Internally stored, updated coordinates
        self.surface = self.origSurface

        ##Reference values/functions
        #Reference chord function
        self.refLen = self.defaultChordFunc if refLenFunc is None else refLenFunc
        #Cross section offset function
        self.csOffsetFunc = self.defaultOffsetFunc if csOffsetFunc is None else csOffsetFunc
        #Reference length - extrusion distance/longitudinal distance
        self.refSpan = refSpan
        #Reference Axis - extrusion direction/rotation axis
        self.refAxes = refAxes
        #Order of shape functions, first is order of cs shape func, second is order of span
        self.order = order

        #Transform the original surface coordinates to re-define them in terms of reference axis
        #Rotation vs extrusion?

        ##Design Variable Coefficients
        #Cross Section Class Function coefficients
        self.csClassCoeffs = csClassCoeffs
        #Cross section modification coefficients
        self.csModCoeffs = csModCoeffs
        #Span Class Function Coefficients
        self.spanClassCoeffs = spanClassCoeffs
        #Shape Function Coefficients
        if len(shapeCoeffs)==self.order[0]+1 and len(shapeCoeffs[0])==self.order[1]+1:
            self.shapeCoeffs = shapeCoeffs
        else:
            self.shapeCoeffs = [shapeScale for _ in range((self.order[0]+1)*(self.order[1]+1))]
        #Reference length function coefficients
        self.chordCoeffs = chordCoeffs
        #Offset Coefficients
        self.offsetCoeffs = offsetCoeffs

        #Variable Masks
        self.nCoeff = len(self.getCoeffs())
        self.masks = [0 for _ in range(self.nCoeff)] if len(masks)==0 else masks

        #Set initial parameterization values from original coordinates
        self.psiEtaZeta = self.surface2PsiEtaZeta(self.origSurface)

    #Default function to define chord length along eta
    def defaultChordFunc(self, etaVals, *coeffs):
        coeffs = coeffs[0]
        return np.ones(len(etaVals))

    #Default function to define offset (blunt TE thickness along span)
    def defaultOffsetFunc(self, etaVals, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]

    #Calculate psi vals from x and y vals
    def calcXY2Psi(self, xVals, yVals):
        etaVals = self.calcY2Eta(yVals)
        return xVals/self.refLen(etaVals,self.chordCoeffs)

    #Calculate eta vals from y vals
    def calcY2Eta(self, yVals):
        return yVals/self.refSpan

    #Calculate zeta vals from y and z vals
    def calcYZ2Zeta(self, yVals, zVals):
        etaVals = self.calcY2Eta(yVals)
        return zVals/self.refLen(etaVals,self.chordCoeffs)

    #calculate x vals from psi and zeta vals
    def calcPsiEta2X(self, psiVals, etaVals):
        return psiVals*self.refLen(etaVals,self.chordCoeffs)

    #Calculate y vals from eta vals
    def calcEta2Y(self, etaVals):
        return etaVals*self.refSpan

    #Calculate z vals from eta and zeta vals
    def calcEtaZeta2Z(self, etaVals, zetaVals):
        return zetaVals*self.refLen(etaVals,self.chordCoeffs)
        
    #Explicitly calculate zeta values
    ##Different for each mode?
    def calcZeta(self, psiVals, etaVals):
        pass

    #Update zeta values from internal psi and eta values
    def updateZeta(self):
        self.psiEtaZeta[:,2] = self.calcZeta(self.psiEtaZeta[:,0], self.psiEtaZeta[:,0])
        return self.psiEtaZeta

    #Calculate cartesian of any parametric set and return
    def calcSurface(self, psiEtaZeta):
        transSurface = np.vstack([self.calcPsiEta2X(psiEtaZeta[:,0]),self.calcEta2Y(psiEtaZeta[:,1]),self.calcEtaZeta2Z(psiEtaZeta[:,2])]).T
        return transSurface #Need to un-transform surface

    #Update internal coordinates from internal parametric coords and return
    def updateSurface(self):
        self.updateZeta()
        self.surface = self.calcSurface(self.psiEtaZeta)
        return self.surface

    #Calculate psi,eta,zeta from surface
    def surface2PsiEtaZeta(self, surface):
        transSurface = surface #Need to transform surface
        return np.vstack([self.calcXY2Psi(transSurface[:,0]),self.calcY2Eta(transSurface[:,1]),self.calcYZ2Zeta(transSurface[:,2])]).T

    #Set psi/eta values and update zeta
    def setPsiEtaZeta(self, psiVals, etaVals):
        zetaVals = self.calcZeta(psiVals, etaVals)
        self.psiEtaZeta = np.vstack([psiVals,etaVals,zetaVals]).T
        return self.psiEtaZeta

    #Calculate z vals from x and y vals
    def calcXY2Z(self, xVals, yVals):
        etaVals = self.calcY2Eta(yVals)
        zVals = self.calcEtaZeta2Z(etaVals,self.calcZeta(self.calcXY2Psi(xVals,yVals),etaVals))
        return zVals

    #Calculate x and y vals from z vals
    #def calcZ2XY(self, zVals):
    #    

    #Update internal coefficients from coefficients list,
    #internal coefficient list length must be same as the one being set
    #Order of coeffs is [csClass, csMod, spanClass, shape, chord, offset]
    def updateCoeffs(self, *coeffs):
        pass

    #Return an ordered list of coeffs from internal coeff values
    #Order of coeffs is []
    def getCoeffs(self):
        pass

    def calcGrad(self, psiVals, etaVals, h=1e-8):
        pass
    def getGrad(self):
        pass
    def calcJacobian(self, psiVals, etaVals, h=1e-8):
        pass
    def getJacobian(self):
        pass
    def fit3d(self):
        pass
    def printFitResiduals(self):
        pass
        

class CSTAirfoil3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded airfoil (Pseudo 2D)
    """
    def __init__(self, surface, csClassFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], offsetCoeffs=[], masks=[],
                 order=[5,0], refSpan=1.0, refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0):
        super().__init__(surface=surface, refLenFunc=None, csClassFunc=csClassFunc, spanClassFunc=None,
                         csClassCoeffs=csClassCoeffs, spanClassCoeffs=[], shapeCoeffs=shapeCoeffs, chordCoeffs=[], offsetCoeffs=offsetCoeffs, masks=masks,
                         order=order, refSpan=refSpan, refAxes=refAxes)
        
        self.csParam = CSTAirfoil2D(surface[:,(0,2)], classFunc=csClassFunc, classCoeffs=csClassCoeffs, shapeCoeffs=shapeCoeffs, masks=masks,
                                    order=order[0], shapeOffset=0.0, shapeScale=shapeScale)

    def calcZeta(self, psiVals, etaVals):
        self.csParam.calcZeta(psiVals)

# class CSTWing3D(CST3DParam):
#     """
#     Class for storing cst parameterization information for extruded airfoil (Pseudo 2D)
#     """
#     def __init__():
#         pass

