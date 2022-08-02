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
        2D reference axes - First row is psi axis, second is zeta axis
        By default refAxes=ndarray([[1.0,0.0,0.0],
                                    [0.0,0.0,1.0]])

    TODO Rotation/Extrusion/Modification definition modes encompassed - Different coordinate systems
         Multiple sections (Piecewise functions) - Higher level of abstraction
         Axes variation?
    """

    def __init__(self, surface, csClassFunc=None, spanClassFunc=None, refLenFunc=None, csOffsetFunc=None,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], shapeCoeffs=[], chordCoeffs=[1.0], offsetCoeffs=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=[[1.0,0.0,0.0],[0.0,1.0,0.0]], shapeScale=1.0):
        ##Coordinates
        #Original coordinates. Used for comparing fit, ie printing fit residuals
        self.origSurface = surface
        #Number of coordinate points
        self.nPts = len(self.origSurface[:,0])
        #Internally stored, updated coordinates
        self.surface = self.origSurface

        ##Reference values/functions
        self.csClassFunc = self.csClassFunc
        #Reference chord function
        self.refLen = self.defaultChordFunc if refLenFunc is None else refLenFunc
        #Cross section offset function
        self.csOffsetFunc = self.defaultOffsetFunc if csOffsetFunc is None else csOffsetFunc

        #Order of shape functions, first is order of cs shape func, second is order of span
        self.order = order
        #Reference length - extrusion distance/longitudinal distance
        self.refSpan = refSpan
        #Origin
        self.origin = np.array(origin)
        #Reference Axis - extrusion direction/rotation axis
        self.refAxes = np.array(refAxes)

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
        #By default, first chordCoeffs coeff is chord length
        coeffs = coeffs[0]
        return np.ones(len(etaVals))*coeffs[0]

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
        
    #calculate psi values from eta, zeta
    def calcPsi(self, etaVals, zetaVals):
        pass

    #calculate eta values form psi, zeta
    def calcEta(self, psiVals, zetaVals):
        pass
    
    #Explicitly calculate zeta values from psi, eta
    ##Different for each mode?
    def calcZeta(self, psiVals, etaVals):
        pass

    #Update zeta values from internal psi and eta values
    def updateZeta(self):
        self.psiEtaZeta[:,2] = self.calcZeta(self.psiEtaZeta[:,0], self.psiEtaZeta[:,0])
        return self.psiEtaZeta

    def transformSurface(self, surface):
        transSurface = surface - self.origin
        return transSurface

    def untransformSurface(self, transSurface):
        surface = transSurface + self.origin
        return surface

    #Calculate cartesian of any parametric set and return
    def calcSurface(self, psiEtaZeta):
        transSurface = np.vstack([self.calcPsiEta2X(psiEtaZeta[:,0], psiEtaZeta[:,1]),
                                  self.calcEta2Y(psiEtaZeta[:,1]),
                                  self.calcEtaZeta2Z(psiEtaZeta[:,1], psiEtaZeta[:,2])]).T
        surface = self.untransformSurface(transSurface)
        return surface

    #Update internal coordinates from internal parametric coords and return
    def updateSurface(self):
        self.updateZeta()
        self.surface = self.calcSurface(self.psiEtaZeta)
        return self.surface

    #Calculate psi,eta,zeta from surface
    def surface2PsiEtaZeta(self, surface):
        transSurface = self.transformSurface(surface)
        psiEtaZeta =  np.vstack([self.calcXY2Psi(transSurface[:,0], transSurface[:,1]),
                                 self.calcY2Eta(transSurface[:,1]),
                                 self.calcYZ2Zeta(transSurface[:,1],transSurface[:,2])]).T
        return psiEtaZeta

    #Set psi/eta values and update zeta
    def setPsiEtaZeta(self, psiVals, etaVals):
        zetaVals = self.calcZeta(psiVals, etaVals)
        self.psiEtaZeta = np.vstack([psiVals,etaVals,zetaVals]).T
        return self.psiEtaZeta

    #Calculate x vals from y and z vals
    #Will have multiple solutions?
    def calcYZ2X(self, yVals, zVals):
        etaVals = self.calcY2Eta(yVals)
        xVals = self.calcPsiEta2X(self.calcPsi(etaVals,self.calcYZ2Zeta(yVals,zVals)),etaVals)
        return xVals

    #Calculate y vals from x and z vals
    #Will have multiple solutions? Pointless?
    def calcXZ2Y(self, xVals, zVals):
        pass

    #Calculate z vals from x and y vals
    def calcXY2Z(self, xVals, yVals):
        etaVals = self.calcY2Eta(yVals)
        zVals = self.calcEtaZeta2Z(etaVals,self.calcZeta(self.calcXY2Psi(xVals,yVals),etaVals))
        return zVals

    #Update internal coefficients from coefficients list,
    #internal coefficient list length must be same as the one being set
    #Order of coeffs is [csClass, csMod, spanClass, shape, chord, offset]
    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        nC = len(coeffs)
        n1 = len(self.csClassCoeffs)
        n2 = n1+len(self.csModCoeffs)
        n3 = n2+len(self.spanClassCoeffs)
        n4 = n3+len(self.shapeCoeffs)
        n5 = n4+len(self.chordCoeffs)
        for _ in range(nC):
            if not self.masks[_]:
                if _<n1:
                    self.csClassCoeffs[_] = coeffs[_]
                elif _<n2 and _>=n1:
                    i = _-n1
                    self.csModCoeffs[i] = coeffs[_]
                elif _<n3 and _>=n2:
                    i = _-n2
                    self.spanClassCoeffs[i] = coeffs[_]
                elif _<n4 and _>=n3:
                    i = _-n3
                    self.shapeCoeffs[i] = coeffs[_]
                elif _<n5 and _>=n4:
                    i = _-n4
                    self.chordCoeffs[i] = coeffs[_]
                elif _>=n5:
                    i = _-n5
                    self.offsetCoeffs[i] = coeffs[_]
                else:
                    raise Exception("Value error: incorrect number of coefficients")
        return 0

    #Return an ordered list of coeffs from internal coeff values
    #Order of coeffs is [csClass, csMod, spanClass, shape, chord, offset]
    def getCoeffs(self):
        coeffs = []
        for _ in self.csClassCoeffs:
            coeffs.append(_)
        for _ in self.csModCoeffs:
            coeffs.append(_)
        for _ in self.spanClassCoeffs:
            coeffs.append(_)
        for _ in self.shapeCoeffs:
            coeffs.append(_)
        for _ in self.chordCoeffs:
            coeffs.append(_)
        for _ in self.offsetCoeffs:
            coeffs.append(_)
        return coeffs

    #Calculate gradient of zeta(psi,eta) (FD)
    def calcGrad(self, psiEtaZeta, h=1e-8):
        pass

    #Get gradient of pts
    def getGrad(self):
        return self.calcGrad(self.psiEtaZeta)

    #Calculate dPsiEtaZetadCoeff Jacobian (FD)
    def calcJacobian(self, psiEtaZeta, h=1e-8):
        pass

    #Get dPsiEtaZetadCoeff Jacobian
    def getJacobian(self):
        return self.calcJacobian(self.psiEtaZeta)

    #Perform a fit of the coefficients
    def fit3d(self):
        def surface(xyVals, *coeffs):
            self.updateCoeffs(coeffs)
            return self.calcXY2Z(xyVals[:,0], xyVals[:,1])
        #TODO Fix warning for cov params being inf in certain conditions
        coeffs,cov = scp.curve_fit(surface,self.surface[:,(0,1)],self.surface[:,2],self.getCoeffs())
        self.updateCoeffs(coeffs)
        self.updateZeta()
        return 0

    def printFitResiduals(self):
        err = [np.linalg.norm(self.origSurface[_]-self.surface[_]) for _ in range(self.nPts)]
        print('Residual statistics\n',
              'Avg Diff: ',np.average(err),'\n',
              'Max Diff: ',np.max(err),'\n',
              'Total Diff: ',np.sum(err),'\n')
        return 0
        

class CSTAirfoil3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded airfoil surface (Pseudo 2D)
    """
    def __init__(self, surface, csClassFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], offsetCoeffs=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0):
        super().__init__(surface=surface, csClassFunc=csClassFunc, spanClassFunc=None, refLenFunc=None, csOffsetFunc=None,
                         csClassCoeffs=csClassCoeffs, spanClassCoeffs=[], shapeCoeffs=shapeCoeffs, chordCoeffs=[1.0], offsetCoeffs=offsetCoeffs, masks=masks,
                         order=order, refSpan=refSpan, origin=[0.0,0.0,0.0], refAxes=refAxes, shapeScale=shapeScale)
        
        self.csGeo = CSTAirfoil2D(surface[:,(0,2)], classFunc=self.csClassFunc, 
                                  classCoeffs=self.csClassCoeffs, shapeCoeffs=self.shapeCoeffs, masks=self.masks,
                                  order=order[0], shapeOffset=self.csOffsetFunc(self.offsetCoeffs), shapeScale=shapeScale)

    def calcPsi(self, etaVals, zetaVals):
        return self.csGeo.calcPsi(zetaVals)

    #Calc eta cannot be and does not need to be implemented, no dependency

    def calcZeta(self, psiVals, etaVals):
        return self.csGeo.calcZeta(psiVals)

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.shapeCoeffs + [self.csOffsetFunc(self.offsetCoeffs)]
        self.csGeo.updateCoeffs(csCoeffs)
        return 0

    def calcGrad(self, psiEtaZeta, h=1e-8):
        psiVals = psiEtaZeta[:,0]; nPts = len(psiVals)
        return np.vstack([self.csGeo.calcDeriv(psiVals, h), np.zeros(nPts)]).T

    #dPsiEtaZetadCoeff
    def calcJacobian(self, psiEtaZeta, h=1e-8):
        dZetadCoeff = self.csGeo.calcJacobian(psiEtaZeta[:,0],h)
        jacTotal = np.zeros([3*_ for _ in dZetadCoeff.shape])
        jacTotal[2::3,2::3] = dZetadCoeff
        return jacTotal


class CSTWing3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded airfoil (Pseudo 2D)
    """
    def __init__(self, surface, csClassFunc=None, refLenFunc=None, csOffsetFunc=None,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], shapeCoeffs=[], chordCoeffs=[1.0], offsetCoeffs=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0):
        super().__init__(self, surface=surface, csClassFunc=csClassFunc, spanClassFunc=None, refLenFunc=refLenFunc, csOffsetFunc=csOffsetFunc,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], shapeCoeffs=[], chordCoeffs=[1.0], offsetCoeffs=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0)

    class CSTWing2D(CSTAirfoil2D):
        """
        Sub-class needed to modify shape coefficients as function of eta
        """
        def __init__():
            pass
