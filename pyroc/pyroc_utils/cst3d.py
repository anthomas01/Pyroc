import scipy.optimize as scp
import numpy as np
from collections import OrderedDict
from .cst2d import *

class CST3DParam(object):
    """
    Baseclass for storing cst parameterization information in 3d

    Parameters
    ----------
    surface : ndarray [nPts, 3]
        Boundary surface to parameterize

    csClassFunc : function [zeta]=f([nPsiEtaZeta, 3], *coeffs)
        Function that defines class of shapes ie airfoil
        Not implemented by default

    csModFunc : function [X]=f([nPsiEtaZeta, 3], *coeffs)
        Function that defines change to cross section, ie leading edge sweep modification
        Not implemented by default

    spanClassFunc: function [eta]=f([nPsiEtaZeta, 3], *coeffs)
        Function that defines class function for span extrusion
        Not implemented by default

    spanModFunc: function [zeta]=f([nPsiEtaZeta, 3], *coeffs)
        Function that modifies zeta along span from intial class/shape functions
        Not implemented by default

    refLenFunc: function [C]=f([nPsiEtaZeta, 3], *coeffs)
        Function that defines reference lengths for psi/zeta based on parameterized values
        By default, returns constant 1.0 for [nPts]

    csClassCoeffs: list of float
        Coefficients for cross section class function

    csModCoeffs: list of float
        Coefficients for cross section modification function

    spanClassCoeffs: list of float
        Coefficients for spanwise class function

    spanModCoeffs: list of float
        Coefficients for spanwise modification

    shapeCoeffs: list of float
        Coefficients for augmenting Bernstein polynomials
        Must have length of (order[0]+1)*(order[1]+1)
        By default, all ones

    chordCoeffs: list of float
        Coefficients for refLenFunc
        By default, [1.0]

    shapeOffsets: list of float
        Coefficients for offset function, which by default returns shapeOffsets[0] for [nPts]
        By default, [0.0]

    masks: list of bool/int
        Must be length of used coefficients
        A 1/True indicates a value is masked, a 0/False indicates it is free
        If dv is masked, one cannot change dv from init value with updateCoeffs() method

    order: list of int
        By default, first integer is order for psi dependent shape function
        second integer is order for eta dependent shape function

    refSpan : float
        Reference length for eta
        By default, 1.0 or 2pi for bodies of revolution

    origin : list of float [1,3]
        Origin for geometry object
        By default, [0.0,0.0,0.0]

    refAxes : ndarray [3,3]
        Reference axes - First row is psi axis, second is eta axis, third is zeta axis
        - For an extruded body (cartesian coords)
            - psi is the axis along chord
            - eta is axis along span
            - zeta is normal to both
        - For a revolved body (cylindrical coords)-
            - psi is the axis of rotation and chord axis for cross section
            - eta is angular axis
            - zeta is the radial axis
        - By default, [[1.0,0.0,0.0],
                       [0.0,1.0,0.0],
                       [0.0,0.0,1.0]]

    shapeScale: float
        If shapeCoeffs is not given, this value will be used for initial shape coeffs.
        shapeScale=-1.0 can be used for a -z surface initialization
        By default, 1.0

    - TODO Multiple sections - Higher level of abstraction?
    - TODO Overall abstraction/simplification, bugfixes
    - TODO Use internal derivatives with scipy optimization
    """

    def __init__(self, surface, csClassFunc=None, csModFunc=None, spanClassFunc=None, spanModFunc=None, refLenFunc=None,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], spanModCoeffs=[], shapeCoeffs=[], chordCoeffs=[1.0], 
                 shapeOffsets=[0.0], masks=[], order=[5,0], refSpan=1.0, origin=np.zeros(3), refAxes=np.eye(3), shapeScale=1.0):
        ##Coordinates
        #Original coordinates. Used for comparing fit, ie printing fit residuals
        self.origSurface = np.copy(surface)
        #Number of coordinate points
        self.nPts = len(self.origSurface[:,0])
        #Internally stored, updated coordinates
        self.surface = np.copy(self.origSurface)

        ##Reference values/functions
        #cross section class function
        self.csClassFunc = csClassFunc
        #Cross section modification
        self.csModFunc = csModFunc
        #span class function
        self.spanClassFunc = spanClassFunc
        #span modification
        self.spanModFunc = spanModFunc
        #Reference chord function
        self.refLen = self.defaultChordFunction if refLenFunc is None else refLenFunc

        self.offsetFunc = self.defaultOffsetFunction

        #Order of shape functions, first is order of cs shape func, second is order of span
        self.order = order.copy()
        #Reference length - extrusion distance/longitudinal distance
        self.refSpan = float(refSpan)
        #Origin
        self.origin = np.copy(origin)
        #Reference Axis - extrusion direction/rotation axis
        self.refAxes = np.copy(refAxes)

        ##Design Variable Coefficients
        #Cross Section Class Function coefficients
        self.csClassCoeffs = csClassCoeffs.copy()
        #Cross section modification coefficients
        self.csModCoeffs = csModCoeffs.copy()
        #Span Class Function Coefficients
        self.spanClassCoeffs = spanClassCoeffs.copy()
        #Span modification function coefficients
        self.spanModCoeffs = spanModCoeffs.copy()
        #Shape Function Coefficients
        numShape = (self.order[0]+1)*(self.order[1]+1)
        if len(shapeCoeffs)==numShape:
            self.shapeCoeffs = shapeCoeffs.copy()
        else:
            self.shapeCoeffs = [float(shapeScale) for _ in range(numShape)]
        #Reference length function coefficients
        self.chordCoeffs = chordCoeffs.copy()
        #Offset Coefficients
        self.shapeOffsets = shapeOffsets.copy()

        #Variable Masks
        self.masks = []
        nCoeffs = len(self.getCoeffs())
        for _ in range(nCoeffs):
            if _ in masks or _-nCoeffs in masks:
                self.masks.append(1)
            else:
                self.masks.append(0)

        #Set initial parameterization values from original coordinates
        self.psiEtaZeta = self.coords2PsiEtaZeta(self.origSurface)

    #Default function to define chord length along eta
    def defaultChordFunction(self, psiEtaZeta, *coeffs):
        #By default, first chordCoeffs coeff is chord length
        coeffs = coeffs[0]
        return np.ones(len(psiEtaZeta[:,1]))*coeffs[0]

    #Default function to define offset (blunt TE thickness along span)
    def defaultOffsetFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return np.ones(len(psiEtaZeta[:,1]))*coeffs[0]

    #TODO Make these functions matrix based? eg.
    # [ psi]   [ psi/x  psi/y  psi/z]   [x]
    # [ eta] = [ eta/x  eta/y  eta/z] @ [y]
    # [zeta]   [zeta/x zeta/y zeta/z]   [z]
    #Calculate psi vals from x and y vals
    def calcXYZ2Psi(self, xyz):
        etaVals = self.calcXYZ2Eta(xyz)
        psiEtaZeta = np.vstack([np.zeros_like(etaVals), etaVals, np.zeros_like(etaVals)]).T
        return xyz[:,0]/self.refLen(psiEtaZeta, self.chordCoeffs)

    #Calculate eta vals from y vals
    def calcXYZ2Eta(self, xyz):
        return xyz[:,1]/self.refSpan

    #Calculate zeta vals from x, y and z vals
    def calcXYZ2Zeta(self, xyz):
        etaVals = self.calcXYZ2Eta(xyz)
        psiEtaZeta = np.vstack([np.zeros_like(etaVals), etaVals, np.zeros_like(etaVals)]).T
        return xyz[:,2]/self.refLen(psiEtaZeta, self.chordCoeffs)

    #calculate x vals from psi and zeta vals
    def calcPsiEtaZeta2X(self, psiEtaZeta):
        return psiEtaZeta[:,0]*self.refLen(psiEtaZeta,self.chordCoeffs)

    #Calculate y vals from eta vals
    def calcPsiEtaZeta2Y(self, psiEtaZeta):
        return psiEtaZeta[:,1]*self.refSpan

    #Calculate z vals from eta and zeta vals
    def calcPsiEtaZeta2Z(self, psiEtaZeta):
        return psiEtaZeta[:,2]*self.refLen(psiEtaZeta,self.chordCoeffs)
        
    #Estimates psi vals from zeta values
    ##TODO Broken, needs to have better initial guess
    def calcPsi(self, psiEtaZeta):
        def objFunc(psiVals):
            psiEtaZetaH = psiEtaZeta
            psiEtaZetaH[:,0] = psiVals
            return self.calcZeta(psiEtaZetaH)-psiEtaZetaH[:,2]
        psiVals = scp.fsolve(objFunc, 0 if isScalar(psiEtaZeta[0]) else np.zeros(len(psiEtaZeta[:,0])))
        return psiVals

    def calcEta(self, psiEtaZeta):
        def objFunc(etaVals):
            psiEtaZetaH = psiEtaZeta
            psiEtaZetaH[:,1] = etaVals
            return self.calcZeta(psiEtaZetaH)-psiEtaZetaH[:,2]
        etaVals = scp.fsolve(objFunc, 0 if isScalar(psiEtaZeta[0]) else np.zeros(len(psiEtaZeta[:,0])))
        return etaVals
    
    #Explicitly calculate zeta values from psi, eta
    ##Different for each inherited class
    def calcZeta(self, psiEtaZeta):
        return np.zeros(len(psiEtaZeta))

    #Update zeta values from internal psi and eta values
    def updateZeta(self):
        self.psiEtaZeta[:,2] = self.calcZeta(self.psiEtaZeta)
        return self.psiEtaZeta

    #xyz->uvw
    def _calcTransformMatrix(self, transSurface):
        transMats = np.zeros((len(transSurface.flatten()), 3))
        for _ in range(len(transSurface)):
            transMats[3*_:3*(_+1),:] = self.refAxes.T
        return transMats

    #uvw->xyz
    def _calcInverseTransform(self, surface):
        transMats = np.zeros((len(surface.flatten()), 3))
        for _ in range(len(surface)):
            transMats[3*_:3*(_+1),:] = np.linalg.inv(self.refAxes.T)
        return transMats

    #TODO Make transforms more efficient
    #Transform surface coordinates into axes used by class functions
    #uvw->xyz
    def transformSurface(self, surface):
        numPts = len(surface)
        transSurface = np.zeros_like(surface)
        transMats = self._calcInverseTransform(surface)
        for _ in range(numPts):
            transMat = transMats[3*_:3*(_+1),:]
            transSurface[_,:] = transMat @ (surface[_,:] - self.origin)
        return transSurface

    #Untransform coordinates into given reference axes
    #xyz->uvw
    def untransformSurface(self, transSurface):
        numPts = len(transSurface)
        surface = np.zeros_like(transSurface)
        transMats = self._calcTransformMatrix(transSurface)
        for _ in range(numPts):
            transMat = transMats[3*_:3*(_+1),:]
            surface[_,:] = (transMat @ transSurface[_,:]) + self.origin
        return surface

    #Update Zeta, calculate cartesian of parametric set and return
    def calcCoords(self, psiEtaZeta):
        transSurface = np.vstack([self.calcPsiEtaZeta2X(psiEtaZeta),
                                  self.calcPsiEtaZeta2Y(psiEtaZeta),
                                  self.calcPsiEtaZeta2Z(psiEtaZeta)]).T
        surface = self.untransformSurface(transSurface)
        return surface

    #Update internal coordinates from internal parametric coords and return
    def updateCoords(self):
        self.updateZeta()
        self.surface = self.calcCoords(self.psiEtaZeta)
        return self.surface

    #Calculate psi,eta,zeta from surface
    def coords2PsiEtaZeta(self, surface):
        transSurface = self.transformSurface(surface)
        psiEtaZeta =  np.vstack([self.calcXYZ2Psi(transSurface),
                                 self.calcXYZ2Eta(transSurface),
                                 self.calcXYZ2Zeta(transSurface)]).T
        return psiEtaZeta

    #Set psi/eta values and update zeta
    def setPsiEtaZeta(self, psiEtaZeta):
        self.psiEtaZeta = np.vstack([psiEtaZeta[:,0], psiEtaZeta[:,1], self.calcZeta(psiEtaZeta)]).T
        return self.psiEtaZeta

    #Calculate x vals from y and z vals
    #Broken will have multiple solutions
    def calcYZ2X(self, yVals, zVals):
        xyz = np.vstack([np.zeros_like(yVals), yVals, zVals]).T
        etaVals = self.calcXYZ2Eta(xyz)
        psiEtaZeta = np.vstack([np.zeros_like(etaVals), etaVals, self.calcXYZ2Zeta(xyz)]).T
        psiEtaZeta[:,0] = self.calcPsi(psiEtaZeta)
        xVals = self.calcPsiEtaZeta2X(psiEtaZeta)
        return xVals

    #Calculate y vals from x and z vals
    ##Different for each inherited class
    def calcXZ2Y(self, xVals, zVals):
        return np.zeros_like(xVals)

    #Calculate z vals from x and y vals
    def calcXY2Z(self, xVals, yVals):
        xyz = np.vstack([xVals, yVals, np.zeros_like(xVals)]).T
        etaVals = self.calcXYZ2Eta(xyz)
        psiEtaZeta = np.vstack([self.calcXYZ2Psi(xyz), etaVals, np.zeros_like(etaVals)]).T
        psiEtaZeta[:,2] = self.calcZeta(psiEtaZeta)
        zVals = self.calcPsiEtaZeta2Z(psiEtaZeta)
        return zVals

    #Update internal coefficients from coefficients list,
    #internal coefficient list length must be same as the one being set
    #Order of coeffs is [csClass, csMod, spanClass, spanMod, shape, chord, offset]
    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        nC = len(coeffs)
        n1 = len(self.csClassCoeffs)
        n2 = n1+len(self.csModCoeffs)
        n3 = n2+len(self.spanClassCoeffs)
        n4 = n3+len(self.spanModCoeffs)
        n5 = n4+len(self.shapeCoeffs)
        n6 = n5+len(self.chordCoeffs)
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
                    self.spanModCoeffs[i] = coeffs[_]
                elif _<n5 and _>=n4:
                    i = _-n4
                    self.shapeCoeffs[i] = coeffs[_]
                elif _<n6 and _>=n5:
                    i = _-n5
                    self.chordCoeffs[i] = coeffs[_]
                elif _>=n6:
                    i = _-n6
                    self.shapeOffsets[i] = coeffs[_]
                else:
                    raise Exception("Value error: incorrect number of coefficients")
        return 0

    #Return an ordered list of coeffs from internal coeff values
    #Order of coeffs is [csClass, csMod, spanClass, spanMod, shape, chord, offset]
    def getCoeffs(self):
        coeffs = []
        for _ in self.csClassCoeffs:
            coeffs.append(_)
        for _ in self.csModCoeffs:
            coeffs.append(_)
        for _ in self.spanClassCoeffs:
            coeffs.append(_)
        for _ in self.spanModCoeffs:
            coeffs.append(_)
        for _ in self.shapeCoeffs:
            coeffs.append(_)
        for _ in self.chordCoeffs:
            coeffs.append(_)
        for _ in self.shapeOffsets:
            coeffs.append(_)
        return coeffs

    #Calculate gradient of zeta(psi,eta) (FD)
    def calcParamsGrad(self, psiEtaZeta, h=1e-8):
        psiEtaZetaH = np.copy(psiEtaZeta)
        dZetadPsi = np.zeros_like(psiEtaZetaH)
        for _ in range(len(psiEtaZetaH)):
            if psiEtaZetaH[_,0] == 1.0:
                psiEtaZetaH[_,0] -= h
                dZetadPsi[_] = (self.calcZeta(psiEtaZeta) - self.calcZeta(psiEtaZetaH))/h
            else:
                psiEtaZetaH[_,0] += h
                dZetadPsi[_] = (self.calcZeta(psiEtaZetaH) - self.calcZeta(psiEtaZeta))/h

        psiEtaZetaH = np.copy(psiEtaZeta)
        dZetadEta = np.zeros_like(psiEtaZetaH)
        for _ in range(len(psiEtaZetaH)):
            if psiEtaZetaH[_,1] == 1.0:
                psiEtaZetaH[_,1] -= h
                dZetadPsi[_] = (self.calcZeta(psiEtaZeta) - self.calcZeta(psiEtaZetaH))/h
            else:
                psiEtaZetaH[_,1] += h
                dZetadEta[_] = (self.calcZeta(psiEtaZetaH) - self.calcZeta(psiEtaZeta))/h
        return np.vstack([dZetadPsi, dZetadEta]).T

    #Calculate gradient of z(x,y)
    def calcGrad(self, surface, h=1e-8):
        psiEtaZeta = self.coords2PsiEtaZeta(surface)
        paramsGrad = self.calcParamsGrad(psiEtaZeta, h)
        dZdX = paramsGrad[:,0]
        dZdY = self.refLen(psiEtaZeta, self.chordCoeffs) * paramsGrad[:,1] / self.refSpan
        return np.vstack([dZdX, dZdY]).T

    #Get gradient of pts
    def getGrad(self):
        return self.calcGrad(self.surface)

    #Calculate dPsiEtaZetadCoeff Jacobian
    def calcParamsJacobian(self, psiEtaZeta, h=1e-8):
        totalJac = []
        csClassJac = self._calcCsClassJacobian(psiEtaZeta, h)
        if csClassJac is not None: totalJac.append(csClassJac.T)
        csModJac = self._calcCsModJacobian(psiEtaZeta, h)
        if csModJac is not None: totalJac.append(csModJac.T)
        spanClassJac = self._calcSpanClassJacobian(psiEtaZeta, h)
        if spanClassJac is not None: totalJac.append(spanClassJac.T)
        spanModJac = self._calcSpanModJacobian(psiEtaZeta, h)
        if spanModJac is not None: totalJac.append(spanModJac.T)
        shapeJac = self._calcShapeJacobian(psiEtaZeta, h)
        if shapeJac is not None: totalJac.append(shapeJac.T)
        refChordJac = self._calcRefChordJacobian(psiEtaZeta, h)
        if refChordJac is not None: totalJac.append(refChordJac.T)
        shapeOffsetJac = self._calcShapeOffsetJacobian(psiEtaZeta, h)
        if shapeOffsetJac is not None: totalJac.append(shapeOffsetJac.T)
        return np.vstack(totalJac).T

    #Calculate dUVWdCoeffs Jacobian (FD)
    #dUVWdCoeff = dUVWdPsiEtaZeta * dPsiEtaZetadCoeff
    def calcJacobian(self, surface, h=1e-8):
        coeffs = self.getCoeffs()
        nCoeffs = len(coeffs)
        if nCoeffs>0 and len(surface)>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            totalJac = np.zeros((len(surface.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                coeffs[_] += h
                self.updateCoeffs(coeffs)
                psiEtaZeta[:,2] = self.calcZeta(psiEtaZeta)
                surfaceH = self.calcCoords(psiEtaZeta)
                for __ in range(3):
                    totalJac[__::3,3*_+__] = (surfaceH[:,__]-surface[:,__])/h
                coeffs[_] -= h
                self.updateCoeffs(coeffs)
        else:
            raise Exception('Class has no internal coefficients!')
        return totalJac

    #Get dXYZdCoeffs Jacobian
    def getJacobian(self):
        return self.calcJacobian(self.surface)

    #Perform a fit of the coefficients
    def fit3d(self, coords):
        def surface(xyVals, *coeffs):
            self.updateCoeffs(list(coeffs))
            return self.calcXY2Z(xyVals[:,0], xyVals[:,1])
        def jac(xyVals, *coeffs):
            surf = np.vstack([xyVals[:,0], xyVals[:,1], surface(xyVals, *coeffs)]).T
            return self.calcJacobian(surf)[2::3,2::3]
        #TODO Fix warning for cov params being inf in certain conditions
        coeffs,cov = scp.curve_fit(surface,coords[:,0:2],coords[:,2],np.atleast_1d(self.getCoeffs()),jac=jac)
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

    #dPsiEtaZetadCsClassCoeffs (FD)
    def _calcCsClassJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.csClassCoeffs)
        if nCoeffs>0:
            csClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.csClassCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    csClassJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.csClassCoeffs[_] -= h
        else:
            csClassJac = None
        return csClassJac

    #dPsiEtaZetadCsModCoeffs (FD)
    def _calcCsModJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.csModCoeffs)
        if nCoeffs>0:
            csModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.csModCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    csModJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.csModCoeffs[_] -= h
        else:
            csModJac = None
        return csModJac

    #dPsiEtaZetadSpanClassCoeffs (FD)
    def _calcSpanClassJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.spanClassCoeffs)
        if nCoeffs>0:
            spanClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.spanClassCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    spanClassJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.spanClassCoeffs[_] -= h
        else:
            spanClassJac = None
        return spanClassJac

    #dPsiEtaZetadSpanModCoeffs (FD)
    def _calcSpanModJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.spanModCoeffs)
        if nCoeffs>0:
            spanModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.spanModCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    spanModJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.spanModCoeffs[_] -= h
        else:
            spanModJac = None
        return spanModJac

    #dPsiEtaZetadShapeCoeffs (FD)
    def _calcShapeJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.shapeCoeffs)
        if nCoeffs>0:
            shapeJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.shapeCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    shapeJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.shapeCoeffs[_] -= h
        else:
            shapeJac = None
        return shapeJac

    #dPsiEtaZetadRefChordCoeffs (FD)
    # - Although default implementation has no dependency for PEZ, we dont know calcZeta implementation here
    def _calcRefChordJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.chordCoeffs)
        if nCoeffs>0:
            refChordJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.chordCoeffs[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    refChordJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.chordCoeffs[_] -= h
        else:
            refChordJac = None
        return refChordJac

    #dPsiEtaZetadShapeOffsetCoeffs (FD)
    # - Although default implementation has no dependency for PEZ, we dont know calcZeta implementation here
    def _calcShapeOffsetJacobian(self, psiEtaZeta, h=1e-8):
        surface = self.calcCoords(psiEtaZeta)
        nCoeffs = len(self.shapeOffsets)
        if nCoeffs>0:
            shapeOffsetJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                self.shapeOffsets[_] += h
                psiEtaZetaH = self.coords2PsiEtaZeta(surface)
                for __ in range(3):
                    shapeOffsetJac[__::3,3*_+__] = (psiEtaZetaH[:,__]-psiEtaZeta[:,__])/h
                self.shapeOffsets[_] -= h
        else:
            shapeOffsetJac = None
        return shapeOffsetJac

#################################################################################################

class CSTAirfoil3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded 'airfoil' surface (Pseudo 2D)
    """
    def __init__(self, surface, csClassFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], chordCoeffs=[1.0], shapeOffsets=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=np.zeros(3), refAxes=np.eye(3), shapeScale=1.0):

        self.csGeo = CSTAirfoil2D(surface[:,(0,2)], classFunc=csClassFunc, classCoeffs=csClassCoeffs,
                                  shapeCoeffs=shapeCoeffs, masks=masks, order=order[0],
                                  shapeOffset=shapeOffsets[0], refLen=chordCoeffs[0], shapeScale=shapeScale)

        super().__init__(surface=surface, csClassFunc=csClassFunc, spanClassFunc=None, refLenFunc=None,
                         csClassCoeffs=csClassCoeffs, csModCoeffs=[], spanClassCoeffs=[], spanModCoeffs=[], 
                         shapeCoeffs=shapeCoeffs, chordCoeffs=chordCoeffs, shapeOffsets=shapeOffsets, masks=masks,
                         order=order, refSpan=refSpan, origin=origin, refAxes=refAxes, shapeScale=shapeScale)

    def calcPsi(self, psiEtaZeta):
        return self.csGeo.calcPsi(psiEtaZeta[:,2])

    def calcEta(self, psiEtaZeta):
        return psiEtaZeta[:,1]

    def calcZeta(self, psiEtaZeta, zetaScale=None):
        zetaVals = self.csGeo.calcZeta(psiEtaZeta[:,0])

        #Implement param scale for above level connections
        if zetaScale is not None:
            zetaVals = zetaVals * zetaScale

        return zetaVals

    #Calculate psi,eta,zeta from surface
    def coords2PsiEtaZeta(self, surface):
        psiEtaZeta =  super().coords2PsiEtaZeta(surface)
        psiEtaZeta[:,0][psiEtaZeta[:,0]==0.0] = 1e-12
        return psiEtaZeta

    def calcXZ2Y(self, xVals, zVals):
        return 0

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.shapeCoeffs + [self.shapeOffsets[0]]
        self.csGeo.updateCoeffs(csCoeffs)
        self.csGeo.refLen = self.chordCoeffs[0]
        return

    #Analytical gradient of zeta
    def calcParamsGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.csGeo.calcDeriv(psiEtaZeta[:,0], h), np.zeros(len(psiEtaZeta))]).T

    #TODO make this more efficient
    #Calculate dXYZdCoeffs Jacobian (Analytical)
    #dXYZdCoeff = dXYZdPsiEtaZeta * dPsiEtaZetadCoeff
    def calcJacobian(self, surface, paramScale=None, h=1e-8):
        nPts = len(surface)
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0 and nPts>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            ptsJac = self._calcPtsJacobian(psiEtaZeta, h)
            paramsJac = self.calcParamsJacobian(psiEtaZeta, paramScale, h)
            totalJac = np.zeros((3*nPts, 3*nCoeffs))
            for _ in range(nPts):
                ptJac = ptsJac[3*_:3*(_+1),:]
                for __ in range(nCoeffs):
                    paramJac = paramsJac[3*_:3*(_+1), 3*__:3*(__+1)]
                    dXYZdCoeffVec = ptJac @ np.diagonal(paramJac)
                    totalJac[3*_:3*(_+1), 3*__:3*(__+1)] = np.diag(dXYZdCoeffVec)

        else:
            totalJac = None
        return totalJac

    #dPsiEtaZetadParams
    def calcParamsJacobian(self, psiEtaZeta, paramScale=None, h=1e-8):
        paramsJac = super().calcParamsJacobian(psiEtaZeta, h)

        #Implement param scale for above level connections
        #Param scale must be (N,3) where N is length of paramsJac
        if paramScale is not None:
            paramsJac = paramsJac * paramScale

        return paramsJac

    #dPsiEtaZetadCsClassCoeffs (Analytical)
    def _calcCsClassJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csClassCoeffs)
        if nCoeffs>0:
            csClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            csClassJac[2::3, 2::3] = self.csGeo._calcClassJacobian(psiEtaZeta[:,0], h)
        else:
            csClassJac = None
        return csClassJac

    #dPsiEtaZetadShapeCoeffs (Analytical)
    def _calcShapeJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.shapeCoeffs)
        if nCoeffs>0:
            shapeJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeJac[2::3, 2::3] = self.csGeo._calcShapeJacobian(psiEtaZeta[:,0], h)
        else:
            shapeJac = None
        return shapeJac

    #dPsiEtaZetadRefChordCoeffs (Analytical)
    def _calcRefChordJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.chordCoeffs)
        if nCoeffs>0:
            refChordJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
        else:
            refChordJac = None
        return refChordJac

    #dPsiEtaZetadShapeOffsetCoeffs (Analytical)
    def _calcShapeOffsetJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.shapeOffsets)
        if nCoeffs>0:
            shapeOffsetJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeOffsetJac[2::3, 2::3] = self.csGeo._calcOffsetJacobian(psiEtaZeta[:,0], h)
        else:
            shapeOffsetJac = None
        return shapeOffsetJac

    #Calc jacobian transformation matrix
    #                              [u]                               [x]
    # If reference coordinates are [v], and internal coordinates are [y]
    #                              [w]                               [z]
    #     [dU/dCoeff]   [dU/dX dU/dY dU/dZ] [dX/dCoeff]
    # i.e [dV/dCoeff] = [dV/dX dV/dY dV/dZ]*[dY/dCoeff]
    #     [dW/dCoeff]   [dW/dX dW/dY dW/dZ] [dZ/dCoeff]
    def _calcTransformJacobian(self, surface, h=1e-8):
        transformJac = np.zeros((len(surface.flatten()),3))
        for _ in range(len(surface)):
            #For a linear transformation, this is just the transpose of the reference axes
            transformJac[3*_:3*(_+1),:] = self.refAxes.T
        return transformJac

    #dXYZdPsiEtaZeta (analytical)
    def _calcPtsJacobian(self, psiEtaZeta, h=1e-8):
        ptsJac = np.zeros((len(psiEtaZeta.flatten()), 3))
        #transformation matrix from normalizations
        ptsJac[0::3, 0] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[1::3, 1] = self.refSpan
        ptsJac[2::3, 2] = self.refLen(psiEtaZeta, self.chordCoeffs)
        #Pre-multiply by transformation from coordinate axes
        surface = self.calcCoords(psiEtaZeta)
        transformJac = self._calcTransformJacobian(surface, h)
        for _ in range(len(psiEtaZeta)):
            ptsJac[3*_:3*(_+1),:] = transformJac[3*_:3*(_+1),:] @ ptsJac[3*_:3*(_+1),:]
        return ptsJac

#################################################################################################

class CSTWing3D(CST3DParam):
    """
    Class for storing cst parameterization information for wing section
    """
    def __init__(self, surface, csClassFunc=None, csModFunc=None, spanModFunc=None, refLenFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], sweepCoeffs=[0.0], shearCoeffs=[0.0],
                 twistCoeffs=[0.0], chordCoeffs=[1.0, 1.0], shapeOffsets=[0.0], masks=[],
                 order=[5,2], refSpan=1.0, origin=np.zeros(3), refAxes=np.eye(3), shapeScale=1.0):

        csClassFunc = self.defaultClassFunction if csClassFunc is None else csClassFunc
        csModFunc = self.defaultCsModFunction if csModFunc is None else csModFunc
        csModCoeffs = sweepCoeffs
        self.shapeFunc = self.defaultShapeFunction

        spanModFunc = self.defaultSpanModFunc if spanModFunc is None else spanModFunc
        spanModCoeffs = shearCoeffs+twistCoeffs
        self.nSpanModCoeffs = [len(shearCoeffs), len(twistCoeffs)]

        self.csGeo = CSTAirfoil2D(surface[:,(0,2)], classFunc=csClassFunc, 
                                  classCoeffs=csClassCoeffs, shapeCoeffs=[], masks=[],
                                  order=order[0], shapeOffset=shapeOffsets[0], shapeScale=shapeScale)

        super().__init__(surface=surface, csClassFunc=csClassFunc, csModFunc=csModFunc, spanClassFunc=None, spanModFunc=spanModFunc, 
                         refLenFunc=refLenFunc, csClassCoeffs=csClassCoeffs, csModCoeffs=csModCoeffs, spanClassCoeffs=[],
                         spanModCoeffs=spanModCoeffs, shapeCoeffs=shapeCoeffs, chordCoeffs=chordCoeffs, shapeOffsets=shapeOffsets,
                         masks=masks, order=order, refSpan=refSpan, origin=origin, refAxes=refAxes, shapeScale=shapeScale)

    #Airfoil Class function
    def defaultClassFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return self.csGeo.airfoilClassFunc(psiEtaZeta[:,0], coeffs)

    #2D shape function
    def defaultShapeFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0] #List of coefficients #ncoeff = (order0+1)*(order1+1)
        augments = bernstein2D(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1]) @ coeffs
        return augments

    #LE modification, default linear dist
    def defaultCsModFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return psiEtaZeta[:,1]*coeffs[0]

    #Shear and twist, default linear dist
    def defaultSpanModFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        shearCoeffs = coeffs[:self.nSpanModCoeffs[0]]
        twistCoeffs = coeffs[self.nSpanModCoeffs[0]:]
        return self._shearFunc(psiEtaZeta, shearCoeffs) - psiEtaZeta[:,0]*np.tan(self._twistFunc(psiEtaZeta, twistCoeffs))

    #Returns shear amount - default linear dist along eta
    def _shearFunc(self, psiEtaZeta, *coeffs):
            coeffs = coeffs[0]
            return psiEtaZeta[:,1]*coeffs[0]

    #Returns angle of twist (radians) - default linear dist along eta
    def _twistFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]*psiEtaZeta[:,1]

    #Function to define chord length along eta, default linear dist [rootChord, tipChord]
    def defaultChordFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]-psiEtaZeta[:,1]*(coeffs[0]-coeffs[1])

    def calcXYZ2Psi(self, xyz):
        xyz2 = np.copy(xyz)
        psiEtaZeta = np.vstack([np.zeros(len(xyz)), self.calcXYZ2Eta(xyz), np.zeros(len(xyz))]).T
        xyz2[:,0] -= self.csModFunc(psiEtaZeta, self.csModCoeffs)
        return super().calcXYZ2Psi(xyz2)

    def calcPsiEtaZeta2X(self, psiEtaZeta):
        return super().calcPsiEtaZeta2X(psiEtaZeta) + self.csModFunc(psiEtaZeta, self.csModCoeffs)

    #Explicitly calculate zeta values
    def calcZeta(self, psiEtaZeta):
        zetaVals = (psiEtaZeta[:,0]*self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                    self.spanModFunc(psiEtaZeta, self.spanModCoeffs) +
                    self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                    self.shapeFunc(psiEtaZeta, self.shapeCoeffs))
        return zetaVals

    def calcXZ2Y(self, xVals, zVals):
        psiEtaZeta = np.zeros((len(xVals),3))
        def objFunc(yVals):
            xyz = np.vstack([xVals, yVals, zVals]).T
            psiEtaZeta[:,0] = self.calcXYZ2Psi(xyz)
            psiEtaZeta[:,1] = self.calcXYZ2Eta(xyz)
            psiEtaZeta[:,2] = self.calcXYZ2Zeta(xyz)
            return self.calcZeta(psiEtaZeta)-psiEtaZeta[:,2]
        yVals = scp.fsolve(objFunc, 0 if isScalar(xVals) else np.zeros(len(xVals)))
        return yVals

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.csGeo.shapeCoeffs + [self.shapeOffsets[0]]
        self.csGeo.updateCoeffs(csCoeffs)
        return

    def _calcShapeGrad(self, psiEtaZeta, h=1e-8):
        dShapedPsi, dShapedEta = bernstein2DGrad(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1], h)
        dShapedPsi = dShapedPsi @ self.shapeCoeffs
        dShapedEta = dShapedEta @ self.shapeCoeffs
        return np.vstack([dShapedPsi.flatten(), dShapedEta.flatten()]).T

    def _calcCsClassGrad(self, psiEtaZeta, h=1e-8):
        numPts = len(psiEtaZeta)
        classGrad = np.zeros((numPts, 2))
        for _ in range(numPts):
            self.csGeo.refLen = self.refLen(psiEtaZeta, self.chordCoeffs)
            classGrad[_,0] = self.csGeo._calcClassDeriv(psiEtaZeta[_,0], h)
        return classGrad

    def _calcOffsetGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.offsetFunc(psiEtaZeta, self.shapeOffsets), np.zeros(len(psiEtaZeta))]).T

    def _calcShearGrad(self, psiEtaZeta, h=1e-8):
        shearCoeffs = self.spanModCoeffs[:self.nSpanModCoeffs[0]]
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), shearCoeffs[0]*np.ones(nPts)]).T

    def _calcTwistGrad(self, psiEtaZeta, h=1e-8):
        twistCoeffs = self.spanModCoeffs[self.nSpanModCoeffs[0]:]
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), twistCoeffs[0]*np.ones(nPts)]).T
            
    #Gradient of span modification terms
    def _calcSpanModGrad(self, psiEtaZeta, h=1e-8):
        twistCoeffs = self.spanModCoeffs[self.nSpanModCoeffs[0]:]
        dSpanModdPsi = -np.tan(self._twistFunc(psiEtaZeta, twistCoeffs))
        dSpanModdEta = (self._calcShearGrad(psiEtaZeta, h)[:,1] - 
                        psiEtaZeta[:,0]*self._calcTwistGrad(psiEtaZeta, h)[:,1]*
                        np.power(np.cos(self._twistFunc(psiEtaZeta, twistCoeffs)),-2.0))
        return np.vstack([dSpanModdPsi, dSpanModdEta]).T

    #Analytical gradient of zeta
    def calcParamsGrad(self, psiEtaZeta, h=1e-8):
        dZetadPsi = (self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                     self._calcSpanModGrad(psiEtaZeta, h)[:,0] +
                     self._calcCsClassGrad(psiEtaZeta, h)[:,0] * 
                     self.shapeFunc(psiEtaZeta, self.shapeCoeffs) +
                     self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                     self._calcShapeGrad(psiEtaZeta, h)[:,0])
        dZetadEta = (psiEtaZeta[:,0]*self._calcOffsetGrad(psiEtaZeta, h)[:,1] +
                     self._calcSpanModGrad(psiEtaZeta, h)[:,1] + 
                     self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                     self._calcShapeGrad(psiEtaZeta, h)[:,1])
        return np.vstack([dZetadPsi, dZetadEta]).T

    #TODO Make this more efficient
    #Calculate dXYZdCoeffs Jacobian (Analytical)
    #dXYZdCoeff = dXYZdPsiEtaZeta * dPsiEtaZetadCoeff + pXYZpCoeff
    def calcJacobian(self, surface, h=1e-8):
        nPts = len(surface)
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0 and nPts>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            ptsJac = self._calcPtsJacobian(psiEtaZeta, h)
            paramsJac = self.calcParamsJacobian(psiEtaZeta, h)
            totalJac = self._calcPartialPtsJacobian(psiEtaZeta, h)
            for _ in range(nPts):
                ptJac = ptsJac[3*_:3*(_+1),:]
                for __ in range(nCoeffs):
                    paramJac = paramsJac[3*_:3*(_+1), 3*__:3*(__+1)]
                    dXYZdCoeffVec = ptJac @ np.diagonal(paramJac)
                    #Add to partial derivatives
                    totalJac[3*_:3*(_+1), 3*__:3*(__+1)] += np.diag(dXYZdCoeffVec)
        else:
            totalJac = None
        return totalJac

    #dPsiEtaZetadCsClassCoeffs (Analytical)
    def _calcCsClassJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csClassCoeffs)
        if nCoeffs>0:
            n1,n2 = self.csClassCoeffs
            psiVals = psiEtaZeta[:,0]
            csClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            csClassJac[2::3, 2] = n1*np.power(psiVals,n1-1)*np.power(1-psiVals,n2)*self.shapeFunc(psiEtaZeta,self.shapeCoeffs)
            csClassJac[2::3, 5] = -n2*np.power(psiVals,n1)*np.power(1-psiVals,n2-1)*self.shapeFunc(psiEtaZeta,self.shapeCoeffs)
        else:
            csClassJac = None
        return csClassJac

    #dPsiEtaZetadCsModCoeffs (Analytical)
    def _calcCsModJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csModCoeffs)
        if nCoeffs>0:
            csModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
        else:
            csModJac = None
        return csModJac

    #dPsiEtaZetadSpanModCoeffs (Analytical)
    def _calcSpanModJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.spanModCoeffs)
        if nCoeffs>0:
            spanModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            spanModJac[2::3, 2] = psiEtaZeta[:,1] #Shear, 1 coeff linear dist
            spanModJac[2::3, 5] = -psiEtaZeta[:,0]*psiEtaZeta[:,1]*np.power(np.cos(self.spanModCoeffs[1]*psiEtaZeta[:,1]),-2.0) #Twist, 1 coeff linear dist
        else:
            spanModJac = None
        return spanModJac

    #dPsiEtaZetadShapeCoeffs (Analytical)
    def _calcShapeJacobian(self, psiEtaZeta, h=1e-8):
        nx,ny = self.order
        nCoeffs = len(self.shapeCoeffs)
        if nCoeffs>0:
            shapeJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeJac[2::3, 2::3] = bernstein2DJacobian(psiEtaZeta[:,0], psiEtaZeta[:,1], nx, ny, h)
            for _ in range(nCoeffs):
                shapeJac[2::3,3*_+2] = shapeJac[2::3,3*_+2] * self.csClassFunc(psiEtaZeta,self.csClassCoeffs)
        else:
            shapeJac = None
        return shapeJac

    #dPsiEtaZetadRefChordCoeffs (Analytical)
    def _calcRefChordJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.chordCoeffs)
        if nCoeffs>0:
            refChordJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
        else:
            refChordJac = None
        return refChordJac

    #dPsiEtaZetadShapeOffsetCoeffs (Analytical)
    def _calcShapeOffsetJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.shapeOffsets)
        if nCoeffs>0:
            shapeOffsetJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeOffsetJac[2::3, 2] = psiEtaZeta[:,0]
        else:
            shapeOffsetJac = None
        return shapeOffsetJac

    #Calc jacobian transformation matrix
    #                              [u]                               [x]
    # If reference coordinates are [v], and internal coordinates are [y]
    #                              [w]                               [z]
    #     [dU/dCoeff]   [dU/dX dU/dY dU/dZ] [dX/dCoeff]
    # i.e [dV/dCoeff] = [dV/dX dV/dY dV/dZ]*[dY/dCoeff]
    #     [dW/dCoeff]   [dW/dX dW/dY dW/dZ] [dZ/dCoeff]
    def _calcTransformJacobian(self, surface, h=1e-8):
        transformJac = np.zeros((len(surface.flatten()),3))
        for _ in range(len(surface)):
            #For a linear transformation, this is just the transpose of the reference axes
            transformJac[3*_:3*(_+1),:] = self.refAxes.T
        return transformJac

    #dXYZdPsiEtaZeta (analytical)
    def _calcPtsJacobian(self, psiEtaZeta, h=1e-8):
        ptsJac = np.zeros((len(psiEtaZeta.flatten()), 3))
        ptsJac[0::3, 0] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[0::3, 1] = psiEtaZeta[:,0]*(self.chordCoeffs[1]-self.chordCoeffs[0]) + self.spanModCoeffs[0]
        ptsJac[1::3, 1] = self.refSpan
        ptsJac[2::3, 1] = psiEtaZeta[:,2]*(self.chordCoeffs[1]-self.chordCoeffs[0])
        ptsJac[2::3, 2] = self.refLen(psiEtaZeta, self.chordCoeffs)
        #Pre-multiply by transformation from coordinate axes
        surface = self.calcCoords(psiEtaZeta)
        transformJac = self._calcTransformJacobian(surface, h)
        for _ in range(len(psiEtaZeta)):
            ptsJac[3*_:3*(_+1),:] = transformJac[3*_:3*(_+1),:] @ ptsJac[3*_:3*(_+1),:]
        return ptsJac

    #partialXYZpartialCoeffs
    def _calcPartialPtsJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0:
            partialPtsJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            n1 = 3*len(self.csClassCoeffs)
            n2 = n1+3*len(self.csModCoeffs)
            partialPtsJac[::3,n1] = psiEtaZeta[:,1]
            n3 = n2+3*len(self.spanClassCoeffs)
            n4 = n3+3*len(self.spanModCoeffs)
            n5 = n4+3*len(self.shapeCoeffs)
            n6 = n5+3*len(self.chordCoeffs)
            partialPtsJac[::3,n5] = psiEtaZeta[:,0]*(1-psiEtaZeta[:,1])
            partialPtsJac[::3,n5+3] = psiEtaZeta[:,0]*psiEtaZeta[:,1]
            partialPtsJac[2::3,n5] = psiEtaZeta[:,2]*(1-psiEtaZeta[:,1])
            partialPtsJac[2::3,n5+3] = psiEtaZeta[:,2]*psiEtaZeta[:,1]
        else:
            partialPtsJac = 0
        return partialPtsJac

#################################################################################################

class CSTRevolve3D(CST3DParam):
    """
    Class for storing cst parameterization information for bodies of revolution
    """
    def __init__(self, surface, csClassFunc=None, csModFunc=None, spanModFunc=None, refLenFunc=None,
                 csClassCoeffs=[], csModCoeffs=[0.0], shearCoeffs=[0.0], twistCoeffs=[0.0], shapeCoeffs=[], chordCoeffs=[1.0],
                 shapeOffsets=[0.0, 0.0], masks=[], order=[5,2], refSpan=2*np.pi, origin=np.zeros(3), refAxes=np.eye(3), shapeScale=1.0):

        csClassFunc = self.defaultClassFunction if csClassFunc is None else csClassFunc
        csModFunc = self.defaultCsModFunction if csModFunc is None else csModFunc
        self.shapeFunc = self.defaultShapeFunction
        spanModFunc = self.defaultSpanModFunc if spanModFunc is None else spanModFunc
        spanModCoeffs = shearCoeffs+twistCoeffs
        self.nSpanModCoeffs = [len(shearCoeffs), len(twistCoeffs)]

        self.csGeo = CSTAirfoil2D(np.zeros((1,2)), classFunc=csClassFunc, classCoeffs=csClassCoeffs,
                                  shapeCoeffs=[], masks=[], order=order[0], shapeOffset=shapeOffsets[0],
                                  refLen=chordCoeffs[0], shapeScale=shapeScale)

        super().__init__(surface=surface, csClassFunc=csClassFunc, csModFunc=csModFunc, spanClassFunc=None, spanModFunc=spanModFunc, refLenFunc=refLenFunc,
                         csClassCoeffs=csClassCoeffs, csModCoeffs=csModCoeffs, spanClassCoeffs=[], spanModCoeffs=spanModCoeffs, shapeCoeffs=shapeCoeffs,
                         chordCoeffs=chordCoeffs, shapeOffsets=shapeOffsets, masks=masks, order=order, refSpan=refSpan, origin=origin, refAxes=refAxes,
                         shapeScale=shapeScale)

        cylindrical = self.transformSurface(surface)
        self.csGeo = CSTAirfoil2D(cylindrical[:,(2,0)], classFunc=csClassFunc, classCoeffs=csClassCoeffs,
                                  shapeCoeffs=[], masks=[], order=order[0], shapeOffset=shapeOffsets[0],
                                  refLen=chordCoeffs[0], shapeScale=shapeScale)
        

    #Airfoil Class function
    def defaultClassFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return self.csGeo.airfoilClassFunc(psiEtaZeta[:,0], coeffs)

    #2D shape function
    def defaultShapeFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0] #List of coefficients #ncoeff = (order0+1)*(order1+1)
        augments = bernstein2D(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1]) @ coeffs
        return augments

    #LE modification - default constant
    def defaultCsModFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]*np.ones(len(psiEtaZeta))

    #Shear and twist, default constants
    def defaultSpanModFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        shearCoeffs = coeffs[:self.nSpanModCoeffs[0]]
        twistCoeffs = coeffs[self.nSpanModCoeffs[0]:]
        return self._shearFunc(psiEtaZeta, shearCoeffs) - psiEtaZeta[:,0]*np.tan(self._twistFunc(psiEtaZeta, twistCoeffs))

    #Returns shear amount - default constant
    def _shearFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]*np.ones(len(psiEtaZeta))

    #Returns angle of twist (radians) - default constant
    def _twistFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]*np.ones(len(psiEtaZeta))

    #Function to define chord length along eta, default constant
    def defaultChordFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return super().defaultChordFunction(psiEtaZeta, coeffs)

    #Calculate psi vals from rThetaZ vals
    def calcXYZ2Psi(self, xyz):
        xyz2 = np.copy(xyz)
        psiEtaZeta = np.vstack([np.zeros(len(xyz)), self.calcXYZ2Eta(xyz), np.zeros(len(xyz))]).T
        xyz2[:,2] -= self.csModFunc(psiEtaZeta, self.csModCoeffs)
        return xyz2[:,2]/self.refLen(psiEtaZeta, self.chordCoeffs)

    #Calculate zeta vals from rThetaZ vals
    def calcXYZ2Zeta(self, xyz):
        etaVals = self.calcXYZ2Eta(xyz)
        psiEtaZeta = np.vstack([np.zeros_like(etaVals), etaVals, np.zeros_like(etaVals)]).T
        return xyz[:,0]/self.refLen(psiEtaZeta, self.chordCoeffs)

    #calculate r vals from psiEtaZeta vals
    def calcPsiEtaZeta2X(self, psiEtaZeta):
        return psiEtaZeta[:,2]*self.refLen(psiEtaZeta, self.chordCoeffs)

    #Calculate z vals from psiEtaZeta vals
    def calcPsiEtaZeta2Z(self, psiEtaZeta):
        return psiEtaZeta[:,0]*self.refLen(psiEtaZeta, self.chordCoeffs) + self.csModFunc(psiEtaZeta, self.csModCoeffs)

    #Explicitly calculate zeta values
    def calcZeta(self, psiEtaZeta):
        zetaVals = (self.shapeOffsets[1] +
                    psiEtaZeta[:,0]*self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                    self.spanModFunc(psiEtaZeta, self.spanModCoeffs) +
                    self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                    self.shapeFunc(psiEtaZeta, self.shapeCoeffs))
        return zetaVals

    #TODO improve efficiency of transforms
    #RThetaZ->uvw = xyz2uvw @ rtz2xyz
    def _calcTransformMatrix(self, transSurface):
        transMats = np.zeros((len(transSurface.flatten()), 3))
        for _ in range(len(transSurface)):
            theta = transSurface[_,1]
            defaultAxes = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]])
            rtz2xyz = defaultAxes @ np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                              [np.sin(theta), np.cos(theta), 0.0],
                                              [0.0, 0.0, 1.0]])
            xyz2uvw = self.refAxes.T
            transMats[3*_:3*(_+1),:] = xyz2uvw @ rtz2xyz
        return transMats

    #uvw->RThetaZ
    def _calcInverseTransform(self, surface):
        nPts = len(surface)
        transMats = np.zeros((3*nPts, 3))
        thetas = np.zeros(nPts)
        for _ in range(nPts):
            defaultAxes = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]])
            uvw2xyz = np.linalg.inv(self.refAxes.T)
            xyz = uvw2xyz @ surface[_]
            if xyz[0]==0:
                theta = 0.5*np.pi
            else:
                theta = np.arctan(xyz[2]/xyz[0])
            thetas[_] = theta
            xyz2rtz = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                [-np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]]) @ defaultAxes
            transMats[3*_:3*(_+1),:] = xyz2rtz @ uvw2xyz
        return transMats, thetas

    #TODO Make transforms more efficient
    #Transform surface coordinates into axes used by class functions
    #uvw->RThetaZ, we want theta to store angle in radians
    def transformSurface(self, surface):
        numPts = len(surface)
        transSurface = np.zeros_like(surface)
        transMats, thetas = self._calcInverseTransform(surface)
        for _ in range(numPts):
            transMat = transMats[3*_:3*(_+1),:]
            transSurface[_,:] = transMat @ (surface[_,:] - self.origin)
        transSurface[:,1] = thetas
        return transSurface

    #Untransform coordinates into given reference axes
    #RThetaZ->uvw, we want remove theta from multiplication
    def untransformSurface(self, transSurface):
        numPts = len(transSurface)
        surface = np.zeros_like(transSurface)
        transMats = self._calcTransformMatrix(transSurface)
        transSurface[:,1] = 0 #set thetas to zero after determining transformation matrices
        for _ in range(numPts):
            transMat = transMats[3*_:3*(_+1),:]
            surface[_,:] = (transMat @ transSurface[_,:]) + self.origin
        return surface

    #Calc theta values from r and z values
    def calcXZ2Y(self, xVals, zVals):
        psiEtaZeta = np.zeros((len(xVals),3))
        def objFunc(yVals):
            xyz = np.vstack([xVals, yVals, zVals]).T
            psiEtaZeta[:,0] = self.calcXYZ2Psi(xyz)
            psiEtaZeta[:,1] = self.calcXYZ2Eta(xyz)
            psiEtaZeta[:,2] = self.calcXYZ2Zeta(xyz)
            return self.calcZeta(psiEtaZeta)-psiEtaZeta[:,2]
        yVals = scp.fsolve(objFunc, 0 if isScalar(xVals) else np.zeros(len(xVals)))
        return yVals

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.csGeo.shapeCoeffs + [self.shapeOffsets[0]]
        self.csGeo.updateCoeffs(csCoeffs)
        return

    def _calcCsClassGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.csGeo._calcClassDeriv(psiEtaZeta[:,0], h), np.zeros(len(psiEtaZeta))]).T

    def _calcShapeGrad(self, psiEtaZeta, h=1e-8):
        dShapedPsi, dShapedEta = bernstein2DGrad(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1], h)
        dShapedPsi = dShapedPsi @ self.shapeCoeffs
        dShapedEta = dShapedEta @ self.shapeCoeffs
        return np.vstack([dShapedPsi, dShapedEta]).T

    def _calcOffsetGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.offsetFunc(psiEtaZeta, self.shapeOffsets),np.zeros(len(psiEtaZeta))]).T

    def _calcShearGrad(self, psiEtaZeta, h=1e-8):
        shearCoeffs = self.spanModCoeffs[:self.nSpanModCoeffs[0]]
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), np.zeros(nPts)]).T

    def _calcTwistGrad(self, psiEtaZeta, h=1e-8):
        twistCoeffs = self.spanModCoeffs[self.nSpanModCoeffs[0]:]
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), np.zeros(nPts)]).T
            
    #Gradient of span modification terms
    def _calcSpanModGrad(self, psiEtaZeta, h=1e-8):
        #twistCoeffs = self.spanModCoeffs[self.nSpanModCoeffs[0]:]
        #dSpanModdPsi = -np.tan(self._twistFunc(psiEtaZeta, twistCoeffs))
        #dSpanModdEta = (self._calcShearGrad(psiEtaZeta, h)[:,1] - 
        #                psiEtaZeta[:,0]*self._calcTwistGrad(psiEtaZeta, h)[:,1]*
        #                np.power(np.cos(self._twistFunc(psiEtaZeta, twistCoeffs)),-2.0))
        #return np.vstack([dSpanModdPsi, dSpanModdEta]).T
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), np.zeros(nPts)]).T

    #Analytical gradient of zeta
    def calcParamsGrad(self, psiEtaZeta, h=1e-8):
        dZetadPsi = (self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                     self._calcSpanModGrad(psiEtaZeta, h)[:,0] +
                     self._calcCsClassGrad(psiEtaZeta, h)[:,0] * 
                     self.shapeFunc(psiEtaZeta, self.shapeCoeffs) +
                     self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                     self._calcShapeGrad(psiEtaZeta, h)[:,0])
        dZetadEta = (psiEtaZeta[:,0]*self._calcOffsetGrad(psiEtaZeta, h)[:,1] +
                     self._calcSpanModGrad(psiEtaZeta, h)[:,1] + 
                     self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                     self._calcShapeGrad(psiEtaZeta, h)[:,1])
        return np.vstack([dZetadPsi, dZetadEta]).T

    #Calculate gradient of R(T,Z)
    def calcGrad(self, surface, h=1e-8):
        psiEtaZeta = self.coords2PsiEtaZeta(surface)
        paramsGrad = self.calcParamsGrad(psiEtaZeta, h)
        dRdT = self.refLen(psiEtaZeta, self.chordCoeffs) * paramsGrad[:,1] / self.refSpan #dRdZeta * dZetadEta * dEtadT
        dRdZ = paramsGrad[:,0] #dRdZeta * dZetadPsi * dPsidZ
        return np.vstack([dRdT, dRdZ]).T

    #TODO Make this more efficient
    #Calculate dXYZdCoeffs Jacobian (Analytical)
    #dXYZdCoeff = dXYZdPsiEtaZeta * dPsiEtaZetadCoeff + pXYZpCoeff
    def calcJacobian(self, surface, h=1e-8):
        nPts = len(surface)
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0 and nPts>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            ptsJac = self._calcPtsJacobian(psiEtaZeta, h)
            paramsJac = self.calcParamsJacobian(psiEtaZeta, h)
            totalJac = self._calcPartialPtsJacobian(psiEtaZeta, h)
            for _ in range(nPts):
                ptJac = ptsJac[3*_:3*(_+1),:]
                for __ in range(nCoeffs):
                    paramJac = paramsJac[3*_:3*(_+1), 3*__:3*(__+1)]
                    dXYZdCoeffVec = ptJac @ np.diagonal(paramJac)
                    #Add to partial derivatives
                    totalJac[3*_:3*(_+1), 3*__:3*(__+1)] += np.diag(dXYZdCoeffVec)
        else:
            totalJac = None
        return totalJac

    #dPsiEtaZetadCsClassCoeffs (Analytical)
    def _calcCsClassJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csClassCoeffs)
        if nCoeffs>0:
            n1,n2 = self.csClassCoeffs
            psiVals = psiEtaZeta[:,0]
            csClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            csClassJac[2::3, 2] = n1*np.power(psiVals,n1-1)*np.power(1-psiVals,n2)*self.shapeFunc(psiEtaZeta,self.shapeCoeffs)
            csClassJac[2::3, 5] = -n2*np.power(psiVals,n1)*np.power(1-psiVals,n2-1)*self.shapeFunc(psiEtaZeta,self.shapeCoeffs)
        else:
            csClassJac = None
        return csClassJac

    #dPsiEtaZetadCsModCoeffs (Analytical)
    def _calcCsModJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csModCoeffs)
        if nCoeffs>0:
            csModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
        else:
            csModJac = None
        return csModJac

    #dPsiEtaZetadSpanModCoeffs (Analytical)
    def _calcSpanModJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.spanModCoeffs)
        if nCoeffs>0:
            spanModJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            spanModJac[2::3, 5] = -psiEtaZeta[:,0]*np.power(np.cos(self.spanModCoeffs[1]),-2.0) #Twist, constant on eta
        else:
            spanModJac = None
        return spanModJac

    #dPsiEtaZetadShapeCoeffs (Analytical)
    def _calcShapeJacobian(self, psiEtaZeta, h=1e-8):
        nx,ny = self.order
        nCoeffs = len(self.shapeCoeffs)
        if nCoeffs>0:
            shapeJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeJac[2::3, 2::3] = bernstein2DJacobian(psiEtaZeta[:,0], psiEtaZeta[:,1], nx, ny, h)
            for _ in range(nCoeffs):
                shapeJac[2::3,3*_+2] = shapeJac[2::3,3*_+2] * self.csClassFunc(psiEtaZeta,self.csClassCoeffs)
        else:
            shapeJac = None
        return shapeJac

    #dPsiEtaZetadRefChordCoeffs (Analytical)
    def _calcRefChordJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.chordCoeffs)
        if nCoeffs>0:
            refChordJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
        else:
            refChordJac = None
        return refChordJac

    #dPsiEtaZetadShapeOffsetCoeffs (Analytical)
    def _calcShapeOffsetJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.shapeOffsets)
        if nCoeffs>0:
            shapeOffsetJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            shapeOffsetJac[2::3, 2] = psiEtaZeta[:,0]
            shapeOffsetJac[2::3, 5] = np.ones(len(psiEtaZeta))
        else:
            shapeOffsetJac = None
        return shapeOffsetJac

    #Calc jacobian transformation matrix
    #                              [U]                               [R]
    # If reference coordinates are [V], and internal coordinates are [T]
    #                              [W]                               [Z]
    #     [dU/dCoeff]   [dU/dX dU/dY dU/dZ]   [dX/dR dX/dT dX/dZ]   [dR/dCoeff]
    # i.e [dV/dCoeff] = [dV/dX dV/dY dV/dZ] @ [dY/dR dY/dT dY/dZ] @ [dT/dCoeff]
    #     [dW/dCoeff]   [dW/dX dW/dY dW/dZ]   [dZ/dR dZ/dT dZ/dZ]   [dZ/dCoeff]
    def _calcTransformJacobian(self, surface, h=1e-8):
        transformJac = np.zeros((len(surface.flatten()),3))
        transSurface = self.transformSurface(surface)
        for _ in range(len(surface)):
            r = transSurface[_,0]
            theta = transSurface[_,1]
            defaultAxes = np.array([[0, 0, 1],  #row swap to ZYX, since zeta becomes r axis
                                    [0, 1, 0],
                                    [1, 0, 0]])
            dXYZdRTZ = defaultAxes @ np.array([[np.cos(theta), -r*np.sin(theta), 0.0],
                                               [np.sin(theta), r*np.cos(theta), 0.0],
                                               [0.0, 0.0, 1.0]])
            dUVWdXYZ = self.refAxes.T
            transformJac[3*_:3*(_+1),:] = dUVWdXYZ @ dXYZdRTZ
        return transformJac

    #dRThetaZdPsiEtaZeta (analytical)
    def _calcPtsJacobian(self, psiEtaZeta, h=1e-8):
        ptsJac = np.zeros((len(psiEtaZeta.flatten()), 3))
        ptsJac[0::3, 1] = psiEtaZeta[:,2] * 0 # dRefLen/dEta = 0
        ptsJac[0::3, 2] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[1::3, 1] = self.refSpan # convert eta to radians
        ptsJac[2::3, 0] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[2::3, 1] = psiEtaZeta[:,0] * 0 + 0 # dRefLen/dEta = 0, dCsModFunc/dEta = 0
        #Pre-multiply by transformation from coordinate axes
        surface = self.calcCoords(psiEtaZeta)
        transformJac = self._calcTransformJacobian(surface, h)
        for _ in range(len(psiEtaZeta)):
            ptsJac[3*_:3*(_+1),:] = transformJac[3*_:3*(_+1),:] @ ptsJac[3*_:3*(_+1),:]
        return ptsJac

    #partialXYZpartialCoeffs
    def _calcPartialPtsJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0:
            #Default all zeros from chord/mod function implementations
            partialPtsJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            n1 = 3*len(self.csClassCoeffs)
            n2 = n1+3*len(self.csModCoeffs)
            n3 = n2+3*len(self.spanClassCoeffs)
            n4 = n3+3*len(self.spanModCoeffs)
            n5 = n4+3*len(self.shapeCoeffs)
            n6 = n5+3*len(self.chordCoeffs)
        else:
            partialPtsJac = 0
        return partialPtsJac

#################################################################################################