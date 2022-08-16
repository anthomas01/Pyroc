import scipy.optimize as scp
import numpy as np
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
        By default, 1.0

    origin : list of float [1,3]
        Origin for geometry object
        By default, [0.0,0.0,0.0]

    TODO refAxes : list of float [2,3]
        2D reference axes - First row is psi axis, second is zeta axis
        By default, [[1.0,0.0,0.0],
                     [0.0,0.0,1.0]]

    shapeScale: float
        If shapeCoeffs is not given, this value will be used for initial shape coeffs.
        shapeScale=-1.0 can be used for a -z surface initialization
        By default, 1.0

    TODO Rotation/Extrusion/Modification definition modes encompassed - Different coordinate systems
         Multiple sections (Piecewise functions) - Higher level of abstraction
         Axes variation?
    """

    def __init__(self, surface, csClassFunc=None, csModFunc=None, spanClassFunc=None, spanModFunc=None, refLenFunc=None,
                 csClassCoeffs=[], csModCoeffs=[], spanClassCoeffs=[], spanModCoeffs=[], shapeCoeffs=[], chordCoeffs=[1.0], 
                 shapeOffsets=[0.0], masks=[], order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], 
                 refAxes=[[1.0,0.0,0.0],[0.0,0.0,1.0]], shapeScale=1.0):
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
        self.origin = np.array(origin)
        #Reference Axis - extrusion direction/rotation axis
        self.refAxes = np.array(refAxes)

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
        self.masks = [0 for _ in range(len(self.getCoeffs()))] if len(masks)==0 else masks.copy()

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
        return coeffs[0]

    #Calculate psi vals from x and y vals
    def calcXY2Psi(self, xVals, yVals):
        etaVals = self.calcY2Eta(yVals)
        psiEtaZeta = np.vstack([np.zeros(len(etaVals)), etaVals, np.zeros(len(etaVals))]).T
        return xVals/self.refLen(psiEtaZeta, self.chordCoeffs)

    #Calculate eta vals from y vals
    def calcY2Eta(self, yVals):
        return yVals/self.refSpan

    #Calculate zeta vals from y and z vals
    def calcYZ2Zeta(self, yVals, zVals):
        etaVals = self.calcY2Eta(yVals)
        psiEtaZeta = np.vstack([np.zeros(len(etaVals)), etaVals, np.zeros(len(etaVals))]).T
        return zVals/self.refLen(psiEtaZeta, self.chordCoeffs)

    #calculate x vals from psi and zeta vals
    def calcPsiEta2X(self, psiEtaZeta):
        return psiEtaZeta[:,0]*self.refLen(psiEtaZeta,self.chordCoeffs)

    #Calculate y vals from eta vals
    def calcEta2Y(self, psiEtaZeta):
        return psiEtaZeta[:,1]*self.refSpan

    #Calculate z vals from eta and zeta vals
    def calcEtaZeta2Z(self, psiEtaZeta):
        return psiEtaZeta[:,2]*self.refLen(psiEtaZeta,self.chordCoeffs)
        
    #calculate psi values from eta, zeta
    def calcPsi(self, psiEtaZeta):
        pass

    #calculate eta values form psi, zeta
    def calcEta(self, psiEtaZeta):
        pass
    
    #Explicitly calculate zeta values from psi, eta
    ##Different for each mode?
    def calcZeta(self, psiEtaZeta):
        pass

    #Update zeta values from internal psi and eta values
    def updateZeta(self):
        self.psiEtaZeta[:,2] = self.calcZeta(self.psiEtaZeta)
        return self.psiEtaZeta

    def transformSurface(self, surface):
        transSurface = surface - self.origin
        return transSurface

    def untransformSurface(self, transSurface):
        surface = transSurface + self.origin
        return surface

    #Calculate cartesian of any parametric set and return
    def calcCoords(self, psiEtaZeta):
        transSurface = np.vstack([self.calcPsiEta2X(psiEtaZeta[:,0], psiEtaZeta[:,1]),
                                  self.calcEta2Y(psiEtaZeta[:,1]),
                                  self.calcEtaZeta2Z(psiEtaZeta[:,1], psiEtaZeta[:,2])]).T
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
        psiVals = self.calcXY2Psi(transSurface[:,0], transSurface[:,1])
        etaVals = self.calcY2Eta(transSurface[:,1])
        psiEtaZeta =  np.vstack([psiVals, etaVals, np.zeros_like(psiVals)]).T
        psiEtaZeta[:,2] = self.calcZeta(psiEtaZeta)
        return psiEtaZeta

    #Set psi/eta values and update zeta
    def setPsiEtaZeta(self, psiEtaZeta):
        self.psiEtaZeta = np.vstack([psiEtaZeta[:,0], psiEtaZeta[:,1], self.calcZeta(psiEtaZeta)]).T
        return self.psiEtaZeta

    #Calculate x vals from y and z vals
    #Will have multiple solutions?
    def calcYZ2X(self, yVals, zVals):
        etaVals = self.calcY2Eta(yVals)
        psiEtaZeta = np.vstack([np.zeros(len(etaVals)), etaVals, self.calcYZ2Zeta(yVals,zVals)]).T
        psiEtaZeta[:,0] = self.calcPsi(psiEtaZeta)
        xVals = self.calcPsiEta2X(psiEtaZeta)
        return xVals

    #Calculate y vals from x and z vals
    #Will have multiple solutions? Pointless?
    def calcXZ2Y(self, xVals, zVals):
        pass

    #Calculate z vals from x and y vals
    def calcXY2Z(self, xVals, yVals):
        etaVals = self.calcY2Eta(yVals)
        psiEtaZeta = np.vstack([self.calcXY2Psi(xVals,yVals), etaVals, np.zeros_like(etaVals)]).T
        psiEtaZeta[:,2] = self.calcZeta(psiEtaZeta)
        zVals = self.calcEtaZeta2Z(psiEtaZeta)
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
    def calcGrad(self, psiEtaZeta, h=1e-8):
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

    #Get gradient of pts
    def getGrad(self):
        return self.calcGrad(self.psiEtaZeta)

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

    #Calculate dXYZdCoeffs Jacobian (FD)
    #dXYZdCoeff = dXYZdPsiEtaZeta * dPsiEtaZetadCoeff
    def calcJacobian(self, surface, h=1e-8):
        coeffs = self.getCoeffs()
        nCoeffs = len(coeffs)
        if nCoeffs>0 and len(surface)>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            surfaceJac = np.zeros((len(surface.flatten()), 3*nCoeffs))
            for _ in range(nCoeffs):
                coeffs[_] += h
                self.updateCoeffs(coeffs)
                surfaceH = self.calcCoords(psiEtaZeta)
                for __ in range(3):
                    surfaceJac[__::3,3*_+__] = (surfaceH[:,__]-surface[:,__])/h
                coeffs[_] -= h
                self.updateCoeffs(coeffs)
        else:
            raise Exception('No Coefficients!')
        return surfaceJac

    #Get dXYZdCoeffs Jacobian
    def getJacobian(self):
        return self.calcJacobian(self.surface)

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


class CSTAirfoil3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded 'airfoil' surface (Pseudo 2D)
    """
    def __init__(self, surface, csClassFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], chordCoeffs=[1.0], shapeOffsets=[0.0], masks=[],
                 order=[5,0], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]]), shapeScale=1.0):
        super().__init__(surface=surface, csClassFunc=csClassFunc, spanClassFunc=None, refLenFunc=None,
                         csClassCoeffs=csClassCoeffs, csModCoeffs=[], spanClassCoeffs=[], spanModCoeffs=[], 
                         shapeCoeffs=shapeCoeffs, chordCoeffs=chordCoeffs, shapeOffsets=shapeOffsets, masks=masks,
                         order=order, refSpan=refSpan, origin=origin, refAxes=refAxes, shapeScale=shapeScale)
        
        self.csGeo = CSTAirfoil2D(surface[:,(0,2)], classFunc=self.csClassFunc, 
                                  classCoeffs=self.csClassCoeffs, shapeCoeffs=self.shapeCoeffs, masks=self.masks,
                                  order=order[0], shapeOffset=shapeOffsets[0], shapeScale=shapeScale)

    def calcPsi(self, psiEtaZeta):
        return self.csGeo.calcPsi(psiEtaZeta[:,2])

    #Calc eta cannot and does not need to be implemented, no dependency

    def calcZeta(self, psiEtaZeta):
        return self.csGeo.calcZeta(psiEtaZeta[:,0])

    #Calc xz2y cannot and does not need to be implemented, no dependency

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.shapeCoeffs + [self.shapeOffsets[0]]
        self.csGeo.updateCoeffs(csCoeffs)
        return

    #Analytical gradient of zeta
    def calcGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.csGeo.calcDeriv(psiEtaZeta[:,0], h), np.zeros(len(psiEtaZeta))]).T

    #TODO make this more efficient
    #Calculate dXYZdCoeffs Jacobian (Analytical)
    #dXYZdCoeff = dXYZdPsiEtaZeta * dPsiEtaZetadCoeff
    def calcJacobian(self, surface, h=1e-8):
        nPts = len(surface)
        nCoeffs = len(self.getCoeffs())
        if nCoeffs>0 and nPts>0:
            psiEtaZeta = self.coords2PsiEtaZeta(surface)
            ptsJac = self._calcPtsJacobian(psiEtaZeta, h)
            paramsJac = self.calcParamsJacobian(psiEtaZeta, h)
            totalJac = np.zeros((3*nPts, 3*nCoeffs))
            for _ in range(nPts):
                ptJac = ptsJac[3*_:3*(_+1),:]
                for __ in range(nCoeffs):
                    paramJac = paramsJac[3*_:3*(_+1), 3*__:3*(__+1)]
                    dPsiZetaEtadCoeffVec = np.array([np.diagonal(paramJac)]).T
                    dXYZdCoeffVec = np.dot(ptJac, dPsiZetaEtadCoeffVec)
                    totalJac[3*_:3*(_+1), 3*__:3*(__+1)] = np.diag(dXYZdCoeffVec.flatten())
        else:
            totalJac = None
        return totalJac

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

    #dXYZdPsiEtaZeta (analytical)
    def _calcPtsJacobian(self, psiEtaZeta, h=1e-8):
        ptsJac = np.zeros((len(psiEtaZeta.flatten()), 3))
        ptsJac[0::3, 0] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[1::3, 1] = self.refSpan
        ptsJac[2::3, 2] = self.refLen(psiEtaZeta, self.chordCoeffs)
        return ptsJac


class CSTWing3D(CST3DParam):
    """
    Class for storing cst parameterization information for wing section
    """
    def __init__(self, surface, csClassFunc=None, csModFunc=None, spanModFunc=None, refLenFunc=None,
                 csClassCoeffs=[0.5,1.0], shapeCoeffs=[], sweepCoeffs=[], shearCoeffs=[], twistCoeffs=[], chordCoeffs=[1.0, 1.0], shapeOffsets=[0.0], masks=[],
                 order=[5,2], refSpan=1.0, origin=[0.0,0.0,0.0], refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), shapeScale=1.0):

        csClassFunc = self.defaultClassFunction if csClassFunc is None else csClassFunc
        csModFunc = self.defaultCsModFunction if csModFunc is None else csModFunc
        csModCoeffs = sweepCoeffs
        self.shapeFunc = self.defaultShapeFunction

        spanModFunc = self.defaultSpanModFunc if spanModFunc is None else spanModFunc
        spanModCoeffs = shearCoeffs+twistCoeffs
        self.nSpanModCoeffs = [len(shearCoeffs), len(twistCoeffs)]

        super().__init__(surface=surface, csClassFunc=csClassFunc, csModFunc=csModFunc, spanClassFunc=None, spanModFunc=spanModFunc, 
                         refLenFunc=refLenFunc, csClassCoeffs=csClassCoeffs, csModCoeffs=csModCoeffs, spanClassCoeffs=[],
                         spanModCoeffs=spanModCoeffs, shapeCoeffs=shapeCoeffs, chordCoeffs=chordCoeffs, shapeOffsets=shapeOffsets,
                         masks=masks, order=order, refSpan=refSpan, origin=origin, refAxes=refAxes, shapeScale=shapeScale)

        self.csGeo = CSTAirfoil2D(surface[:,(0,2)], classFunc=self.csClassFunc, 
                                  classCoeffs=self.csClassCoeffs, shapeCoeffs=[], masks=[],
                                  order=order[0], shapeOffset=shapeOffsets[0], shapeScale=shapeScale)

    #Airfoil Class function
    def defaultClassFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return self.csGeo.airfoilClassFunc(psiEtaZeta[:,0], coeffs)

    #2D shape function
    def defaultShapeFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0] #List of coefficients #ncoeff = (order0+1)*(order1+1)
        augments = np.dot(bernstein2D(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1]), np.array([coeffs]).T)
        return augments.flatten()

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

    #Function to define chord length along eta, default linear dist [rootChord, tipChord]
    def defaultChordFunction(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]-psiEtaZeta[:,1]*(coeffs[0]-coeffs[1])

    def calcXY2Psi(self, xVals, yVals):
        xVals = xVals - self.csModFunc(self.calcY2Eta(yVals), self.csModCoeffs)
        return super().calcXY2Psi(xVals, yVals)

    def calcPsiEta2X(self, psiEtaZeta):
        return super().calcPsiEta2X(psiEtaZeta) + self.csModFunc(psiEtaZeta, self.csModCoeffs)

    #Estimates psi vals from zeta values
    ##TODO Broken, needs to have better initial guess
    def calcPsi(self, psiEtaZeta):
        def objFunc(psiVals):
            return self.calcZeta(psiVals, psiEtaZeta[:,1])-psiEtaZeta[:,2]
        psiVals = scp.fsolve(objFunc, 0 if isScalar(psiEtaZeta[:,2]) else np.zeros(len(psiEtaZeta[:,2])))
        return psiVals

    def calcEta(self, psiEtaZeta):
        pass

    #Explicitly calculate zeta values
    def calcZeta(self, psiEtaZeta):
        zetaVals = (psiEtaZeta[:,0]*self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                    self.spanModFunc(psiEtaZeta, self.spanModCoeffs) +
                    self.csClassFunc(psiEtaZeta, self.csClassCoeffs) * 
                    self.shapeFunc(psiEtaZeta, self.shapeCoeffs))
        return zetaVals

    def calcXZ2Y(self, xVals, zVals):
        pass

    def updateCoeffs(self, *coeffs):
        coeffs = coeffs[0]
        super().updateCoeffs(coeffs)
        csCoeffs = self.csClassCoeffs + self.csGeo.shapeCoeffs + [self.shapeOffsets[0]]
        self.csGeo.updateCoeffs(csCoeffs)
        return

    def _calcShapeGrad(self, psiEtaZeta, h=1e-8):
        dShapedPsi, dShapedEta = bernstein2DGrad(psiEtaZeta[:,0], psiEtaZeta[:,1], self.order[0], self.order[1], h)
        dShapedPsi = np.dot(dShapedPsi, np.array([self.shapeCoeffs]).T)
        dShapedEta = np.dot(dShapedEta, np.array([self.shapeCoeffs]).T)
        return np.vstack([dShapedPsi.flatten(), dShapedEta.flatten()]).T

    def _calcClassGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([self.csGeo.calcDeriv(psiEtaZeta[:,0], h), np.zeros(len(psiEtaZeta))]).T

    def _calcOffsetGrad(self, psiEtaZeta, h=1e-8):
        return np.vstack([np.zeros(len(psiEtaZeta)),np.zeros(len(psiEtaZeta))]).T

    #Returns shear amount - default linear dist along eta
    def _shearFunc(self, psiEtaZeta, *coeffs):
            coeffs = coeffs[0]
            return psiEtaZeta[:,1]*coeffs[0]

    def _calcShearGrad(self, psiEtaZeta, h=1e-8):
        shearCoeffs = self.spanModCoeffs[:self.nSpanModCoeffs[0]]
        nPts = len(psiEtaZeta[:,1])
        return np.vstack([np.zeros(nPts), shearCoeffs[0]*np.ones(nPts)]).T

    #Returns angle of twist (radians) - default linear dist along eta
    def _twistFunc(self, psiEtaZeta, *coeffs):
        coeffs = coeffs[0]
        return coeffs[0]*psiEtaZeta[:,1]

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
    def calcGrad(self, psiEtaZeta, h=1e-8):
        dZetadPsi = (self.offsetFunc(psiEtaZeta, self.shapeOffsets) +
                     self._calcSpanModGrad(psiEtaZeta, h)[:,0] +
                     self._calcClassGrad(psiEtaZeta, h)[:,0] * 
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
                    dPsiZetaEtadCoeffVec = np.array([np.diagonal(paramJac)]).T
                    dXYZdCoeffVec = np.dot(ptJac, dPsiZetaEtadCoeffVec)
                    #Add to partial derivatives
                    totalJac[3*_:3*(_+1), 3*__:3*(__+1)] += np.diag(dXYZdCoeffVec.flatten())
        else:
            totalJac = None
        return totalJac

    #dPsiEtaZetadCsClassCoeffs (Analytical)
    def _calcCsClassJacobian(self, psiEtaZeta, h=1e-8):
        nCoeffs = len(self.csClassCoeffs)
        if nCoeffs>0:
            csClassJac = np.zeros((len(psiEtaZeta.flatten()), 3*nCoeffs))
            csClassJac[2::3, 2::3] = self.csGeo._calcClassJacobian(psiEtaZeta[:,0], h)
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
            spanModJac[2::3, 5] = -psiEtaZeta[:,0]*psiEtaZeta[:,1]*np.power(self.spanModCoeffs[1]*psiEtaZeta[:,1],-2.0) #Twist, 1 coeff linear dist
        else:
            spanModJac = None
        return spanModJac

    #dPsiEtaZetadShapeCoeffs (Analytical)
    def _calcShapeJacobian(self, psiEtaZeta, h=1e-8):
        nx,ny = self.order
        nCoeffs = len(self.shapeCoeffs)
        if nCoeffs>0:
            shapeJac = bernstein2DJacobian(psiEtaZeta[:,0], psiEtaZeta[:,1], nx, ny, h)
            for _ in range(nCoeffs):
                shapeJac[:,_] *= self.csClassFunc(psiEtaZeta,self.csClassCoeffs)
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

    #dXYZdPsiEtaZeta (analytical)
    def _calcPtsJacobian(self, psiEtaZeta, h=1e-8):
        ptsJac = np.zeros((len(psiEtaZeta.flatten()), 3))
        ptsJac[0::3, 0] = self.refLen(psiEtaZeta, self.chordCoeffs)
        ptsJac[0::3, 1] = psiEtaZeta[:,0]*(self.chordCoeffs[1]-self.chordCoeffs[0]) + self.spanModCoeffs[0]
        ptsJac[1::3, 1] = self.refSpan
        ptsJac[2::3, 1] = psiEtaZeta[:,2]*(self.chordCoeffs[1]-self.chordCoeffs[0])
        ptsJac[2::3, 2] = self.refLen(psiEtaZeta, self.chordCoeffs)
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