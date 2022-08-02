from .miscVar import *
from .miscCst import *
import numpy as np
import scipy.optimize as scp

class CST2DParam(object):
    """
    Baseclass for storing information about cst parameterization in 2d

    Parameters
    ----------
    coords: ndarray
        List of x and z coordinates for baseline shape (2d), x,z=ndarray(N,2)

    classFunc: function
        Function that defines class of shapes ie airfoil, must be [zeta]=f([psi])

    classCOeffs: list of floats, len=order+1
        Defines augmenting coefficients for class function

    shapeCoeffs: list of floats, default len=2
        Defines weight coefficients for shape function

    masks: list of bool/int
        Must be length of used coefficients, atleast order+1
        A 1/True indicates a value is masked, a 0/False indicates it is free
        If dv is masked, one cannot change dv from init value with updateCoeffs()

    order: int
        Order of shape polynomial
        Number of shape coefficients is order+1

    shapeOffset: float, defaut=0.0
        Z Offset for parameterization (ex: Blunt TE offset)

    refLen: float, default=1.0
        Chord length to use

    shapeScale: float, default=1.0
        If shapeCoeffs is not given, this value will be used for initial shape coeffs.
        shapeScale=-1.0 can be used for a -z surface initialization

    TODO -Designing on curved centerline/variable camber?
         -Error checking
            Coeffs?
            BCs?
    """

    def __init__(self, coords, classFunc=None, classCoeffs=[], shapeCoeffs=[], masks=[], 
                 order=5, shapeOffset=0.0, refLen=1.0, shapeScale=1.0):
        #Original coordinates. Used for comparing fit, ie printing fit residuals
        self.origCoords = coords
        #Number of points
        self.nPts = len(self.origCoords[:,0])
        #Internally stored, updated coordinates
        self.coords = self.origCoords
        #reference length/chord
        self.refLen = refLen

        #Function that controls class (Baseline shape before augmentation)
        self.classFunc = self.defaultClassFunction if classFunc is None else classFunc
        #Coefficients for class function
        self.classCoeffs = classCoeffs

        #Order of augmenting polynomial
        self.order = int(order)
        self.shapeFunc = self.defaultShapeFunction
        #Coeffs for shape function, weights for augmenting bernstein polynomials
        #   Use shapeScale=-1.0 to set surface negative
        if len(shapeCoeffs)==self.order+1:
            self.shapeCoeffs = shapeCoeffs
        else:
            self.shapeCoeffs =  [shapeScale for _ in range(self.order+1)]

        #Offset coefficient for 'trailing edge', offset at psi=1.0 is shapeOffset
        self.shapeOffset = shapeOffset
        #Masks on specific coefficients
        self.nCoeff = len(self.getCoeffs())
        self.masks = [0 for _ in range(self.nCoeff)] if len(masks)==0 else masks

        #Set initial parameterization values from original coordinates
        self.psiZeta = self.coords2PsiZeta(self.origCoords)

        #Default Class Function, 0
    def defaultClassFunction(self, psiVals, *coeffs):
        coeffs = coeffs[0]
        return np.zeros(len(psiVals))

    #Default Shape Function, Bernstein Polynomials
    def defaultShapeFunction(self,psiVals,*coeffs):
        coeffs = coeffs[0] #List of coefficients
        augments = np.dot(bernstein(psiVals, self.order),np.array([coeffs]).T)
        return augments.flatten()

    # Functions for converting arrays of coordinates from parametric space to cartesian
    def calcX2Psi(self, xVals):
        return xVals/self.refLen

    def calcZ2Zeta(self, zVals):
        return zVals/self.refLen

    def calcPsi2X(self, psiVals):
        return psiVals*self.refLen

    def calcZeta2Z(self, zetaVals):
        return zetaVals*self.refLen

    #Estimates psi vals from zeta values
    ##Broken, needs to have better initial guess
    def calcPsi(self, zetaVals):
        def objFunc(psiVals):
            return self.calcZeta(psiVals)-zetaVals
        psiVals = scp.fsolve(objFunc,0 if isScalar(zetaVals) else np.zeros(len(zetaVals)))
        return psiVals

    #Explicitly calculate zeta values
    def calcZeta(self, psiVals):
        return self.classFunc(psiVals,self.classCoeffs)*self.shapeFunc(psiVals,self.shapeCoeffs) + psiVals*(self.shapeOffset/self.refLen)

    #Update zeta values from internal psi values
    def updateZeta(self):
        self.psiZeta[:,1] = self.calcZeta(self.psiZeta[:,0])
        return self.psiZeta

    #Calculate cartesian of any parametric set and return
    def calcCoords(self, psiZeta):
        return np.vstack([self.calcPsi2X(psiZeta[:,0]),self.calcZeta2Z(psiZeta[:,1])]).T

    #Update internal coordinates from internal parametric coords and return
    def updateCoords(self):
        self.updateZeta()
        self.coords = self.calcCoords(self.psiZeta)
        return self.coords

    #Calculate psi,zeta from coords
    def coords2PsiZeta(self, coords):
        return np.vstack([self.calcX2Psi(coords[:,0]),self.calcZ2Zeta(coords[:,1])]).T

    #Set Psi values and update zeta
    def setPsiZeta(self, psiVals):
        zetaVals = self.calcZeta(psiVals)
        self.psiZeta = np.vstack([psiVals,zetaVals]).T
        return self.psiZeta

    #Calculate ZVals from XVals
    def calcX2Z(self, xVals):
        zVals = self.calcZeta2Z(self.calcZeta(self.calcX2Psi(xVals)))
        return zVals

    #Calculate XVals from ZVals
    def calcZ2X(self, zVals):
        xVals = self.calcPsi2X(self.calcPsi(self.calcZ2Zeta(zVals)))
        return xVals

    #Update internal coefficients from coefficients list,
    #internal coefficient list length must be same as the one being set
    #Order of coeffs is [CLASS, SHAPE, OFFSET]
    def updateCoeffs(self,*coeffs):
        coeffs=coeffs[0]
        nC = len(coeffs)
        n1 = len(self.classCoeffs)
        n2 = n1+self.order+1
        for _ in range(nC):
            if not self.masks[_]:
                if _<n1:
                    self.classCoeffs[_] = coeffs[_]
                elif _<n2 and _>=n1:
                    i = _-n1
                    self.shapeCoeffs[i] = coeffs[_]
                elif _>=n2:
                    self.shapeOffset = coeffs[_]
        return 0

    #Return an ordered list of coeffs from internal coeff values
    #Order of coeffs is [CLASS, SHAPE, OFFSET]
    def getCoeffs(self):
        coeffs = []
        for _ in self.classCoeffs:
            coeffs.append(_)
        for _ in self.shapeCoeffs:
            coeffs.append(_)
        coeffs.append(self.shapeOffset)
        return coeffs

    #Derivative functions, FD by default because class func is arbitrary
    #dZetadPsi
    def calcDeriv(self, psiVals, h=1e-8):
        psih = np.copy(psiVals)
        for _ in range(len(psih)):
            if psih[_] == 1.0:
                psih[_] -= h
            else:
                psih[_] += h
        dZetadPsi = (self.calcZeta(psih)-self.calcZeta(psiVals))/h
        return np.array(dZetadPsi)

    def getDeriv(self):
        return self.calcDeriv(self.psiZeta[:,0])

    #dZetadCoeffs jacobian matrix [nPts, nCoeffs]
    def calcJacobian(self, psiVals, h=1e-8):
        jacClass = self._calcClassJacobian(psiVals)
        jacShape = self._calcShapeJacobian(psiVals)
        jacOffset = self._calcOffsetJacobian(psiVals)
        jacTotal = np.array(list(zip(jacClass, jacShape, jacOffset)))
        for _ in range(self.nCoeff):
            if self.masks[_]:
                jacTotal[:,_] = np.zeros(self.nCoeff)
        return jacTotal

    def getJacobian(self):
        return self.calcJacobian(self.coords)

    #Perform a fit of the coefficients
    def fit2d(self):
        def curve(xVals,*coeffs):
            self.updateCoeffs(coeffs)
            return self.calcX2Z(xVals)
        #TODO Fix warning for cov params being inf in certain conditions
        coeffs,cov = scp.curve_fit(curve,self.coords[:,0],self.coords[:,1],self.getCoeffs())
        self.updateCoeffs(coeffs)
        self.updateZeta()
        return 0

    #Print residual info between original coords and current coords
    #Mainly used for fit
    def printFitResiduals(self):
        err = [np.linalg.norm(self.origCoords[_]-self.coords[_]) for _ in range(self.nPts)]
        print('Residual statistics\n',
              'Avg Diff: ',np.average(err),'\n',
              'Max Diff: ',np.max(err),'\n',
              'Total Diff: ',np.sum(err),'\n')
        return 0

    #dZetadClassCoeffs - default FD, dependant on class func
    def _calcClassJacobian(self, psiVals, h=1e-8):
        zeta = self.calcZeta(psiVals)
        nClassCoeffs = len(self.classCoeffs)
        jacClass = np.zeros((len(psiVals)),nClassCoeffs)
        for _ in range(nClassCoeffs):
            self.classCoeffs[_] += h
            zetah = self.calcZeta(psiVals)
            jacClass[:,_] = (zetah-zeta)/h
            self.classCoeffs[_] -= h
        return jacClass

    #dZetadShapeCoeffs - analytic, independent of class func
    def _calcShapeJacobian(self, psiVals, h=1e-8):
        n = self.order
        nShapeCoeffs = len(self.shapeCoeffs)
        binCoeffs = binaryCoefficients(n)
        jacShape = np.zeros((len(psiVals)),nShapeCoeffs)
        for r in range(n+1):
            jacShape[:,r] = self.classFunc(psiVals,self.classCoeffs)*binCoeffs[r]*np.power(psiVals,r)*np.power(1-psiVals,n-r)
        return jacShape

    #dZetadOffset - analytic, independent of class func
    def _calcOffsetJacobian(self, psiVals, h=1e-8):
        return psiVals/self.refLen


class CSTAirfoil2D(CST2DParam):
    """
    Class For Storing CST Parameterization Information For An Airfoil
    Pseudo-2D requires upper and lower surface
    """
    def __init__(self, coords, classFunc=None, classCoeffs=[0.5,1.0], shapeCoeffs=[], masks=[],
                 order=5, shapeOffset=0.0, refLen=1.0, shapeScale=1.0):
        self.classFunc = self.airfoilClassFunc if classFunc is None else classFunc
        super().__init__(coords=coords, classFunc=self.classFunc, classCoeffs=classCoeffs, shapeCoeffs=shapeCoeffs, masks=masks,
                         order=order, shapeOffset=shapeOffset, refLen=refLen, shapeScale=shapeScale)

    #Class Function for an airfoil
    def airfoilClassFunc(self, psiVals, *coeffs):
        coeffs = coeffs[0]
        return np.power(psiVals,coeffs[0])*np.power(1-psiVals,coeffs[1])

    #LE Radius Functions
    def updateLERadius(self, r):
        self.shapeCoeffs[0] = np.sqrt(2*r/self.refLen)
        return 0

    def getLERadius(self):
        r = 0.5*self.refLen*np.power(self.shapeCoeffs[0],2)
        return r

    #Boattail Angle Function
    def updateTEAngle(self, beta):
        self.shapeCoeffs[-1] = np.tan(beta)+self.shapeOffset
        return 0

    def getTEAngle(self):
        beta = np.arctan(self.shapeCoeffs[-1]-self.shapeOffset)
        return beta

    #Analytical derivatives based on airfoil assumption
    #dZetadPsi - analytic, class func is known, split into S' C' and Zeta'
    def calcDeriv(self, psiVals, h=1e-8):
        for _ in range(len(psiVals)):
            if psiVals[_] == 0.0:
                psiVals[_] += h
            elif psiVals[_] == 1.0:
                psiVals[_] -= h
        
        dZetadPsi = (self._calcClassDeriv(psiVals)*self.shapeFunc(psiVals,self.shapeCoeffs) +
                     self.classFunc(psiVals,self.classCoeffs)*self._calcShapeDeriv(psiVals) +
                     self.shapeOffset/self.refLen)
        return dZetadPsi

    def _calcClassDeriv(self, psiVals, h=1e-8):
        n1,n2 = self.classCoeffs
        dClassdPsi = n1*self.classFunc(psiVals,[n1-1.0,n2])-n2*self.classFunc(psiVals,[n1,n2-1.0])
        return dClassdPsi

    def _calcShapeDeriv(self, psiVals, h=1e-8):
        n,r = self.order,np.arange(self.order+1) #binary coefficient constants
        shapeSum = np.array([np.array(self.shapeCoeffs)*binaryCoefficients(n) * (
            r*np.power(psi,r-1)*np.power(1-psi,n-r)-(n-r)*np.power(psi,r)*np.power(1-psi,n-r-1)
        ) for psi in psiVals])
        dShapedPsi = np.sum(shapeSum,1)
        return dShapedPsi

    #dZetadClassCoeff - analytic, class func is known
    def _calcClassJacobian(self, psiVals, h=1e-8):
        n1,n2 = self.classCoeffs
        dZetadN1 = n1*np.power(psiVals,n1-1)*np.power(1-psiVals,n2)*self.shapeFunc(psiVals,self.shapeCoeffs)
        dZetadN2 = -n2*np.power(psiVals,n1)*np.power(1-psiVals,n2-1)*self.shapeFunc(psiVals,self.shapeCoeffs)
        return np.vstack([dZetadN1,dZetadN2]).T