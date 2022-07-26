import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt

class CST2DParam():
    """
    Class for storing information about cst parameterization in 2d

    Parameters
    ----------
    coords: ndarray
        List of x and z coordinates for baseline shape (2d), x,z=[N,2]

    classFunc: function
        Function that defines class of shapes of shapes, must be [zeta]=f([psi])

    shapePoly: function
        Optional user input of class function (ex: Bernstein Polynomial)
        must be f(psi,order,termOrder)

    order: int
        Order of shape polynomial, this dvars=order+1

    shapeCoeffs: list
        Defines augmenting coefficients for shape function, also defines order of shape function

    shapeOffset: float
        Z Offset for parameterization (ex: Blunt TE offset)

    zScale: float
        Scale factor, for explicit negative z values use scale=-1.0

    TODO Introduce designing on curved centerline
         Handle vertical sections

    """

    def __init__(self, coords, classFunc=None, shapePoly=None, order=5, classCoeffs=[], shapeCoeffs=[],
                 shapeOffset=0.0, zScale=1.0, refLen=1.0):
        self.origCoords = coords
        self.coords = self.origCoords
        self.classFunc = classFunc
        self.classCoeffs = classCoeffs
        self.shapePoly = self.bernstein if shapePoly==None else shapePoly
        self.order = int(order)
        self.shapeCoeffs = np.ones(order+1)*zScale if len(shapeCoeffs)!=self.order+1 else np.array(shapeCoeffs)*zScale
        self.shapeOffset = shapeOffset
        self.refLen = refLen
        self.psiZeta = None
        self.updatePsiZeta()

    def calcZetaToZ(self, zetaVals):
        return np.array([zeta*self.refLen for zeta in zetaVals])

    def calcZToZeta(self, zVals):
        return np.array([z/(self.refLen) for z in zVals])

    def calcPsiToX(self, psiVals):
        return np.array([psi*self.refLen for psi in psiVals])

    def calcXToPsi(self, xVals):
        return np.array([x/self.refLen for x in xVals])

    def calcZeta(self, psiVals):
        zetaVals = []
        for psi in psiVals:
            zeta = self.classFunc(psi,self.classCoeffs)*self.shapeFunction(psi) + psi*(self.shapeOffset/self.refLen)
            zetaVals.append(zeta)
        return np.array(zetaVals)

    def calcPsi(self, zetaVals):
        psiVals = []
        for zeta in zetaVals:
            def solveX(psi):
                return self.calcZeta([psi])[0]-zeta
            psiVals.append(scp.fsolve(solveX,[0]))
        return np.array(psiVals)

    def getParam(self):
        return self.psiZeta

    def updateCoords(self):
        self.coords = np.transpose(np.array([self.calcPsiToX(self.psiZeta[:,0]),self.calcZetaToZ(self.psiZeta[:,1])]))

    def getCoords(self):
        return self.coords

    def updatePsiZeta(self):
        psiVals = [x/self.refLen for x in self.coords[:,0]]
        self.psiZeta = np.array(list(zip(psiVals, self.calcZeta(psiVals))))

    def setPsiZeta(self, psiVals):
        self.psiZeta = np.array(list(zip(psiVals, self.calcZeta(psiVals))))

    def calcXToZ(self, xVals):
        zetaVals = self.calcZeta(self.calcXToPsi(xVals))
        return np.array(zetaVals)*self.refLen

    def shapeFunction(self,psi):
        polyTerms = []
        for term in range(self.order+1):
            polyTerms.append(self.shapeCoeffs[term]*self.shapePoly(psi,self.order,term))
        return sum(polyTerms)

    def bernstein(self,psi,n,r):
        #Default Class Function, Bernstein Polynomials
        binCoeff = np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r))
        return binCoeff*np.power(psi,r)*np.power(1-psi,n-r)

    def updateCoeffs(self,*coeffs):
        coeffs=coeffs[0]
        n1 = len(self.shapeCoeffs)
        if len(coeffs)>=n1+1 and n1>0:
            self.shapeCoeffs = coeffs[:n1]
            self.shapeOffset = coeffs[n1]
        n2 = n1+len(self.classCoeffs)
        if len(coeffs)>n2 and n2>n1:
            self.classCoeffs = coeffs[n1+1:n2]

    def getCoeffs(self):
        coeffs = []
        for _ in self.shapeCoeffs:
            coeffs.append(_)
        coeffs.append(self.shapeOffset)
        for _ in self.classCoeffs:
            coeffs.append(_)
        return coeffs

    def fitPsiZeta(self):
        #Perform a fit of the coefficients
        def curve(xVals,*coeffs):
            self.updateCoeffs()
            
            psiVals = [x/self.refLen for x in xVals]
            zetaVals = self.calcZeta(psiVals)
            return self.calcZetaToZ(zetaVals)
        #TODO Fix warning for cov params being inf in certain conditions
        coeffs,cov = scp.curve_fit(curve,self.coords[:,0],self.coords[:,1],self.shapeCoeffs+[self.shapeOffset]+self.classCoeffs)
        self.updateCoeffs
        self.updatePsiZeta()

    def printResiduals(self):
        err = np.array([np.linalg.norm(self.origCoords[_]-self.coords[_]) for _ in range(len(self.origCoords))])
        print('Residual statistics\n'
              'Max Error: ' + np.max(err) +'\n'
              'Total Error: ' + np.sum(err) + '\n')

    #def maxXZ(self):


class CSTAirfoil2D(CST2DParam):
    """
    Class For Storing CST Parameterization Information For An Airfoil
    Pseudo-2D requires upper and lower surface
    """
    def __init__(self, coords, classFunc=None, shapePoly=None, order=7, classCoeffs=[0.5,1.0], shapeCoeffs=[], shapeOffset=0.0, zScale=1.0):
        self.powerTerms = classCoeffs
        super().__init__(coords=coords, classFunc=self.airfoilClassFunc, shapePoly=shapePoly, order=order, classCoeffs=[], shapeCoeffs=shapeCoeffs, shapeOffset=shapeOffset, zScale=zScale)

    def airfoilClassFunc(self, psi, *coeffs):
        return np.power(psi,self.powerTerms[0])*np.power(1-psi,self.powerTerms[1])


##Plot Airfoil ex:

# fig = plt.subplot()
# x=np.linspace(0,1.0,100)
# z=np.zeros_like(x)
# arr = np.array(list(zip(x,z)))

# cst = CSTAirfoil2D(arr,order=7)
# #cst.fitPsiZeta()
# cst.updatePsiZeta()
# cst.updateCoords()
# cstX = cst.coords[:,0]
# cstZ = cst.coords[:,1]
# plt.plot(cstX,cstZ,'-',label='fitUpper')

# cstLower = CSTAirfoil2D(arr,order=7,zScale=-1.0)
# cstLower.updatePsiZeta()
# cstLower.updateCoords()
# cstXLower = cstLower.coords[:,0]
# cstZLower = cstLower.coords[:,1]
# plt.plot(cstXLower,cstZLower,'-',label='fitLower')

# plt.title('Base Airfoil Shape - '+str(cst.shapeCoeffs)+', '+str(np.array([cst.shapeOffset])))
# plt.legend()
# plt.show()