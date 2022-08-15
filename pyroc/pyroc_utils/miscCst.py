from .miscVar import *
import numpy as np

def binaryCoefficients(n):
    binCoeffs = []
    for r in range(n+1): #Loop through terms
        binCoeffs.append(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))
    return np.array(binCoeffs)

#Function to compute bernstein polynomial
def bernstein1D(psiVals, n):
    orderRange = np.arange(n+1)
    binCoeffs = binaryCoefficients(n)
    if isScalar(psiVals):
        terms = binCoeffs*np.power(psiVals,orderRange)*np.power(1-psiVals,n-orderRange)
    else:
        terms = []
        for psi in psiVals:
            terms.append(binCoeffs*np.power(psi,orderRange)*np.power(1-psi,n-orderRange))
    return np.array(terms) #Sum of terms for any psi is 1

#Derivative of bernstein1D with respect to psi
def bernstein1DDeriv(psiVals, n, h=1e-8):
    orderRange = np.arange(n+1) #binary coefficient constants
    binCoeffs = binaryCoefficients(n)
    if isScalar(psiVals):
        terms = binCoeffs*(orderRange*np.power(psiVals,orderRange-1)*np.power(1-psiVals,n-orderRange) -
                           (n-orderRange)*np.power(psiVals, orderRange)*np.power(1-psiVals,n-orderRange-1))
    else:
        terms = []
        for psi in psiVals:
            terms.append(binCoeffs * (orderRange * np.power(psi, orderRange-1) * np.power(1-psi, n-orderRange) -
                                      (n-orderRange) * np.power(psi, orderRange) * np.power(1-psi, n-orderRange-1)))
    return np.array(terms)

#dBernsteindShapeCoeffs
def bernstein1DJacobian(psiVals, n, h=1e-8):
    if n+1>0:
        binCoeffs = binaryCoefficients(n)
        bernsteinJac = np.zeros((len(psiVals),n+1))
        for _ in range(n+1):
            bernsteinJac[:,_] = binCoeffs[_]*np.power(psiVals,_)*np.power(1-psiVals,n-_)
    else:
        bernsteinJac = None
    return bernsteinJac

#Taylor expanded bernstein polynomial into two dimensions
def bernstein2D(psiVals, etaVals, nx, ny):
    xOrderRange = np.arange(nx+1)
    yOrderRange = np.arange(ny+1)
    xBinCoeffs = binaryCoefficients(nx)
    yBinCoeffs = binaryCoefficients(ny)
    if isScalar(psiVals):
        shape = np.zeros((nx+1)*(ny+1))
        xShape = xBinCoeffs*np.power(psiVals,xOrderRange)*np.power(1-psiVals,nx-xOrderRange)
        yShape = yBinCoeffs*np.power(etaVals,yOrderRange)*np.power(1-etaVals,ny-yOrderRange)
        for __ in range(ny+1):
            shape[__::ny+1] = xShape*yShape[__]
        terms = shape
    else:
        terms = []
        for _ in range(len(psiVals)):
            psi, eta = psiVals[_], etaVals[_]
            shape = np.zeros((nx+1)*(ny+1))
            xShape = xBinCoeffs*np.power(psi,xOrderRange)*np.power(1-psi,nx-xOrderRange)
            yShape = yBinCoeffs*np.power(eta,yOrderRange)*np.power(1-eta,ny-yOrderRange)
            for __ in range(ny+1):
                shape[__::ny+1] = xShape*yShape[__]
            terms.append(shape)
        terms = np.array(terms)
    return terms

#Gradient of 2d Bernstein polynomial
def bernstein2DGrad(psiVals, etaVals, nx, ny, h=1e-8):
    xOrderRange = np.arange(nx+1)
    yOrderRange = np.arange(ny+1)
    xBinCoeffs = binaryCoefficients(nx)
    yBinCoeffs = binaryCoefficients(ny)
    if isScalar(psiVals):
        dShapedPsi = np.zeros((nx+1)*(ny+1))
        dShapedEta = np.zeros((nx+1)*(ny+1))
        xShape = xBinCoeffs*np.power(psiVals,xOrderRange)*np.power(1-psiVals,nx-xOrderRange)
        yShape = yBinCoeffs*np.power(etaVals,yOrderRange)*np.power(1-etaVals,ny-yOrderRange)
        dxShapedPsi = xBinCoeffs*(xOrderRange*np.power(psiVals,xOrderRange-1)*np.power(1-psiVals,nx-xOrderRange) -
                                  (nx-xOrderRange)*np.power(psiVals,xOrderRange)*np.power(1-psiVals,nx-xOrderRange-1))
        dyShapedEta = yBinCoeffs*(yOrderRange*np.power(etaVals,yOrderRange-1)*np.power(1-etaVals,ny-yOrderRange) -
                                  (ny-yOrderRange)*np.power(etaVals,yOrderRange)*np.power(1-etaVals,ny-yOrderRange-1))
        for __ in range(ny+1):
            dShapedPsi[__::ny+1] = dxShapedPsi*yShape[__]
            dShapedEta[__::ny+1] = xShape*dyShapedEta[__]
        termsPsi = dShapedPsi
        termsEta = dShapedEta
    else:
        terms = []
        for _ in range(len(psiVals)):
            psi, eta = psiVals[_], etaVals[_]
            dShapedPsi = np.zeros((nx+1)*(ny+1))
            dShapedEta = np.zeros((nx+1)*(ny+1))
            xShape = xBinCoeffs*np.power(psi,xOrderRange)*np.power(1-psi,nx-xOrderRange)
            yShape = yBinCoeffs*np.power(eta,yOrderRange)*np.power(1-eta,ny-yOrderRange)
            dxShapedPsi = xBinCoeffs*(xOrderRange*np.power(psi,xOrderRange-1)*np.power(1-psi,nx-xOrderRange) -
                                      (nx-xOrderRange)*np.power(psi,xOrderRange)*np.power(1-psi,nx-xOrderRange-1))
            dyShapedEta = yBinCoeffs*(yOrderRange*np.power(eta,yOrderRange-1)*np.power(1-eta,ny-yOrderRange) -
                                      (ny-yOrderRange)*np.power(eta,yOrderRange)*np.power(1-eta,ny-yOrderRange-1))
            for __ in range(ny+1):
                dShapedPsi[__::ny+1] = dxShapedPsi*yShape[__]
                dShapedEta[__::ny+1] = xShape*dyShapedEta[__]
            termsPsi.append(dShapedPsi)
            termsEta.append(dShapedEta)
        termsPsi = np.array(termsPsi)
        termsEta = np.array(termsEta)
    return termsPsi, termsEta

#dBernsteindShapeCoeffs
def bernstein2DJacobian(psiVals, etaVals, nx, ny, h=1e-8):
    nCoeff = (nx+1)*(ny+1)
    if nCoeff>0:
        xBinCoeffs = binaryCoefficients(nx)
        yBinCoeffs = binaryCoefficients(ny)
        bernsteinJac = np.zeros((len(psiVals),nCoeff))

        for i in range(nx+1):
            xShape = xBinCoeffs*np.power(psiVals,i)*np.power(1-psiVals,nx-i)
            for j in range(ny+1):
                yShape = yBinCoeffs*np.power(etaVals,j)*np.power(1-etaVals,ny-j)
                bernsteinJac[:,i*(ny+1)+j] = xShape*yShape
    else:
        bernsteinJac = None
    return bernsteinJac