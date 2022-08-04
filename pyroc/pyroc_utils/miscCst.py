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

def bernstein2D(psiVals, etaVals, nx, ny):
    xOrderRange = np.arange(nx+1)
    yOrderRange = np.arange(ny+1)
    xBinCoeffs = binaryCoefficients(nx)
    yBinCoeffs = binaryCoefficients(ny)

    terms = []
    for _ in range(len(psiVals)):
        psi, eta = psiVals[_], etaVals[_]
        shape = np.zeros((nx+1)*(ny+1))
        xShape = xBinCoeffs*np.power(psi,xOrderRange)*np.power(1-psi,nx-xOrderRange)
        yShape = yBinCoeffs*np.power(eta,yOrderRange)*np.power(1-eta,ny-yOrderRange)
        for __ in range(ny+1):
            shape[__::ny+1] = xShape*yShape[__]
        terms.append(shape)
    return np.array(terms)

    
    