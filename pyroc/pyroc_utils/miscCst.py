from miscVar import *
import numpy as np

def binaryCoefficients(n):
    binCoeffs = []
    for r in range(n+1): #Loop through terms
        binCoeffs.append(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))
    return np.array(binCoeffs)

#Function to compute bernstein polynomial
def bernstein(psiVals, n):
    orderRange = np.arange(n+1)
    binCoeffs = binaryCoefficients(n)
    if isScalar(psiVals):
        terms = binCoeffs*np.power(psiVals,orderRange)*np.power(1-psiVals,n-orderRange)
    else:
        terms = []
        for psi in psiVals:
            terms.append(binCoeffs*np.power(psi,orderRange)*np.power(1-psi,n-orderRange))
    return np.array(terms) #Sum of terms for any psi is 1