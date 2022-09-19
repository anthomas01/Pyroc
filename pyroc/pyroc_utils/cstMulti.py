from .cst3d import *
from collections import OrderedDict
from scipy import sparse
import numpy as np
import os

class CSTMultiParam(object):
    #Class for managing multiple CST geometry objects

    def __init__(self):

        self.coef = None
        self.embeddedSurfaces = {}
        self.embeddedParams = {}

    def attachPoints(self, coordinates, ptSetName):
        #Project points to surface/CST geom object if some were passed in
        self.embeddedSurfaces[ptSetName] = EmbeddedSurface(coordinates)

    def getAttachedPoints(self, ptSetName):
        #Refresh attached points
        self.embeddedSurfaces[ptSetName].updateCoords()
        #return coordinates
        return self.embeddedSurfaces[ptSetName].coordinates

    def attachCSTParam(self, CSTParam, paramName, embeddedParams=None):
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        self.embeddedParams[paramName] = EmbeddedParameterization(CSTParam, embeddedParams)

    def updateCoeffs(self):
        "Update CST coefficients"
        i = 0
        for paramName in self.embeddedParams:
            self.embeddedParams[paramName].updateCoeffs(self.coef[i])
            i += 1

    def setCoeffs(self):
        "Update internal coefficients stored in multi from CST coefficients"
        self.coef = []
        for paramName in self.embeddedParams:
            paramCoeffs = self.embeddedParams[paramName].coeffs
            self.coef.append(paramCoeffs)

    #TODO Make this more efficient
    def calcdPtdCoef(self, ptSetName):
        #Get nPts
        embeddedSurface = self.embeddedSurfaces[ptSetName]
        nPts = embeddedSurface.N

        #find nCoef and coefMap
        nCoef = 0
        coefMap = {}
        for paramName in np.unique(embeddedSurface.paramMap):
            paramCoeffs = self.embeddedParams[paramName].coeffs
            coefMap[paramName] = [_+nCoef for _ in range(len(paramCoeffs))]
            nCoef += len(paramCoeffs)

        #size of dPtdCoef will be 3*Npts x 3*Ncoef
        dPtdCoef = np.zeros((3*nPts, 3*nCoef))

        for _ in range(nPts):
            paramName = embeddedSurface.paramMap[_]
            param = self.embeddedParams[paramName]
            choppedJac = param.calcdPtdCoef(embeddedSurface.coordinates[_]) #This is the expensive operation, vectorize here?
            #Insert into dPtdCoef based on coefficient mapping
            i = 0
            for __ in coefMap[paramName]:
                dPtdCoef[3*_:3*(_+1), 3*__:3*(__+1)] = choppedJac[:, 3*i:3*(i+1)]
                i+=1

        #Finally, store variable as sparse matrix
        self.embeddedSurfaces[ptSetName].dPtdCoef = sparse.csr_matrix(dPtdCoef)



class EmbeddedSurface(object):
    #Class for managing multiple CST surfaces

    def __init__(self, coordinates, embeddedParams, fit=True):
        #Read in coords, ndarray (N,3)
        self.coordinates = coordinates
        self.embeddedParams = embeddedParams
        self.dPtdCoef = None
        self.N = len(coordinates)

        #Determine which embedded parameterization object each coordinate belongs to
        self.paramMap = self.mapCoords2Params() #ndarray (N,1), ndarray (N,3)

        #Group coordinates by parameterization and apply fit
        self.parameterization = self.fitParams()
    
    #TODO Make this more efficient
    def mapCoords2Params(self):
        #Returns array that is the same size as the coordinate list
        #Each indice contains the name of an externally stored EmbeddedParameterization object

        paramMap = np.zeros((self.N,1), dtype=str)
        for _ in range(self.N):
            #Brute force least square distance?
            closest = 1e8
            for paramName in self.embeddedParams:
                param = self.embeddedParams[paramName]
                psiEtaZeta = param.calcParameterization(self.coordinates[_])
                computedXYZ = param.calcCoords(psiEtaZeta)

                d2 = np.sum(np.power(self.coordinates[_]-computedXYZ,2))
                if paramMap[_]=='' or d2<closest:
                    closest = d2
                    paramMap[_] = paramName

        return paramMap

    #TODO Make this more efficient
    def updateCoords(self):
        coordinates = np.zeros((self.N,3))
        for _ in range(self.N):
            paramName = self.paramMap[_]
            param = self.embeddedParams[paramName]
            psiEtaZeta = self.parameterization[_]
            coordinates[_] = param.calcCoords(psiEtaZeta)
        self.coordinates = coordinates

    def fitParams(self, performFit=True):
        #Group coordinates by parameterization
        paramCoords = {}
        uniqueParams = np.unique(self.paramMap)
        for paramName in uniqueParams:
            paramCoords[paramName] = []
        for _ in range(self.N):
            paramCoords[self.paramMap[_]].append(self.coordinates[_])

        #Perform coefficient fit if performFit
        if performFit:
            for paramName in uniqueParams:
                param = self.embeddedParams[paramName]



class EmbeddedParameterization(object):
    #Class for managing single CST parameterization object

    def __init__(self, param):
        self.param = param
        self.coeffs = param.getCoeffs()
        self.dependantCoeffs = OrderedDict()

    def addConstraint(self, name, IVEmbeddedParam, IVList, DVList):
        # IV/DV's are lists of indices for coefficients, check cst3d.py for positions
        # IV/DV lists must be same length
        self.dependantCoeffs[name] = {
            'IVParam' : IVEmbeddedParam,
            'IVList' : np.array(IVList),
            'DVList' : np.array(DVList),
        }

    def applyConstraints(self):
        # Apply constraints to coefficients
        for dependantCoeffName in self.dependantCoeffs:
            dependantCoeff = self.dependantCoeffs[dependantCoeffName]
            n = len(dependantCoeff['IVList'])
            for _ in range(n):
                i = dependantCoeff['IVList'][_]
                j = dependantCoeff['DVList'][_]
                self.coeffs[j] = dependantCoeff['IVParam'].coeffs[i]

    def updateCoeffs(self):
        self.applyConstraints()
        self.param.updateCoeffs(self.coeffs)

    def fitCoefficients(self, coordinates):
        self.updateCoeffs()
        self.applyMasks()
        self.param.fit3d(coordinates)
        self.removeMasks()

    def calcParameterization(self, coordinates):
        return self.param.coords2PsiEtaZeta(coordinates)

    def calcCoords(self, psiEtaZeta):
        return self.param.calcCoords(psiEtaZeta)

    def calcdPtdCoef(self, coordinates):
        return self.param.calcJacobian(coordinates)

    def applyMasks(self):
        for dependantCoeffName in self.dependantCoeffs:
            dependantCoeff = self.dependantCoeffs[dependantCoeffName]
            for _ in dependantCoeff['DVList']:
                self.param.masks[_] = 1

    def removeMasks(self):
        self.param.masks = [0 for _ in range(len(self.param.masks))]