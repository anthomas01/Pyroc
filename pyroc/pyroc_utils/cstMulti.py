from .cst3d import *
from collections import OrderedDict
from scipy import sparse
from scipy.spatial import Delaunay
import numpy as np
import os

class CSTMultiParam(object):
    #Class for managing multiple CST geometry objects

    def __init__(self):

        self.coef = None
        self.embeddedSurfaces = OrderedDict()
        self.embeddedParams = OrderedDict()
        #Refit when adding new coords?
        self.fit = False

    def attachPoints(self, coordinates, ptSetName, embeddedParams=None, parameterization=False):
        #Project points to surface/CST geom object if some were passed in
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        self.embeddedSurfaces[ptSetName] = EmbeddedSurface(coordinates, embeddedParams, parameterization=parameterization)

    def attachParameterization(self, parameterization, ptSetName, embeddedParams=None):
        #Project points to surface/CST geom object if some were passed in
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        self.embeddedSurfaces[ptSetName] = EmbeddedSurface(parameterization, embeddedParams, self.fit)
        if self.fit: self.setCoeffs()

    def getAttachedPoints(self, ptSetName):
        #Refresh attached points
        self.embeddedSurfaces[ptSetName].updateCoords()
        #return coordinates
        return self.embeddedSurfaces[ptSetName].coordinates

    def attachCSTParam(self, CSTParam, paramName, embeddedParams=None):
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        embeddedParams[paramName] = EmbeddedParameterization(CSTParam)

    def updateCoeffs(self):
        "Update CST coefficients"
        i = 0
        for paramName in self.embeddedParams:
            self.embeddedParams[paramName].coeffs = self.coef[i].copy()
            self.embeddedParams[paramName].updateCoeffs()
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

        for paramName in np.unique(embeddedSurface.paramMap):
            param = self.embeddedParams[paramName]
            coordIndices = np.where(np.array(embeddedSurface.paramMap)==paramName)[0]
            choppedJac = param.calcdPtdCoef(embeddedSurface.coordinates[coordIndices])
            #Insert into dPtdCoef based on coefficient mapping
            i = 0
            for _ in coordIndices:
                j = 0
                for __ in coefMap[paramName]:
                    dPtdCoef[3*_:3*(_+1), 3*__:3*(__+1)] = choppedJac[3*i:3*(i+1), 3*j:3*(j+1)]
                    j+=1
                i+=1

        #Finally, store variable as sparse matrix
        self.embeddedSurfaces[ptSetName].dPtdCoef = sparse.csr_matrix(dPtdCoef)

    def plotMulti(self, ax):
        psiVals = np.linspace(0, 1, 25)
        etaVals = np.linspace(0, 1, 10)
        psi, eta = np.meshgrid(psiVals, etaVals)
        psi, eta = psi.flatten(), eta.flatten()
        psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T
        tri = Delaunay(np.array([psi, eta]).T)

        for paramName in self.embeddedParams:
            param = self.embeddedParams[paramName]
            param.param.setPsiEtaZeta(psiEtaZeta.copy())
            coords = param.param.updateCoords()
            ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2], triangles=tri.simplices.copy())

    def fitMulti(self, file):
        #Project points to surface/CST geom object (csv input)
        if (os.path.exists(file)):
            f = open(file,'r')
            lines = f.readlines()
            coordinates = []
            for line in lines:
                coordinates.append([float(_) for _ in line.split(',')])
            self.embeddedSurfaces['fit'] = EmbeddedSurface(np.array(coordinates), self.embeddedParams, fit=True)
            self.setCoeffs()
            f.close()
        else:
            raise Exception('%s does not exist' % file)


class EmbeddedSurface(object):
    #Class for managing multiple CST surfaces

    def __init__(self, coordinates, embeddedParams, fit=False, fitIter=1, parameterization=False):
        self.embeddedParams = embeddedParams
        self.N = len(coordinates)

        if parameterization:
            for _ in range(fitIter):
                self.parameterization = coordinates
                #Determine which embedded parameterization object each coordinate belongs to
                self.paramMap = self.mapParameterization2Params(self.parameterization) #list

                self.coordinates = self.parameterization
                self.updateCoords()

        else:
            #Read in coords, ndarray (N,3)
            self.coordinates = coordinates

            for _ in range(fitIter):
                #Determine which embedded parameterization object each coordinate belongs to
                self.paramMap = self.mapCoords2Params() #list

                #Group coordinates by parameterization and apply fit if True
                self.parameterization = self.fitParams(fit)
    
    #TODO Make these more efficient
    def mapCoords2Params(self):
        #Returns array that is the same size as the coordinate list
        #Each indice contains the name of an EmbeddedParameterization object
        paramMap = ['' for _ in range(self.N)]

        for _ in range(self.N):
            #Brute force least square distance?
            closest = 1e8
            for paramName in self.embeddedParams:
                param = self.embeddedParams[paramName]
                psiEtaZeta = param.calcParameterization(self.coordinates[_])
                computedXYZ = param.calcCoords(psiEtaZeta)

                d2 = np.sum(np.power(self.coordinates[_]-computedXYZ,2))
                if d2==closest and paramMap[_-1]==paramName:
                    paramMap[_] = paramName
                elif d2<closest:
                    paramMap[_] = paramName
                    closest = d2
        return paramMap

    def mapParameterization2Params(self, parameterization):
        '''Returns array that is the same size as the coordinate list.
           Each indice contains the name of an EmbeddedParameterization object'''
        N = len(parameterization)
        paramMap = ['' for _ in range(N)]
        for _ in range(N):
            #Brute force least square distance?
            closest = 1e8
            for paramName in self.embeddedParams:
                param = self.embeddedParams[paramName]
                psiEtaZeta = parameterization
                psiEtaZeta[:,2] = param.calcZeta(psiEtaZeta)

                d2 = np.sum(np.power(parameterization-psiEtaZeta,2))
                if d2==closest and paramMap[_-1]==paramName:
                    paramMap[_] = paramName
                elif d2<closest:
                    paramMap[_] = paramName
                    closest = d2
        return paramMap

    #Vectorize based on parameterization and update coordinates
    def updateCoords(self):
        coordinates = np.zeros((self.N,3))
        for paramName in np.unique(self.paramMap):
            param = self.embeddedParams[paramName]
            coordinateIndices = np.where(np.array(self.paramMap)==paramName)[0]
            psiEtaZeta = self.parameterization[coordinateIndices]
            coordinates[coordinateIndices] = param.calcCoords(psiEtaZeta)
        self.coordinates = coordinates

    def fitParams(self, performFit=False):
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
                param.fitCoefficients(paramCoords[paramName])

        #Update parameterization
        parameterization = np.zeros_like(self.coordinates)
        for paramName in uniqueParams:
            param = self.embeddedParams[paramName]
            coordIndices = np.where(np.array(self.paramMap)==paramName)[0]
            parameterization[coordIndices] = param.calcParameterization(self.coordinates[coordIndices])

        return parameterization



class EmbeddedParameterization(object):
    #Class for managing single CST parameterization object

    def __init__(self, param):
        self.param = param
        self.coeffs = self.param.getCoeffs()
        self.dependantCoeffs = OrderedDict()
        self.masks = self.param.masks

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
        self.param.fit3d(np.atleast_2d(coordinates))
        self.coeffs = self.param.getCoeffs()
        self.removeMasks()

    def calcParameterization(self, coordinates):
        return self.param.coords2PsiEtaZeta(np.atleast_2d(coordinates))

    def calcCoords(self, psiEtaZeta):
        return self.param.calcCoords(np.atleast_2d(psiEtaZeta))

    def calcZeta(self, psiEtaZeta):
        return self.param.calcZeta(np.atleast_2d(psiEtaZeta))

    def calcdPtdCoef(self, coordinates):
        return self.param.calcJacobian(np.atleast_2d(coordinates))

    def applyMasks(self):
        for dependantCoeffName in self.dependantCoeffs:
            dependantCoeff = self.dependantCoeffs[dependantCoeffName]
            for _ in dependantCoeff['DVList']:
                self.param.masks[_] = 1

    def removeMasks(self):
        #Reapply original masks
        self.param.masks = self.masks