from .cst3d import *
from collections import OrderedDict
from scipy import sparse
from scipy.spatial import Delaunay
import numpy as np
import os
import matplotlib.pyplot as plt

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

    def addConnection(self, connectionName, connectingParam1, connectingParam2, connectionType=0, *args, **kwargs):
        '''
        Add connection between parameterization objects

        Parameters
        -----------

        connectionName - str
        Name for connections dictionary

        connectingParam - EmbeddedParameterization
        Parameterization object to make connection to

        connectionType - str
        Valid options are:
            - 0 (duplicate psi/eta or blunt face, i.e. blunt TE)
                kwargs
                - linear=False, if true the n values are integer
        '''

        if connectionType in [0]:
            #Create blunt face for duplicate Psi, Eta Values
            
            #Set connection dictionaries
            connectingParam1.connectionsDict[connectionName] = {
                'type': 0,
                'connectingParam': connectingParam2,
                'linear': kwargs['linear'] if 'linear' in kwargs else False,
                'initialized': False
            }
            connectingParam2.connectionsDict[connectionName] = {
                'type': 0,
                'connectingParam': connectingParam1,
                'linear': kwargs['linear'] if 'linear' in kwargs else False,
                'initialized': False
            }

        else:
            raise Exception('Connection type %d not implemented' % connectionType)

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
            choppeddPtdCoef = param.calcdPtdCoef(embeddedSurface.coordinates[coordIndices])
            #Insert into dPtdCoef based on coefficient mapping
            i = 0
            for _ in coordIndices:
                j = 0
                for __ in coefMap[paramName]:
                    dPtdCoef[3*_:3*(_+1), 3*__:3*(__+1)] = choppeddPtdCoef[3*i:3*(i+1), 3*j:3*(j+1)]
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
        self.connectionsDict = OrderedDict()
        self.dPtdCoef = None

        if parameterization:
            #Read in parameterization, ndarray (N,3)
            self.parameterization = np.copy(coordinates)
            #Determine which embedded parameterization object each coordinate belongs to
            self.paramMap = self.mapParameterization2Params(self.parameterization) #list

            self.initializeParameterizationConnections()

            self.coordinates = self.parameterization

            self.updateCoords()

        else:
            #Read in coords, ndarray (N,3)
            self.coordinates = np.copy(coordinates)

            for _ in range(fitIter):
                #Determine which embedded parameterization object each coordinate belongs to
                self.paramMap = self.mapCoords2Params() #list

                #Group coordinates by parameterization and apply fit if True
                self.parameterization = self.fitParams(fit)

                self.initializeParameterizationConnections()
    
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
                psiEtaZeta[_,2] = param.calcZeta(psiEtaZeta)
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
                embeddedParam = self.embeddedParams[paramName]
                psiEtaZeta = parameterization
                psiEtaZeta[:,2] = embeddedParam.param.calcZeta(psiEtaZeta)

                d2 = np.sum(np.power(parameterization-psiEtaZeta,2))
                if d2==closest and paramMap[_-1]==paramName:
                    paramMap[_] = paramName
                elif d2<closest:
                    paramMap[_] = paramName
                    closest = d2
        return paramMap

    def initializeParameterizationConnections(self):
        '''
        Determines connection parameters for point set
        '''
        for paramName in self.embeddedParams:
            embeddedParam = self.embeddedParams[paramName]

            for connectionName in embeddedParam.connectionsDict:
                connection = embeddedParam.connectionsDict[connectionName]

                if connection['type'] in [0] and not connection['initialized']:
                    ##dn = f1-(f1-f2)*(n-1)/(N-1)
                    #Need to calculate N and n map for duplicate indices
                    dtypeCounts = int if connection['linear'] else float

                    #Find duplicate psi,eta values for entire point set
                    psiEtaVals = self.parameterization[:,0:1]
                    uniquePsiEta, uniquePsiEtaIndices, countsPsiEta = np.unique(psiEtaVals, return_index=True, return_counts=True, axis=0)
                    duplicatePsiEtaIndex = [i for i in range(len(self.parameterization)) if i not in uniquePsiEtaIndices]

                    totalDuplicatePsiEtaIndices = np.append(uniquePsiEtaIndices[np.where(countsPsiEta>1)][0], duplicatePsiEtaIndex)
                    totalCountsDuplicates = np.ones_like(self.parameterization, dtype=dtypeCounts)
                    countsDuplicates = np.ones_like(self.parameterization, dtype=dtypeCounts)
                    connectingCountsDuplicates = np.ones_like(self.parameterization, dtype=dtypeCounts)

                    for _ in range(len(totalDuplicatePsiEtaIndices)):
                        index = totalDuplicatePsiEtaIndices[_]
                        psiEtaZeta = self.parameterization[index]
                        psiEta = psiEtaZeta[0:1]
                        uniquePsiEtaIndex = np.where(uniquePsiEta==psiEta)[0][0]

                        #Set N for each parameter vector index
                        totalCountsDuplicates[index] = countsPsiEta[uniquePsiEtaIndex]

                        #Set n for each parameter vector index
                        boundZeta1 = embeddedParam.param.calcZeta(np.atleast_2d(psiEtaZeta))[0]
                        boundZeta2 = connection['connectingParam'].param.calcZeta(np.atleast_2d(psiEtaZeta))[0]
                        n1 = 1.0 + (totalCountsDuplicates[_] - 1.0) * (boundZeta1 - self.parameterization[_,2]) / (boundZeta1 - boundZeta2)
                        n2 = 1.0 + (totalCountsDuplicates[_] - 1.0) * (boundZeta2 - self.parameterization[_,2]) / (boundZeta2 - boundZeta1)
                        countsDuplicates[_] = round(n1) if connection['linear'] else n1
                        connectingCountsDuplicates[_] = round(n2) if connection['linear'] else n2

                    #Store needed information
                    connectingParamName = [name for name in self.embeddedParams if embeddedParam==connection['connectingParam']][0]
                    self.connectionsDict[connectionName]['type'] = connection['type']
                    self.connectionsDict[connectionName]['counts_'+paramName] = countsDuplicates
                    self.connectionsDict[connectionName]['counts_'+connectingParamName] = connectingCountsDuplicates
                    self.connectionsDict[connectionName]['totalCounts'] = totalCountsDuplicates

                    #Set to initialized so same connection is not repeated in different order of parameterization
                    connection['initialized'] = True
                    connection['connectingParam'].connectionsDict[connectionName]['initialized'] = True

    #Vectorize based on parameterization and update coordinates
    def updateCoords(self):
        coordinates = np.zeros((self.N,3))
        for paramName in np.unique(self.paramMap):
            param = self.embeddedParams[paramName]
            coordinateIndices = np.where(np.array(self.paramMap)==paramName)[0]
            psiEtaZeta = self.parameterization[coordinateIndices]

            #Get args for various connections
            connectionArgs = {}
            for connectionName in self.connectionsDict:
                connection = self.connectionsDict[connectionName]
                if connection['type'] in [0] and 'counts_'+paramName in connection:
                    connectionArgs['countsDuplicates'] = connection['counts_'+paramName][coordinateIndices]
                    connectionArgs['totalCountsDuplicates'] = connection['totalCounts'][coordinateIndices]

            psiEtaZeta[:,2] = param.calcZeta(psiEtaZeta, **connectionArgs)
            coordinates[coordinateIndices] = param.calcCoords(psiEtaZeta)
        self.coordinates = coordinates
        fig = plt.figure()
        plotAx = fig.add_subplot(1, 1, 1, projection='3d')
        plotAx.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
        plt.show()

    #Fit parameters for all parameterizations used by this pointset
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
    '''
    Class for managing single CST parameterization object
    '''

    def __init__(self, param):
        self.param = param
        self.coeffs = self.param.getCoeffs()
        self.dependantCoeffs = OrderedDict()
        self.connectionsDict = OrderedDict()
        self.masks = self.param.masks

    #This is unnecessary with dafoam DV's
    def addConstraint(self, name, IVEmbeddedParam, IVList, DVList):
        '''Constrain specific coefficients to match another parameterization
           IV/DV\'s are lists of indices for coefficients, check cst3d.py for positions
           IV/DV lists must be same length'''
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

    def calcXYZ2Zeta(self, coordinates):
        return self.param.calcXYZ2Zeta(coordinates)

    def calcParameterization(self, coordinates):
        return self.param.coords2PsiEtaZeta(np.atleast_2d(coordinates))

    def calcCoords(self, psiEtaZeta):
        coords = self.param.calcCoords(np.atleast_2d(psiEtaZeta))
        return coords

    def calcZeta(self, psiEtaZeta, **kwargs):
        zetaVals = self.param.calcZeta(np.atleast_2d(psiEtaZeta))

        for connectionName in self.connectionsDict:
            connection = self.connectionsDict[connectionName]

            if connection['type'] in [0]:
                ##dn = f1-(f1-f2)*(n-1)/(N-1)
                countsDuplicates = kwargs['countsDuplicates'] if 'countsDuplicates' in kwargs else np.ones_like(zetaVals)
                totalCountsDuplicates = kwargs['totalCountsDuplicates'] if 'totalCountsDuplicates' in kwargs else np.ones_like(zetaVals)

                #Find duplicate psi,eta values
                psiEtaVals = psiEtaZeta[:,0:1]
                uniquePsiEta, uniquePsiEtaIndices, countsPsiEta = np.unique(psiEtaVals, return_index=True, return_counts=True, axis=0)
                duplicatePsiEtaIndex = [i for i in range(len(psiEtaZeta)) if i not in uniquePsiEtaIndices]
                duplicatePsiEtaZeta = psiEtaZeta[duplicatePsiEtaIndex]

                duplicateZeta1 = zetaVals[duplicatePsiEtaIndex]
                duplicateZeta2 = connection['connectingParam'].param.calcZeta(np.atleast_2d(duplicatePsiEtaZeta))

                #Scale zeta values for duplicate psi,eta
                zetaVals[duplicatePsiEtaIndex] = (duplicateZeta1 - (duplicateZeta1 - duplicateZeta2) *
                                                  (countsDuplicates[duplicatePsiEtaIndex] - 1) /
                                                  (totalCountsDuplicates[duplicatePsiEtaIndex] - 1))

        return zetaVals

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