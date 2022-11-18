from .cst3d import *
from collections import OrderedDict
from scipy.spatial import Delaunay
import numpy as np
import os

#TODO Move connections to separate file

class CSTMultiParam(object):
    #Class for managing multiple CST geometry objects

    def __init__(self):

        self.coef = None
        self.embeddedSurfaces = OrderedDict()
        self.embeddedParams = OrderedDict()
        #Refit when adding new coords?
        self.fit = False
        self.fitIter = 2
        self.sigFigs = 16 # Sig figs for fitting connections

        #self.mapper

    def attachPoints(self, coordinates, ptSetName, embeddedParams=None, parameterization=False):

        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        self.embeddedSurfaces[ptSetName] = EmbeddedSurface(coordinates, 
                                                           embeddedParams, 
                                                           fit=self.fit, 
                                                           fitIter=self.fitIter, 
                                                           parameterization=parameterization, 
                                                           sig=self.sigFigs)

    def attachParameterization(self, parameterization, ptSetName, embeddedParams=None):
        #Project points to surface/CST geom object if some were passed in
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        self.embeddedSurfaces[ptSetName] = EmbeddedSurface(parameterization, embeddedParams, self.fit)
        if self.fit: self.setCoeffs()

    def getAttachedPoints(self, ptSetName):
        if ptSetName in self.embeddedSurfaces:
            #Refresh attached points
            self.embeddedSurfaces[ptSetName].updateCoords()
            #return coordinates
            return self.embeddedSurfaces[ptSetName].coordinates
        return None

    def attachCSTParam(self, CSTParam, paramName, embeddedParams=None):
        embeddedParams = embeddedParams if embeddedParams is not None else self.embeddedParams
        embeddedParams[paramName] = EmbeddedParameterization(CSTParam)

    def updateCoeffs(self):
        "Update CST coefficients"
        i = 0
        for paramName in self.embeddedParams:
            self.embeddedParams[paramName].coeffs = np.real(self.coef[i])
            self.embeddedParams[paramName].updateCoeffs()
            i += 1

    def setCoeffs(self):
        "Update internal coefficients stored in multi from CST coefficients"
        self.coef = []
        for paramName in self.embeddedParams:
            paramCoeffs = self.embeddedParams[paramName].coeffs
            self.coef.append(paramCoeffs)

    #Credit: Moral support from Nolan Jeffrey Glock
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
            self.embeddedParams[connectingParam1].connectionsDict[connectionName] = {
                'type': 0,
                'connectingParam': self.embeddedParams[connectingParam2],
                'linear': kwargs['linear'] if 'linear' in kwargs else False,
            }
            self.embeddedParams[connectingParam2].connectionsDict[connectionName] = {
                'type': 0,
                'connectingParam': self.embeddedParams[connectingParam1],
                'linear': kwargs['linear'] if 'linear' in kwargs else False,
            }

        else:
            raise Exception('Connection type %d not implemented' % connectionType)

    def calcdPtdCoef(self, ptSetName):
        if ptSetName in self.embeddedSurfaces:
            embeddedSurface = self.embeddedSurfaces[ptSetName]
            embeddedSurface.updatedPtdCoef()

    def outputCoordinates(self, file, nx=100, ny=1):
        '''
        Print nx*ny coordinates for each embeddedParam, evenly spaced across psi,eta

        Parameters:

        file - str
            file path
        nx - int
            Number of values in psi direction
        ny - int
            Number of values in eta direction
        '''
        psiVals = np.linspace(0.0, 1.0, nx)
        etaVals = np.linspace(0.0, 1.0, ny)
        psi, eta = np.meshgrid(psiVals, etaVals)
        psi, eta = psi.flatten(), eta.flatten()
        psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T

        f = open(file, 'w')
        for paramName in self.embeddedParams:
            param = self.embeddedParams[paramName]
            param.param.setPsiEtaZeta(psiEtaZeta.copy())
            coords = param.param.updateCoords()
            for _ in range(len(coords)):
                f.write(' '.join(coords[_,:]) + '\n')
        f.close()

    def plotMulti(self, ax, nx=100, ny=10):
        psiVals = np.linspace(0, 1, nx)
        etaVals = np.linspace(0, 1, ny)
        psi, eta = np.meshgrid(psiVals, etaVals)
        psi, eta = psi.flatten(), eta.flatten()
        psiEtaZeta = np.vstack([psi, eta, np.zeros_like(psi)]).T
        tri = Delaunay(np.array([psi, eta]).T)

        for paramName in self.embeddedParams:
            param = self.embeddedParams[paramName]
            param.param.setPsiEtaZeta(psiEtaZeta.copy())
            coords = param.param.updateCoords()
            ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2], triangles=tri.simplices.copy())

    def fitMulti(self, file, fitIter=2):
        #Project points to surface/CST geom object (csv input)
        #TODO multiple file types
        if (os.path.exists(file)):
            f = open(file,'r')
            lines = f.readlines()
            coordinates = []

            #Determine file type, read file
            extension = file.split('.')[-1]
            #Move file reading somewhere common to other classes
            if extension in ['csv']:
                for line in lines:
                    coordinates.append([float(_) for _ in line.split(',')])
            if extension in ['stl']:
                for line in lines:
                    splitLine = line.split()
                    if 'vertex' in splitLine:
                        coordinates.append([float(_) for _ in splitLine[-3:]])
                coordinates = np.unique(coordinates,axis=0)
            f.close()
            
            self.embeddedSurfaces['fit'] = EmbeddedSurface(np.atleast_2d(coordinates), self.embeddedParams, fit=True, fitIter=fitIter)
            self.setCoeffs()
        else:
            raise Exception('%s does not exist' % file)



class EmbeddedSurface(object):
    #Class for managing multiple CST surfaces

    def __init__(self, coordinates, embeddedParams, fit=False, fitIter=1, parameterization=False, sig=16):
        self.embeddedParams = embeddedParams
        self.N = len(coordinates)
        self.connectionsDict = OrderedDict()
        self.dPtdCoef = None

        if self.N>0:

            if parameterization:
                #Read in parameterization, ndarray (N,3)
                self.parameterization = np.copy(coordinates)

                #Determine which embedded parameterization object each coordinate belongs to
                self.paramMap = self.mapParameterization2Params(self.parameterization) #list

                self.coordinates = np.copy(self.parameterization)

                self.initializeParameterizationConnections(sig)

                self.updateCoords()

            else:
                #Read in coords, ndarray (N,3)
                self.coordinates = np.copy(coordinates)

                for _ in range(fitIter):
                    #Determine which embedded parameterization object each coordinate belongs to
                    self.paramMap = self.mapCoords2Params() #list

                    #Group coordinates by parameterization and apply fit if True
                    self.parameterization = self.fitParams(fit)

                self.initializeParameterizationConnections(sig)

        else:
            
            self.coordinates = np.copy(coordinates)
            
    
    #TODO FIXME
    #Move mapping to seperate class?
    #The love algorithm
    def mapCoords2Params(self):
        #Returns array that is the same size as the coordinate list
        #Each indice contains the name of an EmbeddedParameterization object

        # FIXME
        # Needs better initial guess
        # Will not work if initial guess creates invalid psiEtaZeta
        # Will not work if final surfaces are very very close?

        paramNames = list(self.embeddedParams.keys())
        paramMap = [paramNames[0] for _ in range(self.N)]
        #Mapping only necessary when there are multiple params
        if len(paramNames)>1:
            coordinates = np.copy(self.coordinates)
            #Find surfaces where certain principles are not violated?
            #Get better initial guess (modified k-means?)
            #Least euclidean squares?
            for paramName in paramNames:
                param = self.embeddedParams[paramName]
                psiEtaZeta = param.calcParameterization(self.coordinates)
                param_psiEtaZeta = np.copy(psiEtaZeta)
                param_psiEtaZeta[:,2] = param.calcZeta(param_psiEtaZeta)
                distSquared = np.sum(np.power(psiEtaZeta-param_psiEtaZeta,2),axis=1)

                if paramName == paramNames[0]:
                    closest = distSquared
                else:
                    for i in range(self.N):
                        if distSquared[i]<closest[i]:
                            paramMap[i] = paramName
                            closest[i] = distSquared[i]

        # objectiveDict = OrderedDict()
        # for paramName in self.embeddedParams:
        #     param = self.embeddedParams[paramName]
        #     #Get parameterization coordinates
        #     psiEtaZeta = param.calcParameterization(self.coordinates)
        #     param_psiEtaZeta = np.copy(psiEtaZeta)
        #     param_psiEtaZeta[:,2] = param.calcZeta(param_psiEtaZeta)
        #     distSquared = np.sum(np.power(psiEtaZeta-param_psiEtaZeta,2),axis=1)
        #     #Get parameterization gradient
        #     param_gradZeta = param.param.calcParamsGrad(param_psiEtaZeta)
        #     objectiveDict[paramName] = distSquared
            
        # for i in range(self.N):
        #     closest = objectiveDict[paramNames[0]][i]
        #     for paramName in paramNames[1:]:
        #         iObj = objectiveDict[paramName][i]
        #         if iObj<closest:
        #             paramMap[i] = paramName
        #             closest = iObj

        if '' in paramMap:
            raise Exception('One or more points were not fit')

        return paramMap

    #TODO FIXME
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

    #TODO Move to connections
    def initializeParameterizationConnections(self, decTol=8):
        '''
        Determines connection parameters for point set
        '''
        for paramName in self.embeddedParams:
            embeddedParam = self.embeddedParams[paramName]

            for connectionName in embeddedParam.connectionsDict:
                connection = embeddedParam.connectionsDict[connectionName]
                connectionFound = connectionName in self.connectionsDict

                if connection['type'] in [0] and not (connectionFound and self.connectionsDict[connectionName]['initialized']):
                    #dn = f1-(f1-f2)*(n-1)/(N-1)
                    #Need to calculate N and n map for duplicate indices
                    dtypeCounts = int if connection['linear'] else float

                    #Find duplicate psi,eta values for entire point set
                    psiEtaVals = np.round(self.parameterization[:,0:2],decTol)
                    uniquePsiEta, uniquePsiEtaIndices, countsPsiEta = np.unique(psiEtaVals, return_index=True, return_counts=True, axis=0)
                    uniqueDuplicatePsiEta = uniquePsiEta[countsPsiEta>1]

                    #Initialize N and n arrays, this is the main calculation goal here
                    totalCountsDuplicates = np.ones(len(self.parameterization), dtype=int)
                    countsDuplicates = np.ones(len(self.parameterization), dtype=dtypeCounts)

                    #totalCountsDuplicates
                    for _ in range(len(uniquePsiEta)):
                        #Find all indices where this unique pair of psi and eta is in the larger parameterization array
                        psiEta = uniquePsiEta[_,:]
                        thisPsiIndices = np.where(psiEtaVals[:,0]==psiEta[0])[0]
                        thisEtaIndices = np.where(psiEtaVals[:,1]==psiEta[1])[0]
                        thisPsiEtaIndices = [i for i in thisPsiIndices if i in thisEtaIndices]

                        #Set N for each index
                        totalCountsDuplicates[thisPsiEtaIndices] = countsPsiEta[_]

                    #countsDuplicates
                    for _ in range(len(uniqueDuplicatePsiEta)):
                        #Find all indices where this unique pair of psi and eta is in the larger parameterization array
                        psiEta = uniqueDuplicatePsiEta[_]
                        thisPsiIndices = np.where(psiEtaVals[:,0]==psiEta[0])[0]
                        thisEtaIndices = np.where(psiEtaVals[:,1]==psiEta[1])[0]
                        thisPsiEtaIndices = [i for i in thisPsiIndices if i in thisEtaIndices]

                        psiEtaZeta = self.parameterization[thisPsiEtaIndices]
                        boundZeta1 = embeddedParam.param.calcZeta(np.atleast_2d(psiEtaZeta[0]))[0]
                        boundZeta2 = connection['connectingParam'].param.calcZeta(np.atleast_2d(psiEtaZeta[0]))[0]

                        #Catch points initialized outside boundary
                        if (boundZeta1 == boundZeta2) and totalCountsDuplicates[thisPsiEtaIndices[0]]>1:
                            raise Exception('Duplicate points while zeta bounds are equal')
                        elif boundZeta1 > boundZeta2:
                            _boundZeta1 = np.max(psiEtaZeta[:,2])
                            _boundZeta2 = np.min(psiEtaZeta[:,2])
                        else:
                            _boundZeta1 = np.min(psiEtaZeta[:,2])
                            _boundZeta2 = np.max(psiEtaZeta[:,2])

                        thisTotalCounts = totalCountsDuplicates[thisPsiEtaIndices]
                        n1 = 1.0 + (thisTotalCounts - 1.0) * (_boundZeta1 - psiEtaZeta[:,2]) / (_boundZeta1 - _boundZeta2)

                        #Set countsDuplicates
                        #Interpolate if necessary
                        if connection['linear']:
                            # Need to find order of points while keeping bounds as 1 and N
                            #Set boundaries
                            firstInd = closestPoint(psiEtaZeta[:,2],boundZeta1)[0]
                            countsDuplicates[thisPsiEtaIndices[firstInd]] = 1
                            lastInd = closestPoint(psiEtaZeta[:,2],boundZeta2)[0]
                            countsDuplicates[thisPsiEtaIndices[lastInd]] = thisTotalCounts[lastInd]
                            #Set others
                            if len(thisPsiEtaIndices)>2:
                                middleInd = [i for i in range(len(thisPsiEtaIndices)) if i not in [firstInd, lastInd]]
                                for __ in range(len(middleInd)):
                                    ind = middleInd[np.where(np.sort(n1[middleInd])==n1[middleInd[__]])[0][0]]
                                    countsDuplicates[thisPsiEtaIndices[ind]] = __+2
                        else:
                            #Keep track of exact point placement
                            countsDuplicates[thisPsiEtaIndices] = n1

                    #Store needed information
                    connectingParamName = [name for name in self.embeddedParams if self.embeddedParams[name]==connection['connectingParam']][0]
                    self.connectionsDict[connectionName] = {
                        'type' : connection['type'],
                        'counts_'+paramName : countsDuplicates,
                        'counts_'+connectingParamName : 1 + totalCountsDuplicates - countsDuplicates,
                        'totalCounts' : totalCountsDuplicates,
                        'initialized' : True
                    }

    #Vectorize based on parameterization and update coordinates
    def updateCoords(self):
        if self.N>0:
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

    #TODO Make this more efficient
    def updatedPtdCoef(self):
        #find nCoef and coefMap
        nCoef = 0
        coefMap = {}

        if self.N>0:
            for paramName in self.embeddedParams:
                paramCoeffs = self.embeddedParams[paramName].coeffs
                coefMap[paramName] = [_+nCoef for _ in range(len(paramCoeffs))]
                nCoef += len(paramCoeffs)

            #size of dPtdCoef will be 3*Npts x Ncoef
            dPtdCoef = np.zeros((3*self.N, nCoef))

            for paramName in np.unique(self.paramMap):
                param = self.embeddedParams[paramName]
                coordinateIndices = np.where(np.array(self.paramMap)==paramName)[0]
                coordinates = self.coordinates[coordinateIndices]

                #Get args for various connections
                connectionArgs = {}
                for connectionName in self.connectionsDict:
                    connection = self.connectionsDict[connectionName]
                    if connection['type'] in [0] and 'counts_'+paramName in connection:
                        connectionArgs['countsDuplicates'] = connection['counts_'+paramName][coordinateIndices]
                        connectionArgs['totalCountsDuplicates'] = connection['totalCounts'][coordinateIndices]

                #Calculate partial dPtdCoef
                partialdPtdCoef = param.calcdPtdCoef(coordinates, **connectionArgs)
                #Insert into dPtdCoef based on coefficient mapping
                #TODO Speed this up
                for i in range(len(coordinateIndices)):
                    ind = coordinateIndices[i]
                    for j in range(len(coefMap[paramName])):
                        coef = coefMap[paramName][j]
                        dPtdCoef[3*ind:3*(ind+1), coef] = partialdPtdCoef[3*i:3*(i+1), j]

                #Set derivatives for connections not covered by partial dPtdCoef jacobians
                for connectionName in self.connectionsDict:
                    paramConnection = param.connectionsDict[connectionName]
                    connection = self.connectionsDict[connectionName]
                    if connection['type'] in [0] and 'counts_'+paramName in connection:
                        #Get index info
                        connectingParamName = [name for name in self.embeddedParams if self.embeddedParams[name]==paramConnection['connectingParam']][0]
                        connectingCoordinateIndices = np.where(np.array(self.paramMap)==connectingParamName)[0]
                        connectingCountsDuplicates = connection['counts_'+paramName][connectingCoordinateIndices]
                        connectingTotalCountsDuplicates = connection['totalCounts'][connectingCoordinateIndices]
                        totalDuplicatePsiEtaIndices = np.where(connectingTotalCountsDuplicates>1)[0]
                        duplicateConnectingCoordinateIndices = connectingCoordinateIndices[totalDuplicatePsiEtaIndices]

                        if len(duplicateConnectingCoordinateIndices)>0:
                            #Get derivatives
                            duplicatePsiEtaZeta = self.parameterization[duplicateConnectingCoordinateIndices]
                            duplicatePsiEtaZeta[:,2] = param.param.calcZeta(np.atleast_2d(duplicatePsiEtaZeta)) #Unscaled zeta
                            jac = param.param.calcParamsJacobian(np.atleast_2d(duplicatePsiEtaZeta))

                            #Calculate offset
                            paramOffset = np.zeros_like(jac)
                            for i in range(len(coefMap[paramName])):
                                paramOffset[2::3,i] = (jac[2::3,i] * (1 - connectingCountsDuplicates[totalDuplicatePsiEtaIndices]) /
                                                          (connectingTotalCountsDuplicates[totalDuplicatePsiEtaIndices] - 1))

                            #Calculate connection derivative
                            indices = np.sort(np.array([[3*_+__ for __ in range(3)] for _ in duplicateConnectingCoordinateIndices]).flatten())
                            connectionJac = param.param.calcJacobian(np.atleast_2d(self.coordinates[duplicateConnectingCoordinateIndices,:]),
                                                                 paramOffset=paramOffset)
                            for i in range(len(coefMap[paramName])):
                                dPtdCoef[indices,coefMap[paramName][i]] = connectionJac[:,i]

            self.dPtdCoef = dPtdCoef

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
                coords = np.array(paramCoords[paramName])
                param.fitCoefficients(coords)

        #Update parameterization
        parameterization = np.zeros_like(self.coordinates)
        for paramName in uniqueParams:
            param = self.embeddedParams[paramName]
            coordinateIndices = np.where(np.array(self.paramMap)==paramName)[0]
            parameterization[coordinateIndices] = param.calcParameterization(self.coordinates[coordinateIndices])

        return parameterization



class EmbeddedParameterization(object):
    '''
    Class for managing single CST parameterization object
    '''

    def __init__(self, param):
        self.param = param
        self.coeffs = np.array(self.param.getCoeffs())
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
        zetaOffset = np.zeros(len(psiEtaZeta))

        for connectionName in self.connectionsDict:
            connection = self.connectionsDict[connectionName]

            if connection['type'] in [0]:
                ##dn = f1-(f1-f2)*(n-1)/(N-1)
                countsDuplicates = kwargs['countsDuplicates'] if 'countsDuplicates' in kwargs else np.ones(len(psiEtaZeta))
                totalCountsDuplicates = kwargs['totalCountsDuplicates'] if 'totalCountsDuplicates' in kwargs else np.ones(len(psiEtaZeta))
                #Find duplicate psi,eta values
                totalDuplicatePsiEtaIndices = np.where(totalCountsDuplicates>1)[0]

                if len(totalDuplicatePsiEtaIndices)>0:
                    duplicatePsiEtaZeta = psiEtaZeta[totalDuplicatePsiEtaIndices]
                    duplicateZeta1 = self.param.calcZeta(np.atleast_2d(duplicatePsiEtaZeta))
                    duplicateZeta2 = connection['connectingParam'].param.calcZeta(np.atleast_2d(duplicatePsiEtaZeta))

                    #Scale zeta values for duplicate psi,eta
                    zetaOffset[totalDuplicatePsiEtaIndices] = ((duplicateZeta2 - duplicateZeta1) * 
                                                               (countsDuplicates[totalDuplicatePsiEtaIndices] - 1) /
                                                               (totalCountsDuplicates[totalDuplicatePsiEtaIndices] - 1))

        zetaVals = self.param.calcZeta(np.atleast_2d(psiEtaZeta), zetaOffset=zetaOffset)
        return zetaVals

    def calcdPtdCoef(self, coordinates, **kwargs):
        nCoeff = len(self.param.getCoeffs())
        paramOffset = np.zeros((len(coordinates.flatten()),nCoeff))

        for connectionName in self.connectionsDict:
            connection = self.connectionsDict[connectionName]

            if connection['type'] in [0]:
                countsDuplicates = kwargs['countsDuplicates'] if 'countsDuplicates' in kwargs else np.ones(len(coordinates))
                totalCountsDuplicates = kwargs['totalCountsDuplicates'] if 'totalCountsDuplicates' in kwargs else np.ones(len(coordinates))

                #Find duplicate psi,eta values
                totalDuplicatePsiEtaIndices = np.where(totalCountsDuplicates>1)[0]

                if len(totalDuplicatePsiEtaIndices)>0:
                    duplicatePsiEtaZeta = self.calcParameterization(coordinates)[totalDuplicatePsiEtaIndices]
                    duplicatePsiEtaZeta[:,2] = self.param.calcZeta(np.atleast_2d(duplicatePsiEtaZeta)) #Unscaled zeta
                    jac = self.param.calcParamsJacobian(np.atleast_2d(duplicatePsiEtaZeta))

                    for _ in range(nCoeff):
                        paramOffset[2+3*totalDuplicatePsiEtaIndices,_] = (jac[2::3,_] * (1 - countsDuplicates[totalDuplicatePsiEtaIndices]) /
                                                                          (totalCountsDuplicates[totalDuplicatePsiEtaIndices] - 1))

        dPtdCoef = self.param.calcJacobian(np.atleast_2d(coordinates), paramOffset=paramOffset)
        return dPtdCoef

    def applyMasks(self):
        for dependantCoeffName in self.dependantCoeffs:
            dependantCoeff = self.dependantCoeffs[dependantCoeffName]
            for _ in dependantCoeff['DVList']:
                self.param.masks[_] = 1

    def removeMasks(self):
        #Reapply original masks
        self.param.masks = self.masks