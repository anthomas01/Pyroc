from ...pyroc_utils import *
from collections import OrderedDict

class DVGeometry(object):
    #Baseclass for manipulating geometry

    def __init__(self, name=None):
        self.DV_listGlobal = OrderedDict()  # Global Design Variable List
        self.DV_listLocal = OrderedDict()  # Local Design Variable List

        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.ptSetNames = []
        self.name = name

        self.JT = {}
        self.nPts = {}

    def addPointSet(self, points, ptName, origConfig=True):
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed
        ptName : str
            A user supplied name to associate with the set of
            coordinates. This name will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.
        origConfig : bool
            Flag determine if the coordinates are projected into the
            undeformed or deformed configuration. This should almost
            always be True except in circumstances when the user knows
            exactly what they are doing.
        """
        pass

    def setDesignVars(self, dvDict):
        """Set design variables from dict"""

        for key in dvDict:
            #Global DVs
            if key in self.DV_listGlobal:
                vals_to_set = np.atleast_1d(dvDict[key]).astype("D")
                if len(vals_to_set) != self.DV_listGlobal[key].nVal:
                    raise Exception(f"Incorrect number of design variables for DV: {key}.\n" + 
                                    f"Expecting {self.DV_listGlobal[key].nVal} variables but received {len(vals_to_set)}")
                self.DV_listGlobal[key].value = vals_to_set

            #Local DVs
            if key in self.DV_listLocal:
                vals_to_set = np.atleast_1d(dvDict[key]).astype("D")
                if len(vals_to_set) != self.DV_listLocal[key].nVal:
                    raise Exception(f"Incorrect number of design variables for DV: {key}.\n" + 
                                    f"Expecting {self.DV_listLocal[key].nVal} variables but received {len(vals_to_set)}")
                self.DV_listLocal[key].value = vals_to_set

            # Jacobians are, in general, no longer up to date
            self.zeroJacobians(self.ptSetNames)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

    def zeroJacobians(self, ptSetNames):
        """
        set stored jacobians to None for ptSetNames

        Parameters
        ----------
        ptSetNames : list
            list of ptSetNames to zero the jacobians.
        """
        for name in ptSetNames:
            self.JT[name] = None  # J is no longer up to date

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to :func:`setDesignVars`

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """

        dvDict = {}
        for key in self.DV_listGlobal:
            dvDict[key] = self.DV_listGlobal[key].value

        # and now the local DVs
        for key in self.DV_listLocal:
            dvDict[key] = self.DV_listLocal[key].value

        return dvDict

    def update(self, ptSetName, config=None):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.

        """
        pass

    def convertSensitivityToDict(self, dIdx, out1D=False):
        """
        This function takes the result of totalSensitivity and
        converts it to a dict for use in pyOptSparse

        Parameters
        ----------
        dIdx : array
           Flattened array of length getNDV(). Generally it comes from
           a call to totalSensitivity()

        out1D : boolean
            If true, creates a 1D array in the dictionary instead of 2D.
            This function is used in the matrix-vector product calculation.

        Returns
        -------
        dIdxDict : dictionary
           Dictionary of the same information keyed by this object's
           design variables
        """

        i=0
        dIdxDict = {}
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i : i + dv.nVal]
            i += dv.nVal

        i = 0
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i : i + dv.nVal]

            i += dv.nVal

        return dIdxDict

    def convertDictToSensitivity(self, dIdxDict):
        """
        This function performs the reverse operation of
        convertSensitivityToDict(); it transforms the dictionary back
        into an array. This function is important for the matrix-free
        interface.

        Parameters
        ----------
        dIdxDict : dictionary
           Dictionary of information keyed by this object's
           design variables

        Returns
        -------
        dIdx : array
           Flattened array of length getNDV().
        """
        dIdx = np.zeros(self.nDV_T, self.dtype)
        i = 0
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        i = 0
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        return dIdx

    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Parameters
        ----------

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        names = list(self.DV_listGlobal.keys())
        names.extend(list(self.DV_listLocal.keys()))
        return names

    def totalSensitivityProd(self, vec, ptSetName, config=None):
        pass

    def totalSensitivityTransProd(self, vec, ptSetName, config=None):
        pass

    def computeDVJacobian(self, config=None):
        pass

    def computeTotalJacobian(self, ptSetName, config=None):
        pass

    def computeTotalJacobianCS(self, ptSetName, config=None):
        pass

    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True, ignoreVars=None, freezeVars=None):
        pass

    def writeTecplot(self, fileName, solutionTime=None):
        pass

    def writePointSet(self, name, fileName, solutionTime=None):
        pass

    def writePlot3d(self, fileName):
        pass

    def writeSTL(self, fileName):
        pass

    def updatePyGeo(self, geo, outputType, fileName, nRefU=0, nRefV=0):
        pass

    #Internal Functions

    def _finalize(self):
        pass
