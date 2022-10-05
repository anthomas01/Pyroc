from ...pyroc_utils import *
from pyspline import *
from collections import OrderedDict
from mpi4py import MPI
from scipy import sparse

class DVGeometry(object):
    #Baseclass for manipulating geometry

    def __init__(self, filepath, name=None):
        self.DV_listGlobal = OrderedDict()  # Global Design Variable List
        self.DV_listLocal = OrderedDict()  # Local Design Variable List

        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.ptSetNames = []
        self.name = name

        self.finalized = False
        self.dtype = "d"

        self.JT = {}
        self.nPts = {}

        # derivative counters for offsets
        self.nDV_T = None  # total number of design variables
        self.nDVG_T = None
        self.nDVL_T = None

        self.useComposite = False

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

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update its pointset or not.
        Essentially what happens is when update() is called with a point set, the self.updated dict entry for pointSet is flagged as true.
        Here we just return that flag. When design variables are set, we then reset all the flags to False since,
        when DVs are set, nothing (in general) will be up to date anymore.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to check.
        """
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            return True
    
    def addGlobalDV(self, dvName, value, func, lower=None, upper=None, scale=1.0, config=None):
        """
        Add a global design variable to the DVGeometry object

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

        value : float, or iterable list of floats
            The starting value(s) for the design variable. This
            parameter may be a single variable or a numpy array
            (or list) if the function requires more than one
            variable. The number of variables is determined by the
            rank (and if rank ==1, the length) of this parameter.

        lower : float, or iterable list of floats
            The lower bound(s) for the variable(s). A single variable
            is permissable even if an array is given for value. However,
            if an array is given for 'lower', it must be the same length
            as 'value'

        func : python function
            The python function handle that will be used to apply the
            design variable

        upper : float, or iterable list of floats
            The upper bound(s) for the variable(s). Same restrictions as
            'lower'

        scale : float, or iterable list of floats
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        """
        # if the parent DVGeometry object has a name attribute, prepend it
        if self.name is not None:
            dvName = self.name + "_" + dvName

        if isinstance(config, str):
            config = [config]
        self.DV_listGlobal[dvName] = GlobalDesignVar(dvName, value, func, lower, upper, scale, config)

    def addLocalDV(self, dvName, value=None, lower=None, upper=None, scale=1.0, config=None):
        """
        Add one or more local design variables ot the DVGeometry
        object. Local variables are used for small shape modifications.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

        value : float
            Initial value for the DV, None will automatically calculate

        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables

        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : float
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.

        Returns
        -------
        N : int
            The number of design variables added.

        Example
        --------
        >>> # 
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

        # compute the various DV offsets
        DVCountGlobal, DVCountLocal = self.getDVOffsets()

        i = DVCountGlobal
        dIdxDict = {}
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i : i + dv.nVal]
            i += dv.nVal

        i = DVCountLocal
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
        # compute the various DV offsets
        DVCountGlobal, DVCountLocal = self.getDVOffsets()
        dIdx = np.zeros(self.nDV_T, self.dtype)

        i = DVCountGlobal
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        i = DVCountLocal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        return dIdx

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Parameters
        ----------

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        if not pyOptSparse or not self.useComposite:
            names = list(self.DV_listGlobal.keys())
            names.extend(list(self.DV_listLocal.keys()))
        else:
            names = [self.DVComposite.name]

        return names

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.


        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse

        """

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]

        # generate the total Jacobian self.JT
        self.computeTotalJacobian(ptSetName, config=config)

        # now that we have self.JT compute the Mat-Mat multiplication
        nDV = self.getNDV()
        dIdx_local = np.zeros((N, nDV), "d")
        for i in range(N):
            if self.JT[ptSetName] is not None:
                dIdx_local[i, :] = self.JT[ptSetName].dot(dIdpt[i, :, :].flatten())

        if comm:  # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdx = self.convertSensitivityToDict(dIdx)

        return dIdx

    def totalSensitivityProd(self, vec, ptSetName, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \times\mathrm{vec}`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """

        self.computeTotalJacobian(ptSetName, config=config)  # This computes and updates self.JT

        names = self.getVarNames()
        newvec = np.zeros(self.getNDV(), self.dtype)

        i = 0
        for vecKey in vec:
            # check if the seed DV is actually a design variable for the DVGeo object
            if vecKey not in names:
                raise Exception(f"{vecKey} is not a design variable, the full list is:{names}")

        # perform the product
        if self.JT[ptSetName] is None:
            xsdot = np.zeros((0, 3))
        else:
            xsdot = self.JT[ptSetName].T.dot(newvec)
            xsdot = np.reshape(xsdot, (len(xsdot)//3, 3))
        return xsdot

    def totalSensitivityTransProd(self, vec, ptSetName, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \times\mathrm{vec}`

        This is useful for reverse AD mode.

        Parameters
        ----------
        vec : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
        """

        self.computeTotalJacobian(ptSetName, config=config)

        # perform the product
        if self.JT[ptSetName] is None:
            xsdot = np.zeros((0, 3))
        else:
            xsdot = self.JT[ptSetName].dot(np.ravel(vec))

        # Pack result into dictionary
        xsdict = {}
        names = self.getVarNames()
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            elif key in self.DV_listLocal:
                dv = self.DV_listLocal[key]
            xsdict[key] = xsdot[i : i + dv.nVal]
            i += dv.nVal
        return xsdict

    def computeDVJacobian(self, config=None):
        """
        return J_temp for a given config
        """

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        J_local = self.localDVJacobian(config=config)

        J_temp = None

        if J_local is not None:
            J_temp = sparse.lil_matrix(J_local)

        return J_temp

    def computeTotalJacobian(self, ptSetName, config=None):
        """Return the total point jacobian in CSR format since we
        need this for TACS"""
        pass

    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True, ignoreVars=None, freezeVars=None):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added

        globalVars : bool
            Flag specifying whether global variables are to be added

        localVars : bool
            Flag specifying whether local variables are to be added

        ignoreVars : list of strings
            List of design variables the user DOESN'T want to use
            as optimization variables.

        freezeVars : list of string
            List of design variables the user WANTS to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.
        """
        if ignoreVars is None:
            ignoreVars = set()
        if freezeVars is None:
            freezeVars = set()

        # Add design variables from the master:
        varLists = OrderedDict(
            [
                ("globalVars", self.DV_listGlobal),
                ("localVars", self.DV_listLocal),
            ]
        )

        for lst in varLists:
            if (lst == "globalVars" and globalVars or lst == "localVars" and localVars):
                for key in varLists[lst]:
                    if key not in ignoreVars:
                        dv = varLists[lst][key]
                        if key not in freezeVars:
                            optProb.addVarGroup(
                                dv.name, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale
                            )
                        else:
                            optProb.addVarGroup(
                                dv.name, dv.nVal, "c", value=dv.value, lower=dv.value, upper=dv.value, scale=dv.scale
                            )

    def writeTecplot(self, name, fileName, solutionTime=None, config=None):
        """
        Write a given point set to a tecplot file

        Parameters
        ----------
        name : str
             The name of the point set to write to a file

        fileName : str
           Filename for tecplot file. Should have no extension, an
           extension will be added

        SolutionTime : float
            Solution time to write to the file. This could be a fictitious time to
            make visualization easier in tecplot.

        Config : str or list
            Config for which to update coordinates
        """

        coords = self.update(name, config)
        fileName = fileName + "_%s.dat" % name
        f = openTecplot(fileName, 3)
        writeTecplot1D(f, name, coords, solutionTime)
        closeTecplot(f)

    def writePlot3d(self, fileName):
        pass

    def writeSTL(self, fileName):
        pass

    #Helper Functions

    def finalize(self):
        pass

    def getNDV(self):
        """
        Return the actual number of design variables, global + local
        """
        return self.getNDVGlobal() + self.getNDVLocal()

    def getNDVGlobal(self):
        """
        Get total number of global variables
        """
        nDV = 0
        for key in self.DV_listGlobal:
            nDV += self.DV_listGlobal[key].nVal

        return nDV

    def getNDVLocal(self):
        """
        Get total number of local variables
        """
        nDV = 0
        for key in self.DV_listLocal:
            nDV += self.DV_listLocal[key].nVal

        return nDV

    def getDVOffsets(self):
        """
        return the global and local DV offsets for this FFD
        """

        # figure out the split between local and global Variables
        # All global vars at all levels come first
        # then spanwise, then section local vars and then local vars.
        # Parent Vars come before child Vars

        # get the global and local DV numbers on the parents if we don't have them
        if (self.nDV_T is None or self.nDVG_T is None or self.nDVL_T is None):
            self.nDV_T = self.getNDV()
            self.nDVG_T = self.getNDVGlobal()
            self.nDVL_T = self.getNDVLocal()
            self.nDVG_count = 0
            self.nDVL_count = self.nDVG_T

        return self.nDVG_count, self.nDVL_count

    def localDVJacobian(self, config=None):
        pass

    def computeTotalJacobianFD(self, ptSetName, config=None):
        """This function takes the total derivative of an objective,
        I, with respect the points controlled on this processor using FD.
        We take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. Note that this function is slow
        and should eventually be replaced by an analytic version.
        """

        self.finalize()
        self.curPtSet = ptSetName

        if self.JT[ptSetName] is not None:
            return

        coords0 = self.update(ptSetName, config=config).flatten()

        if self.nPts[ptSetName] is None:
            self.nPts[ptSetName] = len(coords0.flatten())

        DVGlobalCount, DVLocalCount = self.getDVOffsets()

        h = 1e-6

        self.JT[ptSetName] = np.zeros([self.nDV_T, self.nPts[ptSetName]])

        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName, config=config).flatten()

                deriv = (coordsph - coords0) / h
                self.JT[ptSetName][DVGlobalCount, :] = deriv

                DVGlobalCount += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName, config=config).flatten()

                deriv = (coordsph - coords0) / h
                self.JT[ptSetName][DVLocalCount, :] = deriv

                DVLocalCount += 1
                self.DV_listLocal[key].value[j] = refVal

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        for dg in self.DV_listGlobal:
            print("%s" % (self.DV_listGlobal[dg].name))
            for i in range(self.DV_listGlobal[dg].nVal):
                print("%20.15f" % (self.DV_listGlobal[dg].value[i]))

        for dl in self.DV_listLocal:
            print("%s" % (self.DV_listLocal[dl].name))
            for i in range(self.DV_listLocal[dl].nVal):
                print("%20.15f" % (self.DV_listLocal[dl].value[i]))
