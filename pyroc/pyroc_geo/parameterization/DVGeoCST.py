from .DVGeometry import *
from ...pyroc_utils import *
from scipy import sparse

class DVGeometryCST(DVGeometry):
    #Baseclass for manipulating CST geometry

    def __init__(self, CSTMultiParam, name=None):
        super().__init__(filepath=None, name=name)

        self.param = CSTMultiParam
        self.origParamCoef = self.param.coef.copy()

    def addPointSet(self, points, ptName, origConfig=True, **kwargs):
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

        # DVCon artifact?
        kwargs.pop("compNames", None)

        self.ptSetNames.append(ptName)
        self.zeroJacobians([ptName])

        points = np.array(points).real.astype('d')
        self.points[ptName] = points
        self.nPts[ptName] = len(points.flatten())

        # Ensure we project into the undeformed geometry
        if origConfig:
            tmpCoef = self.param.coef.copy()
            self.param.coef = self.origParamCoef
            self.param.updateCoeffs()

        self.param.attachPoints(self.points[ptName], ptName)

        if origConfig:
            self.param.coef = tmpCoef
            self.param.updateCoeffs()

        self.param.calcdPtdCoef(ptName)
        self.updated[ptName] = False

    def addLocalDV(self, dvName, coefDict, lower=None, upper=None, scale=1.0, config=None):
        """
        Add one or more local design variables ot the DVGeometry
        object. Local variables are used for small shape modifications.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

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
        if self.name is not None:
            dvName = self.name + "_" + dvName

        if isinstance(config, str):
            config = [config]

        nVal = sum([len(coefDict[_]) for _ in coefDict])
        values = np.zeros(nVal)
        ind = np.zeros((nVal, 2), dtype=int) # paramNameIndex, index
        paramNames = list(self.param.embeddedParams.keys())

        i=0
        for dvParamName in coefDict:
            indices = coefDict[dvParamName]
            for index in indices:
                ind[i,0] = paramNames.index(dvParamName)
                ind[i,1] = index
                values[i] = self.param.embeddedParams[dvParamName].coeffs[index]
                i+=1

        self.DV_listLocal[dvName] = CSTLocalDesignVar(dvName, values, ind, lower, upper, scale, config)

        return self.DV_listLocal[dvName].nVal
    
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

        self.curPtSet = ptSetName
        # We've postponed things as long as we can...do the finalization.
        self.finalize()

        # Apply Global DVs
        for key in self.DV_listGlobal:
            self.DV_listGlobal[key](self, config)

        # Now add in the local DVs
        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.param.coef, config)

        # Update all coef
        self.param.updateCoeffs()

        # Evaluate coordinates
        Xfinal = self.param.getAttachedPoints(ptSetName)

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return Xfinal

    def computeTotalJacobian(self, ptSetName, config=None):
        """Return the total point jacobian in CSR format since we
        need this for TACS"""
    
        # Finalize the object, if not done yet
        self.finalize()
        self.curPtSet = ptSetName

        if self.JT[ptSetName] is not None:
            return

        if self.nPts[ptSetName] is None:
            coords0 = self.update(ptSetName, config=config).flatten()
            self.nPts[ptSetName] = len(coords0.flatten())

        NDV = self.getNDV()
        if NDV > 0:
            dvCounter = 0
            totalJac = np.zeros((NDV, self.nPts[ptSetName]))

            # Update dPtdGlobal
            dPtdGlobal = self.computeGlobalJacobian(ptSetName, config)
            if dPtdGlobal is not None:
                dvCounter = self.getNDVGlobal()
                totalJac[:dvCounter,:] = dPtdGlobal

            # Update dPtdLocal
            self.param.calcdPtdCoef(ptSetName)
            dPtdLocal = self.param.embeddedSurfaces[ptSetName].dPtdCoef.T
            if dPtdLocal is not None:
                totalJac[dvCounter:,:] = dPtdLocal
        
            self.JT[ptSetName] = totalJac.tocsr()
            self.JT[ptSetName].sort_indices()

        else:
            self.JT[ptSetName] = None

    ### Internal Functions

    def finalize(self):
        super().finalize()
        self.nCoefFull = sum([len(_) for _ in self.param.coef])