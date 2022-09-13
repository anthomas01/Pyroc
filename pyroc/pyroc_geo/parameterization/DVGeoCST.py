from .DVGeometry import *
from ...pyroc_utils import *
from scipy import sparse

class DVGeometryCST(DVGeometry):
    #Baseclass for manipulating CST geometry

    def __init__(self, filepath, name=None):
        super().__init__(filepath, name)

        self.param = CSTMultiParam(filepath)
        self.origParamCoef = self.param.coef.copy()

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

        self.ptSetNames.append(ptName)
        self.zeroJacobians([ptName])
        self.nPts[ptName] = None

        points = np.array(points).real.astype("d")
        self.points[ptName] = points

        # TODO Ensure we project into the undeformed geometry
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

        #TODO
        #Determine how to set local design variables
        #self.DV_listLocal[dvName] = CSTLocalDesignVar()

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

        # Set all coef Values back to initial values
        #self.param.coef = self.origParamCoef.copy()
        #self._setInitialValues()

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

        # compute the derivatives of the coefficients of this level wrt all of the design
        # variables at this level and all levels above
        J_temp = self.computeDVJacobian(config=config)

        # now get the derivative of the points for this level wrt the coefficients(dPtdCoef)
        if self.param.embeddedSurfaces[ptSetName].dPtdCoef is not None:
            dPtdCoef = self.param.embeddedSurfaces[ptSetName].dPtdCoef.tocoo()

            # Do Sparse Mat-Mat multiplication and resort indices
            if J_temp is not None:
                self.JT[ptSetName] = (J_temp.T * dPtdCoef.T).tocsr()
                self.JT[ptSetName].sort_indices()
        else:
            self.JT[ptSetName] = None

    #Internal Functions

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True
        self.nPtAttachFull = len(self.param.coef)

    def localDVJacobian(self, config=None):
        """
        Return the derivative of the coefficients wrt the local design
        variables
        """

        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros
        nDV = self.getNDVLocal()
        self.getDVOffsets()

        if nDV != 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull * 3, self.nDV_T))

            iDVLocal = self.nDVL_count
            for key in self.DV_listLocal:
                if (
                    self.DV_listLocal[key].config is None
                    or config is None
                    or any(c0 == config for c0 in self.DV_listLocal[key].config)
                ):

                    self.DV_listLocal[key](self.param.coef, config)
                    nVal = self.DV_listLocal[key].nVal
                    for j in range(nVal):
                        pt_dv = self.DV_listLocal[key].coefList[j]
                        irow = pt_dv[0] * 3 + pt_dv[1]
                        Jacobian[irow, iDVLocal] = 1.0
                        iDVLocal += 1

                else:
                    iDVLocal += self.DV_listLocal[key].nVal
        else:
            Jacobian = None
        return Jacobian