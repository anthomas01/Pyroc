from .DVGeometry import *
from ...pyroc_utils import *
from scipy import sparse

class DVGeometryCST(DVGeometry):
    #Baseclass for manipulating CST geometry

    def __init__(self, filepath, name=None):
        super().__init__(name)

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
            # We have a slight problem...dPtdCoef only has the shape
            # functions, so it size Npt x Coef. We need a matrix of
            # size 3*Npt x 3*nCoef, where each non-zero entry of
            # dPtdCoef is replaced by value * 3x3 Identity matrix.

            # Extract IJV Triplet from dPtdCoef
            row = dPtdCoef.row
            col = dPtdCoef.col
            data = dPtdCoef.data

            new_row = np.zeros(3 * len(row), "int")
            new_col = np.zeros(3 * len(row), "int")
            new_data = np.zeros(3 * len(row))

            # Loop over each entry and expand:
            for j in range(3):
                new_data[j::3] = data
                new_row[j::3] = row * 3 + j
                new_col[j::3] = col * 3 + j

            # Size of New Matrix:
            Nrow = dPtdCoef.shape[0] * 3
            Ncol = dPtdCoef.shape[1] * 3

            # Create new matrix in coo-dinate format and convert to csr
            new_dPtdCoef = sparse.coo_matrix((new_data, (new_row, new_col)), shape=(Nrow, Ncol)).tocsr()

            # Do Sparse Mat-Mat multiplication and resort indices
            if J_temp is not None:
                self.JT[ptSetName] = (J_temp.T * new_dPtdCoef.T).tocsr()
                self.JT[ptSetName].sort_indices()
        else:
            self.JT[ptSetName] = None

    #Internal Functions

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True
        self.nPtAttachFull = len(self.param.coef)