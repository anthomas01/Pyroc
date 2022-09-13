from .DVGeometry import *
from ...pyroc_utils import *

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
        self._finalize()

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

    