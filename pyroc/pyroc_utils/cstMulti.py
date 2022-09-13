from collections import OrderedDict
from .cst3d import *
import os
from scipy import sparse

class CSTMultiParam(object):
    #Class for managing CST surfaces

    def __init__(self, filepath):
        
        if os.path.exists(filepath):
            self.filepath = filepath
        else:
            print("Path does not exist - '%s'" % filepath)
            self.filepath = None

        self.coef = None
        self.embeddedSurfaces = {}

        self.readCSTInput(filepath)

    def readCSTInput(self):
        pass

    def attachPoints(self, coordinates, ptSetName):
        pass

    def getAttachedPoints(self):
        pass

    def updateCoeffs(self):
        "Update CST coefficients"
        pass

    def setCoeffs(self):
        "Update internal coefficients stored in multi from CST coefficients"
        pass

    def calcdPtdCoef(self):
        pass

    def getBounds(self):
        """Determine the extents of the set of volumes

        Returns
        -------
        xMin : array of length 3
            Lower corner of the bounding box
        xMax : array of length 3
            Upper corner of the bounding box
        """
        pass


class EmbeddedSurface(object):
    #Class 
    def __init__(self):
        pass