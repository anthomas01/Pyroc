from .cst3d import *
import os

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

    def attachPoints(self, coordinates, ptSetName):
        pass

    def getAttachedPoints(self):
        pass

    def updateCoeffs(self):
        pass

    def calcdPtdCoef(self, ptSetName):
        pass