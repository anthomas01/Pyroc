from collections import OrderedDict

class BaseData(object):
    def __init__(self, casePath=None):
        self.rootDir = casePath
        pass

    def storeData(self):
        pass

    def retrieveData(self):
        pass

class BaseFoam(BaseData):
    def __init__(self, casePath=None):
        super().__init__(casePath)
        pass

class BaseCFD(BaseData):
    def __init__(self):
        self.iterations = OrderedDict()
        pass

class FoamCFD(BaseFoam, BaseCFD):
    def __init__(self,casePath):
        super(BaseFoam,self).__init__(casePath)
        pass

    def getOptions(self):
        #Create instance of FOAMOPTION for this case setup
        pass