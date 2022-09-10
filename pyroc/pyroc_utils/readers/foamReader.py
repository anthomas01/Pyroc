import os
from .readers import Reader

class FoamRead(Reader):
    #Reader for OpenFOAM Cases
    def __init__(self, casePath=None):
        super().__init__(casePath=casePath)

    def readCase(self):
        pass

    def readLog(self, filepath):
        if (os.path.exists(filepath)):
            file = open(filepath, 'r')
            lns = file.readlines()
            nLns = len(lns)

            for _ in range(nLns):
                ln = lns[_]
                #Check Line For Condition
                #Store Data
                pass