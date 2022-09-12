import os

class Reader(object):
    #Baseclass for readers
    def __init__(self, casePath=None):
        self.rootDir = casePath
        pass

    def readCase(self):
        pass

    def readLog(self, filepath):
        pass