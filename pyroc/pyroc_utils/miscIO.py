import os
import re
import numpy as np

#Functions
def copyDir(old,new):
    if os.name=='nt':
        dirCopy = 'Xcopy /e /i /y '
        os.system(dirCopy + old.replace('/','\\') + ' ' + new.replace('/','\\'))
    else:
        dirCopy = 'cp -r '
        os.system(dirCopy + old + ' ' + new)
    return 0
    
def copyFile(old,new):
    if os.name=='nt':
        fileCopy = 'copy /v /y '
        os.system(fileCopy + old.replace('/','\\') + ' ' + new.replace('/','\\'))
    else:
        fileCopy = 'cp '
        os.system(fileCopy + old + ' ' + new)
    return 0
    
def delDir(rm):
    if os.name=='nt':
        dirDel = 'rmdir /s /q '
        os.system(dirDel + rm.replace('/','\\'))
    else:
        dirDel = 'rm -r '
        os.system(dirDel + rm)
    return 0
    
def delFile(rm):
    if os.name=='nt':
        fileDel = 'del /f /q '
        os.system(fileDel + rm.replace('/','\\'))
    else:
        fileDel = 'rm '
        os.system(fileDel + rm)
    return 0

def getFloats(line):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    out = rx.findall(line)
    return np.array([float(i) for i in out])

