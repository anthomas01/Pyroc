

def isScalar(var):
    if hasattr(var, '__len__'):
        return False
    return True