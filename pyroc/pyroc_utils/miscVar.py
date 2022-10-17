from numpy import random

def isScalar(var):
    if hasattr(var, '__len__'):
        return False
    return True

def positiveOrNegative(n):
    '''
    Returns random 1 or -1
    
    Parameters:
    -----------------------

    n: int
    - number of integers
    '''
    return random.randint(0,1,n)*2-1
    