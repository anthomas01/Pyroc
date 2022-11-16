import numpy as np

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
    return np.random.randint(0,1,n)*2-1

def closestPoint(v, point, axis=None):
    '''
    Find the closest point(s) in v to given point
    Returns index of closest point(s) in v
    Returns multiple if equally close
    '''
    dim = len(np.shape(v))
    ax = dim-1 if axis is None else axis
    if dim in [1]:
        d2 = np.power(v-point,2)
    elif dim in [2]:
        d2 = np.sum(np.power(v-point,2),axis=ax)
    #Add if needed
    return np.where(d2==np.min(d2))[0]

def backwardDifference(x,y):
    return np.append([0],(y[1:]-y[:-1])/(x[1:]-x[:-1]))

def forwardDifference(x,y):
    return np.append((y[1:]-y[:-1])/(x[1:]-x[:-1]),[0])

def centralDifference(x,y):
    ctr = np.append([0],(y[2:]-y[:-2])/(x[2:]-x[:-2]))
    return np.append(ctr,[0])

def compositeDifference(x,y):
    dydx = forwardDifference(x,y)
    dydx[-1] = backwardDifference(x,y)[-1]
    dydx[1:-1] = centralDifference(x,y)[1:-1]
    return dydx