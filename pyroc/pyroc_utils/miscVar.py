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

def uniqueVector(v, axis=0):
    '''
    Returns unique slices of Nd array
    i.e. vectors of 2d array
    '''
    output = []
    shape = np.shape(v)
    dim = len(shape)
    if axis<dim:
        for _ in range(shape[axis]):
            if axis in [0] and v[_] not in output:
                output.append(v[_])
            elif axis in [1] and v[:,_] not in output:
                output.append(v[:,_])
            elif axis in [2] and v[:,:,_] not in output:
                output.append(v[:,:,_])
            #Add if needed
    else:
        raise Exception('Axis %d out of range for array with dim=%d' % (axis, dim))

    return np.array(output)