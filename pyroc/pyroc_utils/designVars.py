import numpy as np

class TkDesignVar():
    def __init__(self, name, value, lower, upper, showSens, mask=False):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.mask = mask
        self.value = float(value.get()) if mask else value
        self.showSens = showSens

    def getValue(self):
        if self.mask!=0:
            return self.value
        else:
            return float(self.value.get())

class GlobalDesignVar():
    def __init__(self, name, value, function, lower=None, upper=None, scale=1.0, config=None):
        self.name = name
        self.value = np.atleast_1d(np.array(value)).astype("d")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.config = config
        self.nVal = len(self.value)

    def __call__(self, geo, config):
        #Apply value when called
        if (self.config is None or config is None or any(c0 == config for c0 in self.config)):
            return self.function(np.real(self.value), geo)

class CSTLocalDesignVar():
    def __init__(self, name, value, ind, lower=None, upper=None, scale=1.0, config=None):
        self.name = name
        self.value = value
        self.ind = ind
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.config = config
        self.nVal = len(self.value)

    def __call__(self, coeffs, config):
        #Apply value when called
        if (self.config is None or config is None or any(c0 == config for c0 in self.config)):
            for _ in range(self.nVal):
                coeffs[self.ind[_,0]][self.ind[_,1]] = self.value[_]

    def apply(self, coeffs):
        for _ in range(self.nVal):
            self.value[_] = coeffs[self.ind[_,0]][self.ind[_,1]]