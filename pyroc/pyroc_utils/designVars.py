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
        self.value = value
        self.function = function
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.config = config
        self.nVal = len(value)

    def __call__(self, geo, config):
        #Apply value when called
        if (self.config is None or config is None or any(c0 == config for c0 in self.config)):
            return self.function(np.real(self.value), geo)

class CSTLocalDesignVar():
    def __init__(self, name, value, lower=None, upper=None, scale=1.0, mask=None, config=None):
        self.name = name
        self.value = np.copy(value)
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.mask = mask
        self.config = config
        self.nVal = len(value)

    def __call__(self, config):
        #Apply value when called
        if (self.config is None or config is None or any(c0 == config for c0 in self.config)):
            return self.value