class TkDesignVar():
    def __init__(self, name, value, lower, upper, showSens, mask=0):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.mask = mask
        self.value = float(value.get()) if mask!=0 else value
        self.showSens = showSens

    def getValue(self):
        if self.mask!=0:
            return self.value
        else:
            return float(self.value.get())