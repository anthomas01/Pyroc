class DesignVar():
    def __init__(self, name, value, lower, upper, mask=0):
        self.name = name
        self.value = value
        self.lower = lower
        self.upper = upper
        self.mask = mask

    def getValue(self):
        return float(self.value.get())