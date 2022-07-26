class DesignVar():
    def __init__(self, name, value, lower, upper):
        self.name = name
        self.value = value
        self.lower = lower
        self.upper = upper

    def getValue(self):
        return float(self.value.get())