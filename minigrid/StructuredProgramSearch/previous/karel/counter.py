class KarelExecuteCounter:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.m = max_steps

    def count_down(self):
        self.m -= 1

    def zero(self):
        self.m = 1

    def terminated(self):
        return self.m <= 1
    
    def total_steps(self):
        return self.max_steps - self.m
